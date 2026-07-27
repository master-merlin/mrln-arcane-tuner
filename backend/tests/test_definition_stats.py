"""Tests for the data-calibrated estimation wall.

Covers the config cost model, history → coefficient aggregation, the estimate
payload (calibrated + zero-data fallback), and on-disk backfill recovery.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest

from app.core.db.engine import get_db
from app.core.stats import backfill, definition_stats_service
from app.engine.utils import cost_model as cm

MB = 1024 * 1024


def _ref_config(**overrides):
    cfg = {
        "max_train_steps": 1000,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "resolutions": [1024],
        "gradient_checkpointing": True,
        "quantization": "none",
        "save_precision": "fp16",
        "network_rank": 16,
    }
    cfg.update(overrides)
    return cfg


def _insert_job(definition_id: str, **fields) -> str:
    """Insert a completed job_history row with controlled cost fields."""
    job_id = str(uuid.uuid4())
    config = fields.pop("config", _ref_config())
    row = {
        "id": job_id,
        "definition_id": definition_id,
        "status": "completed",
        "config": json.dumps(config),
        "created_at": time.time(),
    }
    row.update(fields)
    cols = ", ".join(row.keys())
    placeholders = ", ".join("?" for _ in row)
    with get_db().write() as conn:
        conn.execute(
            f"INSERT INTO job_history ({cols}) VALUES ({placeholders})",
            list(row.values()),
        )
    return job_id


# ── Cost model ──────────────────────────────────────────────────────────


def test_reference_config_costs_unity():
    assert cm.step_time_cost(_ref_config()) == pytest.approx(1.0)


def test_defaults_match_legacy_heuristics():
    cfg = _ref_config()
    # 1.2 s/step × 1000 steps
    assert cm.wall_time_seconds(cfg, cm.DEFAULT_TIME_COEFF) == pytest.approx(1200.0)
    # 14 MB/rank × 16
    assert cm.lora_bytes(cfg, cm.DEFAULT_BYTES_PER_RANK) / MB == pytest.approx(224.0)


def test_step_cost_scales_with_resolution_and_batch():
    cfg = _ref_config(resolutions=[2048], train_batch_size=2)
    # (2048/1024)^2 × batch 2 = 8
    assert cm.step_time_cost(cfg) == pytest.approx(8.0)


def test_save_precision_doubles_output():
    fp16 = cm.lora_bytes(_ref_config(save_precision="fp16"), cm.DEFAULT_BYTES_PER_RANK)
    fp32 = cm.lora_bytes(_ref_config(save_precision="fp32"), cm.DEFAULT_BYTES_PER_RANK)
    assert fp32 == pytest.approx(fp16 * 2)


def test_normalize_time_roundtrip():
    assert cm.normalize_time(2.4, _ref_config()) == pytest.approx(2.4)


# ── Recompute + estimate ────────────────────────────────────────────────


def test_estimate_zero_data_falls_back_to_defaults():
    def_id = f"fresh_{uuid.uuid4().hex[:8]}"
    est = definition_stats_service.estimate(def_id, _ref_config())

    assert est["stats_available"] is False
    assert est["samples"] == 0
    assert est["wall_time"]["calibrated"] is False
    # Default coeff → legacy 1.2 s/step
    assert est["wall_time"]["seconds"] == pytest.approx(1200, abs=1)
    assert est["output_size"]["calibrated"] is False


def test_recompute_then_estimate_is_calibrated():
    def_id = f"calib_{uuid.uuid4().hex[:8]}"
    # A run that took 2.0 s/step at the reference config, 100 MB LoRA at rank 16.
    _insert_job(
        def_id,
        avg_step_time=2.0,
        final_lora_size_bytes=100 * MB,
        completed_steps=1000,
        total_run_bytes=600 * MB,
        config=_ref_config(),
    )

    summary = definition_stats_service.recompute(def_id)
    assert summary.get(def_id) == 1

    est = definition_stats_service.estimate(def_id, _ref_config())
    assert est["stats_available"] is True
    assert est["samples"] == 1
    # 2.0 s/step × 1000 steps = 2000 s (not the 1200 s default)
    assert est["wall_time"]["seconds"] == pytest.approx(2000, abs=2)
    assert est["wall_time"]["calibrated"] is True
    # 100 MB at rank 16 → scales linearly to rank 32
    est32 = definition_stats_service.estimate(def_id, _ref_config(network_rank=32))
    assert est32["output_size"]["bytes"] == pytest.approx(200 * MB, rel=0.01)
    # Disk overhead 600/100 = 6× the final LoRA
    assert est["disk_footprint"]["bytes"] == pytest.approx(600 * MB, rel=0.02)


def test_recompute_uses_median_across_runs():
    def_id = f"median_{uuid.uuid4().hex[:8]}"
    for sec in (1.0, 2.0, 9.0):  # median = 2.0
        _insert_job(def_id, avg_step_time=sec, completed_steps=1000, config=_ref_config())
    definition_stats_service.recompute(def_id)
    est = definition_stats_service.estimate(def_id, _ref_config())
    assert est["samples"] == 3
    assert est["wall_time"]["seconds"] == pytest.approx(2000, abs=2)


def test_estimate_scales_for_different_config_than_history():
    def_id = f"scale_{uuid.uuid4().hex[:8]}"
    # Trained at 1024px, 2 s/step. Estimate at 2048px should be ~4× slower.
    _insert_job(def_id, avg_step_time=2.0, completed_steps=1000, config=_ref_config())
    definition_stats_service.recompute(def_id)
    est = definition_stats_service.estimate(def_id, _ref_config(resolutions=[2048]))
    assert est["wall_time"]["seconds"] == pytest.approx(8000, abs=4)


# ── Backfill ────────────────────────────────────────────────────────────


def test_backfill_recovers_fields_from_disk(tmp_path):
    def_id = f"bf_{uuid.uuid4().hex[:8]}"
    out_dir = tmp_path / "run_out"
    out_dir.mkdir()
    # A final LoRA file + a fat optimizer file → total > final.
    lora = out_dir / "my_lora_final.safetensors"
    lora.write_bytes(b"x" * (10 * MB))
    (out_dir / "optimizer.pt").write_bytes(b"y" * (40 * MB))
    (out_dir / "training_log.json").write_text(json.dumps({
        "lora_filename": "my_lora_final.safetensors",
        "lora_file_size_mb": 10.0,
        "elapsed_seconds": 1500.0,
        "step": 1000,
    }))

    job_id = _insert_job(
        def_id,
        output_dir=str(out_dir),
        completed_steps=1000,
        # Deliberately missing: final_lora_size_bytes, total_run_bytes, avg_step_time
        config=_ref_config(),
    )

    result = backfill.run_backfill()
    assert result["rows_updated"] >= 1

    conn = get_db().connection()
    rec = conn.execute(
        "SELECT * FROM job_history WHERE id = ?", (job_id,)
    ).fetchone()
    assert rec["final_lora_size_bytes"] == 10 * MB
    # 10 MB LoRA + 40 MB optimizer (+ the small training_log.json)
    assert rec["total_run_bytes"] == pytest.approx(50 * MB, abs=64 * 1024)
    assert rec["avg_step_time"] == pytest.approx(1.5)  # 1500 / 1000

    # And the definition is now calibrated.
    est = definition_stats_service.estimate(def_id, _ref_config())
    assert est["stats_available"] is True
    assert est["disk_footprint"]["bytes"] == pytest.approx(50 * MB, rel=0.05)


def test_backfill_skips_missing_output_dir():
    def_id = f"nodir_{uuid.uuid4().hex[:8]}"
    _insert_job(def_id, output_dir="Z:/does/not/exist", completed_steps=1000,
                config=_ref_config())
    # Should not raise even though the dir is gone.
    result = backfill.run_backfill()
    assert "completed_runs" in result


def test_backfill_persists_final_lora_file(tmp_path):
    """When the backfill locates the actual LoRA file it must persist the
    path too (final_lora_file), not just the size — on-disk availability
    in the stats modal depends on it."""
    def_id = f"bff_{uuid.uuid4().hex[:8]}"
    out_dir = tmp_path / "run_out"
    out_dir.mkdir()
    lora = out_dir / "my_lora_final.safetensors"
    lora.write_bytes(b"x" * (10 * MB))
    (out_dir / "training_log.json").write_text(json.dumps({
        "lora_filename": "my_lora_final.safetensors",
        "elapsed_seconds": 1500.0,
        "step": 1000,
    }))
    job_id = _insert_job(def_id, output_dir=str(out_dir),
                         completed_steps=1000, config=_ref_config())

    backfill.run_backfill()

    rec = get_db().connection().execute(
        "SELECT final_lora_file, final_lora_size_bytes FROM job_history WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert rec["final_lora_file"] == str(lora)
    assert rec["final_lora_size_bytes"] == 10 * MB


def test_backfill_size_only_fallback_persists_no_path(tmp_path):
    """The training_log mb-size fallback (file itself gone) records the size
    but must NOT invent a path."""
    def_id = f"bfmb_{uuid.uuid4().hex[:8]}"
    out_dir = tmp_path / "run_out_mb"
    out_dir.mkdir()
    (out_dir / "training_log.json").write_text(json.dumps({
        "lora_filename": "gone.safetensors",
        "lora_file_size_mb": 10.0,
    }))
    job_id = _insert_job(def_id, output_dir=str(out_dir),
                         completed_steps=1000, config=_ref_config())

    backfill.run_backfill()

    rec = get_db().connection().execute(
        "SELECT final_lora_file, final_lora_size_bytes FROM job_history WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert rec["final_lora_file"] is None
    assert rec["final_lora_size_bytes"] == 10 * MB


def test_backfill_sets_lora_on_disk_when_file_recovered(tmp_path):
    """W5.T9: the reconcile pass sets lora_on_disk=1 when it locates (or
    already has) a final_lora_file that verifiably exists on disk."""
    def_id = f"bfdisk_{uuid.uuid4().hex[:8]}"
    out_dir = tmp_path / "run_out_disk"
    out_dir.mkdir()
    lora = out_dir / "my_lora_final.safetensors"
    lora.write_bytes(b"x" * (10 * MB))
    (out_dir / "training_log.json").write_text(json.dumps({
        "lora_filename": "my_lora_final.safetensors",
        "elapsed_seconds": 1500.0,
        "step": 1000,
    }))
    job_id = _insert_job(def_id, output_dir=str(out_dir),
                         completed_steps=1000, config=_ref_config())

    backfill.run_backfill()

    rec = get_db().connection().execute(
        "SELECT lora_on_disk FROM job_history WHERE id = ?", (job_id,),
    ).fetchone()
    assert rec["lora_on_disk"] == 1


def test_backfill_self_heals_lora_on_disk_when_file_deleted(tmp_path):
    """A row already carries lora_on_disk=1 from a prior pass, but the user
    deleted the file outside the app since — the NEXT reconcile pass must
    flip it back to 0 (refreshed every pass, unlike the other recovered
    fields which are gated behind "missing")."""
    def_id = f"bfheal_{uuid.uuid4().hex[:8]}"
    out_dir = tmp_path / "run_out_heal"
    out_dir.mkdir()
    lora = out_dir / "will_be_deleted.safetensors"
    lora.write_bytes(b"x" * (5 * MB))
    job_id = _insert_job(
        def_id,
        output_dir=str(out_dir),
        completed_steps=1000,
        final_lora_file=str(lora),
        final_lora_size_bytes=5 * MB,
        lora_on_disk=1,
        config=_ref_config(),
    )

    lora.unlink()  # simulate the user deleting the LoRA outside the app
    backfill.run_backfill()

    rec = get_db().connection().execute(
        "SELECT lora_on_disk FROM job_history WHERE id = ?", (job_id,),
    ).fetchone()
    assert rec["lora_on_disk"] == 0


# ── Stats getter ────────────────────────────────────────────────────────


def test_get_reports_freshness():
    def_id = f"get_{uuid.uuid4().hex[:8]}"
    _insert_job(def_id, avg_step_time=1.0, completed_steps=1000, config=_ref_config())
    definition_stats_service.recompute(def_id)
    info = definition_stats_service.get(def_id)
    assert info["stats_available"] is True
    assert info["run_count"] == 1
    assert "time_coeff" in info["stats"]
    assert info["updated_at"] is not None


# ── Per-component VRAM calibration ──────────────────────────────────────


class _StubDefn:
    family = "sdxl"
    detected_precision: dict = {}
    architecture_params: dict = {}
    model_size_mb: dict = {}


def test_fit_uses_free_vram_not_total(monkeypatch):
    """A 24 GB card with 20 GB held by ComfyUI fits against the 4 GB FREE."""
    from types import SimpleNamespace

    from app.engine.utils.vram_estimator import VRAMEstimator

    fake = SimpleNamespace(gpus=[SimpleNamespace(vram_total_mb=24576, vram_used_mb=20480)])
    monkeypatch.setattr("app.core.system_monitor.system_monitor.snapshot", lambda: fake)

    report = VRAMEstimator.estimate(_StubDefn(), _ref_config())
    assert report.total_mb == 24576
    assert report.used_mb == 20480
    assert report.available_mb == 4096  # 24 − 20 GB free
    # Fit is decided against FREE VRAM, not the 24 GB card capacity.
    assert report.fits == (report.peak_mb < 4096 * 0.95)
    assert any("other processes" in w for w in report.warnings)


def test_per_component_vram_calibration(tmp_path, monkeypatch):
    from app.engine.utils.vram_estimator import VRAMEstimator

    defn = _StubDefn()
    monkeypatch.setattr(definition_stats_service, "_get_definition", lambda _id: defn)

    cfg = _ref_config()
    analytic = VRAMEstimator.estimate(defn, cfg).to_dict()
    assert analytic["model_weights_mb"] > 0  # sanity

    # Measured breakdown: model weights came in 2× the analytic guess; the
    # rest matched. The learned coefficient should isolate exactly that.
    measured = {f: analytic[f] for f in definition_stats_service.VRAM_COMPONENTS}
    measured["model_weights_mb"] = analytic["model_weights_mb"] * 2.0
    out_dir = tmp_path / "vrun"
    out_dir.mkdir()
    (out_dir / "training_log.json").write_text(json.dumps({"vram_measured": measured}))

    def_id = f"vram_{uuid.uuid4().hex[:8]}"
    _insert_job(def_id, output_dir=str(out_dir), avg_step_time=1.0,
                completed_steps=1000, config=cfg)

    definition_stats_service.recompute(def_id)
    vram_stats = definition_stats_service.get(def_id)["stats"]["vram"]
    assert vram_stats["model_weights_mb"]["value"] == pytest.approx(2.0, rel=0.01)
    assert vram_stats["overhead_mb"]["value"] == pytest.approx(1.0, rel=0.01)

    # Estimate applies the per-component multiplier and re-sums the peak.
    est = definition_stats_service.estimate(def_id, cfg)
    assert est["vram"]["calibrated"] is True
    assert "model_weights_mb" in est["vram"]["calibrated_components"]
    assert est["vram"]["model_weights_mb"] == pytest.approx(
        analytic["model_weights_mb"] * 2.0, rel=0.02
    )


# ── API plumbing (route registration + request/response shapes) ─────────


def test_api_estimate_endpoint(client):
    def_id = f"api_{uuid.uuid4().hex[:8]}"
    resp = client.post("/api/jobs/estimate", json={
        "definition_id": def_id,
        "config": _ref_config(),
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["stats_available"] is False
    assert body["wall_time"]["seconds"] == pytest.approx(1200, abs=2)
    for key in ("wall_time", "output_size", "throughput", "disk_footprint"):
        assert "display" in body[key]


def test_api_stats_recompute_and_get(client):
    def_id = f"apirc_{uuid.uuid4().hex[:8]}"
    _insert_job(def_id, avg_step_time=3.0, completed_steps=1000, config=_ref_config())

    resp = client.post("/api/jobs/stats/recompute")
    assert resp.status_code == 200
    assert "completed_runs" in resp.json()

    got = client.get(f"/api/jobs/stats/{def_id}")
    assert got.status_code == 200
    assert got.json()["run_count"] == 1

    # Estimate now reflects the calibrated 3 s/step.
    est = client.post("/api/jobs/estimate", json={
        "definition_id": def_id, "config": _ref_config(),
    }).json()
    assert est["wall_time"]["seconds"] == pytest.approx(3000, abs=3)
    assert est["stats_available"] is True
