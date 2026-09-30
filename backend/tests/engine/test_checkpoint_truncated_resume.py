"""LANE-133: resuming after a checkpoint file was truncated by a hard reset.

The incident: ``checkpoint-002500/optimizer.pt`` kept its full size but its
last 121 MB were zeros (a hard reset mid-save); resume surfaced a bare torch
zip error instead of falling back to the intact ``checkpoint-002250``.

Fixtures are REAL ``CheckpointManager`` saves of a tiny model.  "Truncated"
means the tail of a ``.pt`` is overwritten with zeros keeping the file size
(the incident's shape).  Symbols the fix introduces (``CheckpointUnreadable``,
``checkpoint_resumable``) are looked up with ``getattr`` inside the test body so
every test fails in its CALL phase, with a message starting ``LANE-133``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from structlog.testing import capture_logs

import app.engine.components.checkpoints as ckpt_mod
from app.core.job import JobStatus
from app.core.job_manager import JobManager
from app.engine.components.checkpoints import CheckpointManager


# ── Fixtures / helpers ───────────────────────────────────────────────────


def _save(run: Path, step: int, *, is_final: bool = False) -> Path:
    """One real CheckpointManager save (model.pt + optimizer.pt + state)."""
    model = nn.Linear(32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.randn(2, 32)).sum().backward()
    opt.step()  # Adam moments make optimizer.pt a real, sizeable zip
    CheckpointManager(str(run)).save_checkpoint(
        step,
        {"model": model},
        optimizer=opt,
        config={"lora_name": "t"},
        is_final=is_final,
    )
    return run / ("final" if is_final else f"checkpoint-{step:06d}")


def _truncate(path: Path) -> None:
    """Zero the tail of a file, keeping its size (the incident's shape)."""
    size = path.stat().st_size
    tail = max(1024, size // 4)
    with open(path, "r+b") as f:
        f.seek(size - tail)
        f.write(b"\x00" * tail)
    assert path.stat().st_size == size


def _truncate_optimizer(folder: Path) -> None:
    _truncate(folder / "optimizer.pt")


def _unreadable_cls():
    cls = getattr(ckpt_mod, "CheckpointUnreadable", None)
    assert cls is not None, "LANE-133: CheckpointUnreadable does not exist yet"
    return cls


def _resumable_fn():
    fn = getattr(ckpt_mod, "checkpoint_resumable", None)
    assert fn is not None, "LANE-133: checkpoint_resumable(folder) does not exist yet"
    return fn


def _job(tmp_path: Path):
    """A terminal job whose run dir is ``tmp_path/run`` (patched seam)."""
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    mgr = JobManager()
    job = mgr.create_job(
        "flux/dev",
        {"output_dir": str(tmp_path), "lora_name": "t", "definition_id": "flux/dev"},
    )
    job.status = JobStatus.STOPPED
    return mgr, job, run


class _Patched:
    """Patch the job manager's process/persistence edges (not the selection)."""

    def __init__(self, mgr, run):
        self._ps = [
            patch.object(mgr, "_get_job_output_dir", return_value=str(run)),
            patch.object(mgr, "start_job"),
            patch.object(mgr, "_stop_tailer"),
            patch.object(mgr, "_reset_job_log_state"),
            patch.object(mgr, "_persist_status"),
            patch.object(mgr, "_persist_config"),
        ]

    def __enter__(self):
        for p in self._ps:
            p.start()
        return self

    def __exit__(self, *a):
        for p in reversed(self._ps):
            p.stop()


def _resume(mgr, job, run, checkpoint_dir):
    """Run the real resume; return the folder name it resumed from."""
    with _Patched(mgr, run):
        mgr.resume_from_checkpoint(job.id, checkpoint_dir)
    return Path(job.config["resume_from_checkpoint"]).name


def _resume_or_fail(mgr, job, run, checkpoint_dir):
    """As ``_resume`` but a raw error (TypeError, bare RuntimeError) becomes a
    LANE-133 failure, so the detail carries the lane tag."""
    try:
        return _resume(mgr, job, run, checkpoint_dir)
    except Exception as e:  # noqa: BLE001 - reported with the lane tag
        pytest.fail(
            f"LANE-133: resume(checkpoint_dir={checkpoint_dir!r}) raised "
            f"{type(e).__name__}: {e}"
        )


def _expect_unreadable(fn, *must_contain: str) -> str:
    cls = _unreadable_cls()
    try:
        fn()
    except cls as e:
        msg = str(e)
    except Exception as e:  # noqa: BLE001
        pytest.fail(
            f"LANE-133: expected CheckpointUnreadable, got bare {type(e).__name__}: {e}"
        )
    else:
        pytest.fail("LANE-133: no CheckpointUnreadable raised for a damaged checkpoint")
    for needle in must_contain:
        assert needle in msg, f"LANE-133: error {msg!r} does not name {needle!r}"
    return msg


class _Killed(BaseException):
    """A hard reset: not an Exception, so no ``except Exception`` swallows it."""


def _kill_during_optimizer_write():
    """Patch torch.save so the optimizer write dies half-way, leaving a
    partial file at whatever path it was handed (the real incident's shape)."""
    real_save = torch.save

    def _save_then_die(obj, f, *a, **k):
        if "optimizer" in os.fspath(f):
            real_save(obj, f, *a, **k)
            _truncate(Path(os.fspath(f)))
            raise _Killed()
        return real_save(obj, f, *a, **k)

    return patch.object(torch, "save", _save_then_die)


# ── Selection ────────────────────────────────────────────────────────────


def test_auto_resume_skips_a_truncated_newest_checkpoint(tmp_path):
    mgr, job, run = _job(tmp_path)
    older = _save(run, 5)
    newer = _save(run, 10)
    _truncate_optimizer(newer)

    job.status = JobStatus.FAILED
    with (
        patch.object(mgr, "_get_job_output_dir", return_value=str(run)),
        patch.object(mgr, "_schedule_auto_resume") as sched,
        patch.object(mgr, "_persist_status"),
        capture_logs() as logs,
    ):
        mgr._maybe_auto_resume(job, "CUDA error: unknown error\ncudaErrorUnknown")

    assert sched.call_args is not None, "LANE-133: auto-resume scheduled nothing"
    assert sched.call_args.args == (job.id, older.name), (
        f"LANE-133: auto-resume picked {sched.call_args.args[1]!r} "
        f"(the truncated newest) instead of {older.name!r}"
    )
    warned = [
        e
        for e in logs
        if e.get("log_level") == "warning"
        and newer.name in repr(e)
        and "optimizer.pt" in repr(e)
    ]
    assert warned, (
        f"LANE-133: no warning names the skipped folder {newer.name} and its "
        f"reason; logs: {[e.get('event') for e in logs]}"
    )


def test_auto_resume_with_no_valid_checkpoint_refuses_by_name(tmp_path):
    mgr, job, run = _job(tmp_path)
    only = _save(run, 10)
    _truncate_optimizer(only)
    _expect_unreadable(
        lambda: _resume(mgr, job, run, None),
        only.name,
        "optimizer.pt",
        "no resumable checkpoint",
    )


def test_manual_resume_of_a_damaged_checkpoint_is_refused_by_name(tmp_path):
    mgr, job, run = _job(tmp_path)
    good = _save(run, 5)
    bad = _save(run, 10)
    _truncate_optimizer(bad)

    # No silent substitution: the damaged folder is refused, naming the file
    # and the newest resumable folder.
    _expect_unreadable(
        lambda: _resume(mgr, job, run, bad.name),
        "optimizer.pt",
        good.name,
    )
    assert "resume_from_checkpoint" not in job.config, (
        "LANE-133: a refused resume must not record a checkpoint on the job"
    )
    # An explicit healthy folder resumes as asked.
    assert _resume_or_fail(mgr, job, run, good.name) == good.name


def test_resume_without_a_folder_takes_the_newest_intact_one(tmp_path):
    mgr, job, run = _job(tmp_path)
    older = _save(run, 5)
    newer = _save(run, 10)
    _truncate_optimizer(newer)

    assert _resume_or_fail(mgr, job, run, None) == older.name

    # The HTTP endpoint with no checkpoint_dir in its body does the same.
    from fastapi.testclient import TestClient

    from app.main import app

    job.config.pop("resume_from_checkpoint", None)
    job.status = JobStatus.STOPPED
    with _Patched(mgr, run), patch("app.api.training.job_routes.job_manager", mgr):
        resp = TestClient(app).post(
            f"/api/jobs/{job.id}/resume-from-checkpoint", json={}
        )
    assert resp.status_code == 200, (
        f"LANE-133: POST resume-from-checkpoint with no checkpoint_dir -> "
        f"{resp.status_code} {resp.text[:200]}"
    )
    assert Path(job.config["resume_from_checkpoint"]).name == older.name, (
        "LANE-133: the endpoint did not resume from the newest intact folder"
    )


def test_selection_between_final_and_numbered_checkpoints(tmp_path):
    """``final/`` IS a resume source (``_RESUMABLE_DIR_RE`` admits it and the
    auto-resume scan ranks it above every numbered step).  A healthy final
    wins; a damaged final is skipped exactly like a damaged numbered folder,
    falling to the numbered sibling of the same final save."""
    mgr, job, run = _job(tmp_path)
    _save(run, 5)
    final = _save(run, 10, is_final=True)  # writes final/ AND checkpoint-000010/
    assert (run / "checkpoint-000010").is_dir()

    assert _resume_or_fail(mgr, job, run, None) == "final"

    # The first resume relaunched the job (PENDING); put it back to a stopped
    # job before resuming again, as the endpoint test above does.
    job.config.pop("resume_from_checkpoint", None)
    job.status = JobStatus.STOPPED
    _truncate_optimizer(final)
    picked = _resume_or_fail(mgr, job, run, None)
    assert picked == "checkpoint-000010", (
        f"LANE-133: with a truncated final/ the selection picked {picked!r}, "
        "expected the intact numbered sibling checkpoint-000010"
    )


# ── Listing ──────────────────────────────────────────────────────────────


def test_checkpoint_list_marks_each_artifact(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    good = _save(run, 5)
    bad = _save(run, 10)
    _truncate_optimizer(bad)
    (run / "t_000005.safetensors").write_bytes(b"a" * 8)
    (run / "t_000010.safetensors").write_bytes(b"b" * 8)

    from fastapi.testclient import TestClient

    from app.main import app

    with (
        patch("app.api.training.job_routes.job_manager") as jm,
        patch("app.api.training.job_routes._resolve_run_dir", return_value=run),
    ):
        jm.get_job.return_value = MagicMock()
        resp = TestClient(app).get("/api/jobs/job-1/checkpoints")
    assert resp.status_code == 200
    by_step = {c["step"]: c for c in resp.json()}

    assert by_step[5]["resumable"] is True
    assert by_step[5]["checkpoint_dir"] == good.name
    assert by_step[5].get("resumable_reason") is None
    assert by_step[10]["resumable"] is False, (
        "LANE-133: a checkpoint whose optimizer.pt is truncated is listed resumable"
    )
    reason = by_step[10].get("resumable_reason")
    assert isinstance(reason, str) and "optimizer.pt" in reason, (
        f"LANE-133: damaged artifact carries no resumable_reason naming the file: {reason!r}"
    )


# ── Load ─────────────────────────────────────────────────────────────────


def test_load_of_a_truncated_checkpoint_names_the_file(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    good = _save(run, 5)
    bad = _save(run, 10)
    _truncate_optimizer(bad)

    model = nn.Linear(32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    _expect_unreadable(
        lambda: CheckpointManager(str(run)).load_checkpoint(
            str(bad), components={"model": model}, optimizer=opt
        ),
        f"{bad.name}/optimizer.pt",
        "truncated or unreadable",
        "last good",
        good.name,
    )


# ── Saving ───────────────────────────────────────────────────────────────


def test_an_interrupted_save_is_never_resumable(tmp_path):
    mgr, job, run = _job(tmp_path)
    good = _save(run, 5)

    with _kill_during_optimizer_write(), pytest.raises(_Killed):
        _save(run, 10)

    assert not (run / "checkpoint-000010").exists(), (
        "LANE-133: an interrupted save left a final-named checkpoint-000010 folder"
    )
    leftovers = [p for p in run.iterdir() if p.is_dir() and p.name != good.name]
    resumable = _resumable_fn()
    for p in leftovers:
        assert ".staging-" in p.name, f"LANE-133: unexpected folder {p.name}"
        ok, reason = resumable(str(p))
        assert ok is False and reason, f"LANE-133: staging folder {p.name} is resumable"
    assert _resume_or_fail(mgr, job, run, None) == good.name


def test_an_interrupted_final_save_is_never_resumable(tmp_path):
    """Review finding 3.01: the final save path too.  An intact final/ from an
    earlier save must survive a later final save killed mid-write."""
    mgr, job, run = _job(tmp_path)
    _save(run, 5, is_final=True)  # intact final/ + checkpoint-000005/

    with _kill_during_optimizer_write(), pytest.raises(_Killed):
        _save(run, 10, is_final=True)

    final = run / "final"
    ok, reason = _resumable_fn()(str(final))
    assert ok is True, (
        f"LANE-133: final/ is damaged by an interrupted re-save ({reason}); the "
        "new save must be staged and never touch the old folder before the swap"
    )
    state = json.loads((final / "training_state.json").read_text())
    assert state["global_step"] == 5, "LANE-133: final/ is a mix of old and new saves"
    assert not (run / "checkpoint-000010").exists(), (
        "LANE-133: an interrupted final save left a final-named checkpoint-000010"
    )
    picked = _resume_or_fail(mgr, job, run, None)
    sd = torch.load(
        run / picked / "optimizer.pt", map_location="cpu", weights_only=True
    )
    assert sd["state"], f"LANE-133: selected {picked!r} has an empty optimizer state"


def test_a_save_over_an_existing_damaged_folder_replaces_it_whole(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    folder = _save(run, 10)
    _truncate_optimizer(folder)
    (folder / "stale_from_old_save.txt").write_text("old")

    _save(run, 10)  # same step, same final name

    assert not (folder / "stale_from_old_save.txt").exists(), (
        "LANE-133: a file from the old damaged folder survived the re-save"
    )
    siblings = sorted(p.name for p in run.iterdir())
    assert siblings == [folder.name, "training_log.json"] or siblings == [
        folder.name
    ], f"LANE-133: staging/replaced folders left behind: {siblings}"
    ok, reason = _resumable_fn()(str(folder))
    assert ok is True, f"LANE-133: re-saved folder does not validate: {reason}"
    manifest = json.loads((folder / "checkpoint_manifest.json").read_text())
    assert manifest.get("format") == 2 and "training_state.json" in manifest.get(
        "files", {}
    ), f"LANE-133: re-saved folder has no format-2 manifest: {manifest}"


# ── Validation helper ────────────────────────────────────────────────────


def _as_legacy(folder: Path) -> None:
    """Rewrite the manifest in today's unversioned shape: a flat size map."""
    manifest = json.loads((folder / "checkpoint_manifest.json").read_text())
    files = manifest.get("files", manifest)
    (folder / "checkpoint_manifest.json").write_text(json.dumps(files))


def test_a_legacy_checkpoint_stays_resumable(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    healthy = _save(run, 5)
    damaged = _save(run, 10)
    _as_legacy(healthy)
    _as_legacy(damaged)
    _truncate_optimizer(damaged)
    resumable = _resumable_fn()

    ok, reason = resumable(str(healthy))
    assert ok is True, f"LANE-133: a healthy legacy folder is refused: {reason}"

    # No manifest at all is legacy too.
    (healthy / "checkpoint_manifest.json").unlink()
    ok, reason = resumable(str(healthy))
    assert ok is True, f"LANE-133: a manifest-less legacy folder is refused: {reason}"

    ok, reason = resumable(str(damaged))
    assert ok is False and isinstance(reason, str) and "optimizer.pt" in reason, (
        f"LANE-133: a truncated legacy folder is resumable or has no reason: {ok} {reason!r}"
    )


def test_a_malformed_manifest_is_not_resumable(tmp_path):
    resumable = _resumable_fn()
    run = tmp_path / "run"
    run.mkdir()

    def fresh(step: int) -> Path:
        folder = _save(run, step)
        ok, reason = resumable(str(folder))
        assert ok is True, f"LANE-133: a fresh save is not resumable: {reason}"
        return folder

    def manifest_of(folder: Path) -> dict:
        return json.loads((folder / "checkpoint_manifest.json").read_text())

    def rewrite(folder: Path, data) -> None:
        (folder / "checkpoint_manifest.json").write_text(
            data if isinstance(data, str) else json.dumps(data)
        )

    cases: dict[str, Path] = {}

    f = fresh(1)
    rewrite(f, "{ not json")
    cases["unparseable"] = f

    f = fresh(2)
    m = manifest_of(f)
    m["files"]["optimizer.pt"] += 1
    rewrite(f, m)
    cases["size mismatch"] = f

    f = fresh(3)
    m = manifest_of(f)
    m["files"] = {}
    rewrite(f, m)
    cases["empty files"] = f

    f = fresh(4)
    m = manifest_of(f)
    m["files"].pop("training_state.json")
    rewrite(f, m)
    cases["omits training_state.json"] = f

    f = fresh(5)
    (f / "training_state.json").unlink()
    cases["training_state.json missing"] = f

    for name, folder in cases.items():
        ok, reason = resumable(str(folder))
        assert ok is False and isinstance(reason, str) and reason, (
            f"LANE-133: folder with {name} manifest is resumable: {ok} {reason!r}"
        )


@pytest.mark.parametrize("mark", [".staging-abc123", ".replaced-abc123"])
def test_a_complete_staging_or_replaced_folder_is_never_resumable(tmp_path, mark):
    """A fully intact folder under a staging/replaced name is still unfinished."""
    import shutil

    _mgr, _job_, run = _job(tmp_path)
    good = _save(run, 5)
    twin = run / f"{good.name}{mark}"
    shutil.copytree(good, twin)

    ok, reason = _resumable_fn()(str(twin))
    assert ok is False and reason, (
        f"LANE-133: intact {twin.name} reported resumable; its name marks an unfinished save"
    )
