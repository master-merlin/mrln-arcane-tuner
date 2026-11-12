"""minimax_h3_image against the three measured memory probes, and the refusal of the one that spilled.

Every measured number below is COPIED from the probe records of 2026-11-12 (never read from a file,
so the pin cannot drift with a re-run). Each probe was 20 steps on a 96 GB card, batch 1, the 1024
bucket, rank 16, gradient checkpointing on, the shipped definition otherwise:

* P1, int8, no block swap: PASS; torch peak allocated 34.32 GiB (35 139 MB), 1.48 s/step.
* P2, bf16, no block swap: PASS; torch peak allocated 64.14 GiB (65 684 MB), 0.97 s/step.
* P3, bf16, ``block_swap_config: {transformer_blocks: 24}`` (a PERCENT: 12 of 50 blocks): SPILLED.
  Shared GPU memory rose 21.07 GiB above the baseline from "Preparing Training" on, 6.6 GiB of it
  beyond the 14.4 GiB of pinned swap blocks; torch peak allocated 50.25 GiB (51 452 MB), 3.13 s/step.

The admission policy: the setting a probe measured as spilled is refused at setup, naming the
probe; everything else is allowed (an unmeasured setting gets the estimator's warning, never a
refusal). No test here queries the device: the card is a fabricated 97 887 MB one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.engine.models.registry import registry
from app.engine.utils.vram_estimator import VRAMEstimator
from tests.engine.families.test_minimax_h3_image_trainer import bare_trainer

DEFINITION_ID = "minimax-h3-image-t2i"
_OVER = 0.20  # the estimate may sit up to 20 % above a measured peak, never below it
_CELLS = {
    "P1": ({"quantization": "int8"}, 35_139.0),
    "P2": ({"quantization": "none"}, 65_684.0),
    "P3": ({"quantization": "none", "block_swap_config": {"transformer_blocks": 24}}, 51_452.0),
}
_PROBE_SETTINGS = {
    "train_batch_size": 1,
    "resolutions": [1024],
    "network_rank": 16,
    "gradient_checkpointing": True,
}


@pytest.fixture(scope="module")
def _loaded_registry():
    registry.discover_families()
    registry.load_definitions("app/engine/models/definitions")
    return registry


@pytest.fixture
def _a_96gb_card(monkeypatch):
    fake = SimpleNamespace(gpus=[SimpleNamespace(vram_total_mb=97_887, vram_used_mb=760)])
    monkeypatch.setattr("app.core.system_monitor.system_monitor.snapshot", lambda: fake)


def _estimate(**over):
    defn = registry.get_definition(DEFINITION_ID)
    assert defn is not None, DEFINITION_ID
    cfg = {
        "definition_id": DEFINITION_ID,
        "mixed_precision": "bf16",
        "te_quantization": "none",
        "cache_text_embeddings": True,
        "cache_latents": True,
        **_PROBE_SETTINGS,
        **over,
    }
    return VRAMEstimator.estimate(defn, cfg)


# ── The estimator against the measured peaks ──────────────────────────────


@pytest.mark.usefixtures("_loaded_registry", "_a_96gb_card")
@pytest.mark.parametrize("probe", sorted(_CELLS))
def test_the_training_peak_never_under_reports_a_measured_probe(probe):
    over, measured = _CELLS[probe]
    got = _estimate(**over).training_peak_mb
    assert measured <= got <= measured * (1 + _OVER), (
        f"{probe}: estimated {got:.0f} MB vs measured {measured:.0f} MB = {(got / measured - 1) * 100:+.1f} %"
    )


@pytest.mark.usefixtures("_loaded_registry", "_a_96gb_card")
@pytest.mark.parametrize("probe", ["P1", "P2"])
def test_the_settings_that_passed_fit_without_a_vram_warning(probe):
    report = _estimate(**_CELLS[probe][0])
    assert report.fit_known is True and report.fits is True, (report.peak_mb, report.available_mb)
    assert not [w for w in report.warnings if "exceeds free" in w or "tight fit" in w], report.warnings


# ── The refusal of the setting P3 measured as spilled ─────────────────────


def _p3(**over):
    return {**_PROBE_SETTINGS, "quantization": "none", "block_swap_config": {"transformer_blocks": 24}, **over}


@pytest.mark.parametrize("pct", [23, 24, 25])  # round(50 * pct / 100) = 12, the pipeline's own rounding
def test_bf16_with_twelve_swapped_blocks_is_refused_naming_p3(pct):
    trainer = bare_trainer(**_p3(block_swap_config={"transformer_blocks": pct}))
    with pytest.raises(ValueError, match=r"block_swap_config.*P3.*spilled"):
        trainer._setup_family()
    assert getattr(trainer, "loader", None) is None and getattr(trainer, "driver", None) is None


def test_the_refusal_reads_the_unset_shared_settings_as_their_defaults():
    config = _p3()
    for key in _PROBE_SETTINGS:
        config.pop(key)
    trainer = bare_trainer(**config)
    with pytest.raises(ValueError, match="P3"):
        trainer._setup_family()


@pytest.mark.parametrize(
    "over",
    [
        {"quantization": "int8"},  # P1
        {"block_swap_config": {}},  # P2
        {"block_swap_config": {"transformer_blocks": 20}},  # 10 blocks: unmeasured
        {"block_swap_config": {"transformer_blocks": 26}},  # 13 blocks: unmeasured
        {"block_swap_config": {"transformer_blocks": 24, "refiner_blocks": 50}},  # another group too
        {"quantization": "int8", "block_swap_config": {"transformer_blocks": 24}},  # unmeasured
        {"network_rank": 32},
        {"train_batch_size": 2},
        {"resolutions": [768]},
        {"gradient_checkpointing": False},
    ],
)
def test_every_setting_no_probe_measured_as_spilled_is_allowed(over):
    trainer = bare_trainer(**_p3(**over))
    trainer._setup_family()
    assert trainer.loader is not None
