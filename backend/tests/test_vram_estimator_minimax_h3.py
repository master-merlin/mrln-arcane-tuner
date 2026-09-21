"""minimax_h3's VRAM estimate against the three measured GATE-5 cells (plan row 4.2).

Every measured number below is COPIED from an artifact of commit ``7cabd483``
(never read from the file, so the pin cannot drift with a re-run):

* ``.agent/output/h3-gates/gate5.json`` / ``gate5-runs/<cell>.json`` —
  ``peak_vram_mb`` (the allocator peak of a training-only process) at
  107 frames @ 768, rank 4, batch 1, AdamW, gradient checkpointing ON:
  bf16 92 400.9 MB · int8 61 554.1 MB · bf16 with 12 of 50 blocks swapped
  (``block_swap_config: {transformer_blocks: 24}``, a PERCENT) 77 572.2 MB.
  The ``config`` dicts below are those files' ``config`` reduced to the keys
  the estimator reads.
* ``.agent/output/h3-gates/gate5-memprofile.json`` — the traced live set at the
  peak: 29 455 MB of activations over 18 805 sequence rows (51 fp32 checkpoint
  boundaries of 385.6 MB + one recomputed block of 9 555 MB).

No test here queries the device: the card is a fabricated 97 887 MB one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.engine.models.registry import registry
from app.engine.utils.vram_estimator import VRAMEstimator

_BAND = 0.15
_CELLS = {
    "bf16": ({"quantization": "none"}, 92_400.9),
    "int8": ({"quantization": "int8"}, 61_554.1),
    "bf16-swap12": ({"quantization": "none", "block_swap_config": {"transformer_blocks": 24}}, 77_572.2),
}
_TRACED_ACTIVATIONS_MB = 29_455.0


@pytest.fixture(scope="module", autouse=True)
def _loaded_registry():
    registry.discover_families()
    registry.load_definitions("app/engine/models/definitions")
    return registry


@pytest.fixture(autouse=True)
def _a_96gb_card(monkeypatch):
    """The GATE-5 card as nvidia-smi showed it before a cell: 97 887 MB, ~760 used."""
    fake = SimpleNamespace(gpus=[SimpleNamespace(vram_total_mb=97_887, vram_used_mb=760)])
    monkeypatch.setattr("app.core.system_monitor.system_monitor.snapshot", lambda: fake)


def _config(**over) -> dict:
    cfg = {
        "definition_id": "minimax-h3-t2va",
        "network_rank": 4,
        "network_alpha": 4,
        "train_batch_size": 1,
        "optimizer_type": "AdamW",
        "resolutions": [768],
        "num_frames": 107,
        "gradient_checkpointing": True,
        "mixed_precision": "bf16",
        "te_quantization": "none",
        "train_audio": True,
    }
    cfg.update(over)
    return cfg


def _estimate(def_id: str = "minimax-h3-t2va", **over):
    defn = registry.get_definition(def_id)
    assert defn is not None, def_id
    return VRAMEstimator.estimate(defn, _config(**over))


@pytest.mark.parametrize("cell", sorted(_CELLS))
def test_training_peak_is_within_the_band_of_the_measured_cell(cell):
    over, measured = _CELLS[cell]
    got = _estimate(**over).training_peak_mb
    assert abs(got - measured) / measured <= _BAND, (
        f"{cell}: estimated {got:.0f} MB vs measured {measured:.0f} MB = {(got / measured - 1) * 100:+.1f} %"
    )


def test_activation_row_matches_the_traced_live_set():
    """The band is wide enough to hide a halved residual stream or a 27-frame
    sequence behind 63 GB of weights; the activation row alone is not."""
    got = _estimate().activations_mb
    assert got == pytest.approx(_TRACED_ACTIVATIONS_MB, rel=0.05), (
        f"activations {got:.0f} MB vs traced {_TRACED_ACTIVATIONS_MB:.0f} MB"
    )


def test_bf16_without_block_swap_is_flagged_on_a_96gb_card():
    report = _estimate(quantization="none")
    assert report.fit_known is True
    assert report.fits is False
    assert any("exceeds free" in w or "tight fit" in w for w in report.warnings), report.warnings


@pytest.mark.parametrize("cell", ["int8", "bf16-swap12"])
def test_the_two_usable_settings_fit_without_a_vram_warning(cell):
    report = _estimate(**_CELLS[cell][0])
    assert report.fit_known is True
    assert report.fits is True, (report.peak_mb, report.available_mb)
    assert not [w for w in report.warnings if "exceeds free" in w or "tight fit" in w], report.warnings


# ── block swap: shared, additive ────────────────────────────────────────────


def test_swapped_blocks_leave_the_device_budget():
    plain = _estimate()
    swapped = _estimate(block_swap_config={"transformer_blocks": 24})
    # 24 % of 50 blocks = 12 (the pipeline's own ``round(count * pct / 100)``) at the
    # topology's stated 1231 MB per bf16 block.
    assert plain.model_weights_mb - swapped.model_weights_mb == pytest.approx(12 * 1231)
    assert swapped.activations_mb == plain.activations_mb


def test_swap_credit_follows_the_quantized_block_size():
    plain = _estimate(quantization="int8")
    swapped = _estimate(quantization="int8", block_swap_config={"transformer_blocks": 24})
    assert plain.model_weights_mb - swapped.model_weights_mb == pytest.approx(12 * 1231 / 2)


def test_swap_credit_uses_the_pipelines_rounding_and_every_stated_group():
    plain = _estimate()
    # round(50 * 25 / 100) = round(12.5) = 12 under Python's rounding, which is what
    # PipelineOptimization._configure_block_swapping swaps; the two refiner blocks at 735 MB.
    swapped = _estimate(block_swap_config={"transformer_blocks": 25, "token_refiner_blocks": 100})
    assert plain.model_weights_mb - swapped.model_weights_mb == pytest.approx(12 * 1231 + 2 * 735)


@pytest.mark.parametrize("swap", [None, {}, {"transformer_blocks": 0}, {"no_such_group": 50}])
def test_a_config_that_swaps_nothing_yields_the_number_it_yields_without_the_key(swap):
    plain = _estimate().to_dict()
    same = _estimate(block_swap_config=swap).to_dict()
    assert same == plain


def test_a_definition_without_block_topology_ignores_the_key():
    real = registry.get_definition("sdxl_base_1.0")
    defn = SimpleNamespace(
        family=real.family,
        detected_precision=real.detected_precision,
        architecture_params=real.architecture_params,
        model_size_mb=real.model_size_mb,
    )
    cfg = {"network_rank": 16, "resolutions": [1024], "quantization": "none"}
    plain = VRAMEstimator.estimate(defn, cfg).to_dict()
    same = VRAMEstimator.estimate(defn, {**cfg, "block_swap_config": {"transformer_blocks": 50}}).to_dict()
    assert same == plain


def test_the_credit_never_takes_more_than_the_weights_it_is_taken_from():
    """``approx_vram_mb`` is a rough stated figure on most definitions (sdxl: 200 MB a block)."""
    real = registry.get_definition("sdxl_base_1.0")
    huge = [{"name": "down_blocks", "attr_path": "down_blocks", "count": 4, "approx_vram_mb": 10**6}]
    defn = SimpleNamespace(
        family=real.family,
        detected_precision=real.detected_precision,
        architecture_params=real.architecture_params,
        model_size_mb=real.model_size_mb,
        block_topology=huge,
    )
    cfg = {"network_rank": 16, "resolutions": [1024], "quantization": "none",
           "block_swap_config": {"down_blocks": 100}}
    assert VRAMEstimator.estimate(defn, cfg).model_weights_mb == 0.0


# ── the token-inventory path is engaged by ONE stated key ───────────────────


@pytest.mark.parametrize("def_id", ["minimax-h3-t2va", "minimax-h3-ref2va", "minimax-h3-fl2va"])
def test_every_h3_definition_states_the_residual_stream_dtype(def_id):
    arch = registry.get_definition(def_id).architecture_params
    assert arch.get("transformer.residual_stream_dtype") == "float32"
    assert _estimate(def_id).activations_mb == pytest.approx(_TRACED_ACTIVATIONS_MB, rel=0.05)


def test_without_the_key_the_legacy_activation_term_is_untouched():
    """43 784 MB is what ``b7f45a4e`` (before this path existed) budgets for the cell."""
    real = registry.get_definition("minimax-h3-t2va")
    arch = {k: v for k, v in real.architecture_params.items() if k != "transformer.residual_stream_dtype"}
    legacy = SimpleNamespace(
        family=real.family,
        detected_precision=real.detected_precision,
        architecture_params=arch,
        model_size_mb=real.model_size_mb,
        block_topology=real.block_topology,
    )
    assert round(VRAMEstimator.estimate(legacy, _config()).activations_mb) == 43_784


def test_activations_scale_with_the_stated_batch():
    assert _estimate(train_batch_size=2).activations_mb == pytest.approx(2 * _estimate().activations_mb)


def test_checkpointing_off_keeps_every_blocks_working_set():
    on = _estimate(gradient_checkpointing=True).activations_mb
    off = _estimate(gradient_checkpointing=False).activations_mb
    # ON = 52 boundaries + ONE block's working set; OFF = 52 working sets. The traced
    # working set is 9 555 MB, so OFF is an order of magnitude above ON, not the legacy 3x.
    assert off == pytest.approx(52 * 9_555, rel=0.10)
    assert off > 10 * on


def test_latent_frames_follow_the_definitions_frame_rule():
    """17n+5 pixel frames are 5n+2 latent frames (32 at 107), not the 4n+1 rule's 27."""
    from app.engine.models.families.minimax_h3.sampler import latent_frames_for
    from app.engine.utils.vram_estimator import _latent_frames_for_rule

    for frames in (22, 39, 107, 124, 345):
        assert _latent_frames_for_rule(frames, "17n+5", 4) == latent_frames_for(frames)
    # The rules every other video family states agree with their (F-1)//t + 1.
    assert _latent_frames_for_rule(81, "4n+1", 4) == 21
    assert _latent_frames_for_rule(121, "8n+1", 8) == 16
    assert _latent_frames_for_rule(81, None, 4) == 21


def test_formula_version_invalidates_coefficients_learned_before_this_path():
    """A v3 activation coefficient was taken against the 43.8 GB legacy row."""
    from app.engine.utils.vram_estimator import VRAM_FORMULA_VERSION

    assert VRAM_FORMULA_VERSION >= 4
