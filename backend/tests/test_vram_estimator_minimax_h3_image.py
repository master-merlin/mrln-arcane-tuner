"""minimax_h3_image against the three measured memory probes, and the refusal of the one that spilled.

Every measured number below is COPIED from the probe records of 2026-11-12 (never read from a file,
so the pin cannot drift with a re-run). Each probe was 20 steps on a 96 GB card, batch 1, the 1024
bucket, rank 16, gradient checkpointing on, the shipped definition otherwise:

* P1, int8, no block swap: PASS; torch peak allocated 34.32 GiB (35 139 MB), 1.48 s/step.
* P2, bf16, no block swap: PASS; torch peak allocated 64.14 GiB (65 684 MB), 0.97 s/step.
* P3, bf16, ``block_swap_config: {transformer_blocks: 24}`` (a PERCENT: 12 of 50 blocks): SPILLED.
  Shared GPU memory rose 21.07 GiB above the baseline from "Preparing Training" on, 6.6 GiB of it
  beyond the 14.4 GiB of pinned swap blocks; torch peak allocated 50.25 GiB (51 452 MB), 3.13 s/step.

All three ran with distillation handling ``none``. The handling is part of a setting's identity, so
they were repeated on 2026-11-13 with the definition's default, ``adapter`` (the training adapter
attached, int8 quantizing fresh with its float32 modules kept):

* P1: PASS; 34.70 GiB (35 533 MB), 1.76 s/step.
* P2: PASS; 64.49 GiB (66 040 MB), 0.95 s/step.
* P3: SPILLED. Shared GPU memory rose 21.17 GiB from "Preparing Training" on, 6.7 GiB beyond the
  14.5 GiB of pinned swap blocks; 50.53 GiB (51 739 MB), 3.15 s/step.

``contrastive_guidance`` and ``both`` were not probed, so P3's setting with them is allowed.

The admission policy: the setting a probe measured as spilled is refused at setup, naming the
probe; everything else is allowed (an unmeasured setting gets the estimator's warning, never a
refusal). No test here queries the device: the card is a fabricated 97 887 MB one.
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from app.engine.models.registry import registry
from app.engine.utils.vram_estimator import VRAMEstimator
from tests.engine.families.test_minimax_h3_image_trainer import bare_trainer

DEFINITION_ID = "minimax-h3-image-t2i"
_OVER = 0.20  # the estimate may sit up to 20 % above a measured peak, never below it
_CELLS = {
    "P1": ({"quantization": "int8"}, 35_139.0),
    "P2": ({"quantization": "none"}, 65_684.0),
    "P3": ({"quantization": "none", "block_swap_config": {"transformer_blocks": 24}}, 51_452.0),
    # The re-probes with the definition's default distillation handling (adapter), 2026-11-13.
    "P1-adapter": ({"quantization": "int8", "distillation_handling": "adapter"}, 35_533.0),
    "P2-adapter": ({"quantization": "none", "distillation_handling": "adapter"}, 66_040.0),
    "P3-adapter": ({"quantization": "none", "block_swap_config": {"transformer_blocks": 24},
                    "distillation_handling": "adapter"}, 51_739.0),
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
@pytest.mark.parametrize("probe", ["P1", "P2", "P1-adapter", "P2-adapter"])
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


def _shipped():
    defn = registry.get_definition(DEFINITION_ID)
    assert defn is not None and defn.distillation is not None, DEFINITION_ID
    return defn


@pytest.mark.usefixtures("_loaded_registry")
@pytest.mark.parametrize("handling", [None, "none", "adapter"])  # None: the definition's default, adapter
def test_the_shipped_definition_refuses_p3_with_each_handling_p3_measured(handling):
    """The distillation handling is part of a setting's identity: P3 spilled with ``none`` and with
    the default ``adapter``, so both are refused."""
    config = _p3() if handling is None else _p3(distillation_handling=handling)
    trainer = bare_trainer(_shipped(), **config)
    with pytest.raises(ValueError, match=r"block_swap_config.*P3.*spilled"):
        trainer._setup_family()


@pytest.mark.usefixtures("_loaded_registry")
@pytest.mark.parametrize("handling", ["contrastive_guidance", "both"])
def test_p3_with_a_handling_no_probe_measured_is_allowed(handling):
    trainer = bare_trainer(_shipped(), **_p3(distillation_handling=handling))
    trainer._setup_family()
    assert trainer.loader is not None


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


# ── The modules int8 keeps in float32 ─────────────────────────────────────


_FIXTURE_CONFIG = (
    Path(__file__).resolve().parent
    / "fixtures" / "minimax_h3_image_diffusers_configs" / "transformer_diffusers" / "config.json"
)
_KEPT = ("proj_in", "time_embedder", "pdd_proj_out")


@pytest.mark.usefixtures("_loaded_registry")
def test_the_estimator_finds_the_kept_modules_of_the_vendored_transformer():
    """The definition names the checkpoint's own class (``MiniMaxH3DiTModel``, harvested from
    ``transformer/config.json``); the estimator must reach the vendored port the loader builds,
    with the full-size model, so an int8 run charges what the port keeps in float32."""
    from app.engine.utils.vram_estimator import _primary_kept

    defn = registry.get_definition(DEFINITION_ID)
    kept = _primary_kept(defn)
    names = [name for name, _ in kept.params]
    for module in _KEPT:
        assert any(n.split(".")[0] == module for n in names), (module, names[:5])

    vendored = importlib.import_module("app.engine.models.families.minimax_h3_image.vendor")
    cls = vendored.MiniMaxH3ImageTransformer2DModel
    with torch.device("meta"):
        reference = cls.from_config(json.loads(_FIXTURE_CONFIG.read_text(encoding="utf-8")))
    declared = list(cls._keep_in_fp32_modules)
    want = {n: p.numel() for n, p in reference.named_parameters() if any(m in n.split(".") for m in declared)}
    assert dict(kept.params) == want
    # The build from the definition's native keys is the checkpoint's model, parameter for
    # parameter: a renamed key falls back to the port's default, which must be the checkpoint's.
    accepted = set(inspect.signature(cls.__init__).parameters)
    arch = defn.architecture_params
    native = {k[len("transformer."):]: v for k, v in arch.items() if k.startswith("transformer.")}
    with torch.device("meta"):
        built = cls.from_config({k: v for k, v in native.items() if k in accepted})
    shapes = {n: tuple(p.shape) for n, p in built.named_parameters()}
    assert shapes == {n: tuple(p.shape) for n, p in reference.named_parameters()}
    assert sum(p.numel() for p in built.parameters()) == 33_125_224_384
