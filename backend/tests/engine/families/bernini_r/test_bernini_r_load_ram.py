"""Bernini-R 14B load-time host-RAM sequencing (no GPU / no weights).

The 14B is the same dual-transformer shape as WAN 2.2 A14B (two ~28 GB bf16
experts), so it inherits the wan22 ``defer_second_expert`` sequencing: the
low-noise expert is left OUT of the Phase-A manifest and materialised only
AFTER the high expert has moved to the GPU — host RAM never holds both experts
at once. The mixin mechanics (idempotence, latch reset on failure, dual-LoRA
wrap after materialisation) are pinned by ``test_wan22_load_ram.py`` against
the SHARED :class:`DualExpertDeferredLoadMixin`; this file pins the
bernini-specific wiring seams:

1. :class:`BerniniRLoader` IS a :class:`Wan22Loader` (specs + deferral
   inherited, not copied) and the 14B deferred manifest omits ``unet_low``.
2. The 1.3B single-expert manifest carries exactly one transformer and is
   untouched by ``expert_mode``/deferral flags.
3. ``BerniniRTrainer._setup_family`` wires deferral on for 14B ``both`` runs
   ONLY (the seam whose silent loss reverts to the eager ~67 GB dual load).
4. The Phase-B hooks materialise the deferred expert via the shared mixin,
   logged under the bernini-keyed event.
"""

from __future__ import annotations

import structlog
import torch

from app.engine.core.pipeline.loader_base import ComponentSpec
from app.engine.models.families.bernini_r.loader import BerniniRLoader
from app.engine.models.families.bernini_r.trainer import BerniniRTrainer
from app.engine.models.families.wan22.loader import Wan22Loader
from app.engine.models.families.wan_shared.trainer_base import (
    DualExpertDeferredLoadMixin,
)


class _Defn14B:
    architecture_params = {
        "dual_expert": True,
        "switch_dit_boundary": 0.875,
    }
    lora_targetable_modules: list[str] = []


class _Defn13B:
    architecture_params = {"skip_transformer_2": True}
    lora_targetable_modules: list[str] = []


def _specs(defn, expert_mode: str, defer: bool = False) -> dict[str, ComponentSpec]:
    loader = BerniniRLoader(
        torch.device("cpu"), expert_mode=expert_mode, defer_second_expert=defer
    )
    return {s.key: s for s in loader.get_component_manifest(defn)}


# ── Loader: wan22 subclass, deferral inherited ─────────────────────────────


def test_loader_is_wan22_subclass_with_shared_mixin_trainer():
    """Reuse seams: the loader inherits Wan22Loader (manifest + deferral +
    load_second_expert), the trainer the shared deferred-load mixin."""
    assert issubclass(BerniniRLoader, Wan22Loader)
    assert issubclass(BerniniRTrainer, DualExpertDeferredLoadMixin)


def test_14b_both_deferred_manifest_omits_low_expert():
    """The 14B Phase-A manifest defers ``transformer_2`` (the RAM saving)."""
    specs = _specs(_Defn14B(), "both", defer=True)
    assert specs["unet"].subfolder == "transformer"
    assert "unet_low" not in specs
    assert {"text_encoder", "vae", "tokenizer"} <= set(specs)


def test_14b_both_eager_manifest_loads_both_experts():
    specs = _specs(_Defn14B(), "both", defer=False)
    assert specs["unet"].subfolder == "transformer"
    assert specs["unet_low"].subfolder == "transformer_2"


def test_14b_single_expert_manifests_load_exactly_one_transformer():
    for mode, sub in (("high", "transformer"), ("low", "transformer_2")):
        specs = _specs(_Defn14B(), mode, defer=True)
        assert "unet_low" not in specs
        assert specs["unet"].subfolder == sub


def test_13b_manifest_single_expert_regardless_of_flags():
    """The 1.3B has no ``transformer_2`` subfolder: one transformer, always."""
    for mode in ("both", "high", "low"):
        specs = _specs(_Defn13B(), mode, defer=True)
        assert "unet_low" not in specs
        assert specs["unet"].subfolder == "transformer"
        assert {"text_encoder", "vae", "tokenizer"} <= set(specs)


def test_bernini_second_expert_log_event_is_family_keyed():
    assert BerniniRLoader.SECOND_EXPERT_LOG_EVENT == "bernini_r_load_second_expert"
    assert (
        BerniniRTrainer.DEFERRED_EXPERT_LOG_EVENT
        == "bernini_r_deferred_low_expert_materialized"
    )


# ── Trainer: _setup_family wires deferral for 14B both-mode only ───────────


def _setup(defn, mode: str) -> BerniniRTrainer:
    t = object.__new__(BerniniRTrainer)
    t.device = torch.device("cpu")
    t.definition = defn
    t.config = {
        "expert_mode": mode,
        "expert_switch_interval": 1,
        "timestep_sampling": "uniform",
        "seed": 0,
    }
    t._setup_family()
    return t


def test_setup_family_wires_deferral_for_14b_both_mode_only():
    """THE WIRING SEAM: dropping ``defer = expert_mode == "both"`` silently
    reverts the 14B to eager dual loading (the ~67 GB host-RAM hang) with
    every other test still green — pin the constructed loader's flag."""
    for mode, expected in (("both", True), ("high", False), ("low", False)):
        t = _setup(_Defn14B(), mode)
        assert isinstance(t.loader, BerniniRLoader)
        assert t.loader.defer_second_expert is expected, (
            f"expert_mode={mode!r}: defer_second_expert must be {expected}"
        )
        assert t.loader.expert_mode == mode


def test_setup_family_13b_never_defers():
    t = _setup(_Defn13B(), "both")
    assert isinstance(t.loader, BerniniRLoader)
    assert t.loader.defer_second_expert is False


# ── Phase-B hooks materialise the deferred expert (shared mixin path) ──────


def test_phase_b_hooks_materialize_deferred_low_expert():
    """Both hook call sites (grad-ckpt first in prepare_for_training, PEFT as
    the safety net) bring the deferred low expert back exactly once."""
    t = _setup(_Defn14B(), "both")
    t.logger = structlog.get_logger("test")
    t.components = {"unet": torch.nn.Linear(2, 2)}
    t.driver.assign_components({"unet": t.components["unet"]})

    low_holder = torch.nn.Linear(2, 2)
    calls: list[dict] = []

    class _FakeLoader:
        defer_second_expert = True

        def load_second_expert(self, definition, torch_dtype, initial_device="cpu"):
            calls.append({"dtype": torch_dtype, "device": initial_device})
            return low_holder

    t.loader = _FakeLoader()
    t.config["gradient_checkpointing"] = False

    assert t.driver.transformer_low is None
    t._configure_gradient_checkpointing()
    assert t.driver.transformer_low is low_holder
    assert t.components["unet_low"] is low_holder
    t._load_deferred_experts()  # idempotent — second entry must not reload
    assert calls == [{"dtype": torch.bfloat16, "device": "cpu"}]
