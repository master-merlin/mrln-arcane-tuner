"""WAN 2.2 load-time host-RAM sequencing contract (no GPU / no weights).

WAN 2.2 A14B is the repo's only dual-transformer family: TWO ~14B experts,
~28 GB bf16 EACH once resident. The phased loader stages every component on CPU
first (``initial_device="cpu"``) and they stay there through the whole TE/VAE
caching stretch, so a naive ``both`` load pins ~2×28 GB = ~56 GB of transformer
in host RAM (plus the ~10.6 GB text encoder) for the entire load — enough to
fill a 64 GB box and hang it (the reported bug). ``wan21`` (one expert) fits.

The fix: dual-expert runs DEFER the low-noise expert out of the Phase-A manifest
and materialise it (``load_second_expert``) only AFTER the high expert has moved
to the GPU, so host RAM never holds both experts at once. These tests pin:

1. The deferred manifest omits ``unet_low`` (the actual RAM saving) while
   single-expert manifests are unchanged.
2. ``load_second_expert`` drives the REAL loader path to load ``transformer_2/``
   at bf16 onto CPU.
3. The trainer wires deferral on for ``both`` runs, materialises the low expert
   before wrapping (so BOTH experts still get LoRA), and does so exactly once.
"""

from __future__ import annotations

import structlog
import torch
import torch.nn as nn

from app.engine.core.pipeline.loader_base import ComponentSpec
from app.engine.models.families.wan22.driver import Wan22Driver
from app.engine.models.families.wan22.expert_router import ExpertRouter
from app.engine.models.families.wan22.loader import Wan22Loader
from app.engine.models.families.wan22.trainer import Wan22Trainer


class _Defn:
    architecture_params = {"mode": "t2v", "moe.boundary_ratio": 0.875}
    lora_targetable_modules: list[str] = []


# ── Tiny LoRA-targetable fake expert ───────────────────────────────────────


class _TinyBlock(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.attn1 = nn.Module()
        self.attn1.to_q = nn.Linear(dim, dim)
        self.attn1.to_k = nn.Linear(dim, dim)
        self.attn1.to_v = nn.Linear(dim, dim)


class _TinyWan(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(dim)])


def _specs(expert_mode: str, defer: bool = False) -> dict[str, ComponentSpec]:
    loader = Wan22Loader(
        torch.device("cpu"), expert_mode=expert_mode, defer_second_expert=defer
    )
    return {s.key: s for s in loader.get_component_manifest(_Defn())}


# ── Deferred manifest omits the low expert (the RAM saving) ────────────────


def test_both_deferred_manifest_omits_low_expert():
    """The Phase-A manifest carries the high expert but NOT the low one."""
    specs = _specs("both", defer=True)
    assert specs["unet"].subfolder == "transformer"  # high expert present
    assert "unet_low" not in specs  # low expert deferred out of Phase A
    # TE / VAE / tokenizer are still loaded up front.
    assert {"text_encoder", "vae", "tokenizer"} <= set(specs)


def test_both_eager_manifest_loads_both_experts():
    """Without deferral (the base manifest) both experts are present."""
    specs = _specs("both", defer=False)
    assert specs["unet"].subfolder == "transformer"
    assert specs["unet_low"].subfolder == "transformer_2"


def test_single_expert_manifests_unaffected_by_deferral():
    """high/low load exactly one transformer as ``unet`` regardless of the flag."""
    for mode, sub in (("high", "transformer"), ("low", "transformer_2")):
        specs = _specs(mode, defer=True)
        assert "unet_low" not in specs
        assert specs["unet"].subfolder == sub


# ── load_second_expert drives the REAL loader path (transformer_2 → CPU) ───


def test_load_second_expert_loads_transformer_2_bf16_to_cpu():
    """``load_second_expert`` reaches ``from_pretrained`` for ``transformer_2/``.

    Exercises the REAL ``GenericComponentLoader._load_single_spec`` (the same
    code the batch load path runs) with a recording fake class, asserting the
    low expert loads at bf16 onto CPU (not the GPU — so it inflates neither host
    peak beyond one expert nor VRAM during caching).
    """
    calls: list[dict] = []

    class _RecordingModel(nn.Module):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append({"path": path, **kwargs})
            return _TinyWan()

    loader = Wan22Loader(
        torch.device("cpu"), expert_mode="both", defer_second_expert=True
    )
    loader._root_path = "/fake/root"  # normally set by the initial load()

    def _fake_resolve_path(spec, definition, root_path):
        return f"{root_path}/{spec.subfolder}"

    loader._resolve_component_path = _fake_resolve_path  # type: ignore[method-assign]
    loader._import_class = staticmethod(lambda dotted: _RecordingModel)  # type: ignore[assignment]

    model = loader.load_second_expert(_Defn(), torch_dtype=torch.bfloat16)

    assert isinstance(model, nn.Module)
    assert len(calls) == 1
    assert calls[0]["path"].endswith("transformer_2")
    assert calls[0]["torch_dtype"] is torch.bfloat16


# ── Trainer: deferral wired on, low expert materialised before wrapping ────


def _both_trainer_with_fake_loader():
    """A ``both``-mode Wan22Trainer whose loader DEFERRED the low expert.

    Mirrors the real post-Phase-A state: driver has the high expert wired and
    ``transformer_low is None`` (deferred out); a fake loader hands back a tiny
    low expert on demand and counts the calls.
    """
    t = object.__new__(Wan22Trainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.definition = _Defn()
    t.expert_mode = "both"
    t.config = {
        "network_rank": 4,
        "network_alpha": 4,
        "timestep_sampling": "uniform",
        "expert_switch_interval": 1,
        "expert_swap_mode": "resident",
        "gradient_checkpointing": False,
        "mixed_precision": "bf16",
        "optimizer_type": "AdamW",
        "learning_rate": 1e-4,
        "seed": 0,
    }
    t.components = {"unet": _TinyWan()}
    driver = Wan22Driver(_Defn(), t.device)
    # Deferred Phase-A state: only the high expert is present.
    driver.assign_components({"unet": t.components["unet"]})  # unet_low absent → None
    driver.transformer_high = t.components["unet"]
    driver.transformer_low = None
    driver._set_active("high")
    t.driver = driver
    t.transformer = driver.get_primary_model()
    router = ExpertRouter(
        boundary=driver.boundary,
        switch_interval=1,
        timestep_cfg=t.config,
        seed=0,
        mc_samples=20_000,
    )
    t.expert_router = router
    driver.set_router(router)
    driver._set_active("high")
    driver.configure_swap_mode("resident")

    low_holder = _TinyWan()

    class _FakeLoader:
        defer_second_expert = True

        def __init__(self):
            self.calls: list[dict] = []

        def load_second_expert(self, definition, torch_dtype, initial_device="cpu"):
            self.calls.append(
                {"dtype": torch_dtype, "device": initial_device}
            )
            return low_holder

    t.loader = _FakeLoader()
    return t, low_holder


def test_load_deferred_experts_materializes_low_to_cpu():
    t, low = _both_trainer_with_fake_loader()
    assert t.driver.transformer_low is None

    t._load_deferred_experts()

    assert t.driver.transformer_low is low
    assert t.components["unet_low"] is low
    assert t.loader.calls == [{"dtype": torch.bfloat16, "device": "cpu"}]


def test_load_deferred_experts_is_idempotent():
    t, _low = _both_trainer_with_fake_loader()
    t._load_deferred_experts()
    t._load_deferred_experts()  # second call must not reload
    assert len(t.loader.calls) == 1


def test_grad_ckpt_hook_triggers_materialization_before_use():
    """The real ``_configure_gradient_checkpointing`` brings the low expert back.

    In ``prepare_for_training`` this hook runs AFTER the high expert moved to the
    GPU and BEFORE PEFT — so triggering the deferred load here is what keeps host
    RAM at one expert while still having both present before wrapping.
    """
    t, low = _both_trainer_with_fake_loader()
    t._configure_gradient_checkpointing()
    assert t.driver.transformer_low is low


def test_deferred_flow_still_wraps_both_experts_with_lora():
    """After materialisation the dual-LoRA path wraps BOTH experts (no regression)."""
    t, _low = _both_trainer_with_fake_loader()
    t._apply_peft()  # calls _load_deferred_experts() at its top

    high = t.driver.transformer_high
    low = t.driver.transformer_low
    assert hasattr(high, "peft_config"), "high expert not PEFT-wrapped"
    assert hasattr(low, "peft_config"), "low expert not PEFT-wrapped (deferral broke wrapping)"
    # And exactly one deferred load happened across the whole flow.
    assert len(t.loader.calls) == 1
