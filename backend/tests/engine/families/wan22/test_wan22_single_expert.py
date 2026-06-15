"""WAN 2.2 single-expert (high/low-noise-only) training (no GPU/weights).

Covers the ``expert_mode`` ∈ {``high``, ``low``} path that halves VRAM by
training ONE expert in a run (ai-toolkit style):

- **Loader** emits exactly ONE transformer spec, mapped to ``unet`` (``low``
  loads ``transformer_2/`` as ``unet``) — the real VRAM save.
- **Router** is PINNED: every step routes to the one expert and timesteps are
  truncated to that expert's boundary range; no Bernoulli, no ``p_high`` MC.
- **Driver** wires the single loaded transformer into the right expert slot and
  leaves the other ``None``.
- **Trainer** wraps / collects / optimizes only the resident adapter.
- **Saver** writes exactly ONE ComfyUI file (``_high_noise`` or ``_low_noise``).
"""

import structlog
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from app.engine.models.families.wan22.driver import Wan22Driver
from app.engine.models.families.wan22.expert_router import HIGH, LOW, ExpertRouter
from app.engine.models.families.wan22.loader import Wan22Loader
from app.engine.models.families.wan22.saver import Wan22Saver
from app.engine.models.families.wan22.trainer import Wan22Trainer


# ── Fakes (mirror the dual-adapter test) ──────────────────────────────────


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


class _Defn:
    architecture_params = {"mode": "t2v", "moe.boundary_ratio": 0.875}
    lora_targetable_modules: list[str] = []


def _single_expert_trainer(expert: str) -> Wan22Trainer:
    """A Wan22Trainer wired for single-expert training with ONE tiny expert."""
    t = object.__new__(Wan22Trainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.expert_mode = expert
    t.config = {
        "expert_mode": expert,
        "network_rank": 4,
        "network_alpha": 4,
        "timestep_sampling": "uniform",
        "expert_switch_interval": 1,
        "mixed_precision": "bf16",
        "optimizer_type": "AdamW",
        "learning_rate": 1e-4,
        "seed": 0,
    }
    t.components = {}
    driver = Wan22Driver(_Defn(), t.device)
    driver.configure_expert_mode(expert)
    # Single-expert: the loader maps the chosen transformer to "unet".
    driver.assign_components({"unet": _TinyWan()})
    t.driver = driver
    t.transformer = driver.get_primary_model()
    router = ExpertRouter(
        boundary=driver.boundary,
        switch_interval=1,
        timestep_cfg=t.config,
        seed=0,
        pinned_expert=expert,
    )
    t.expert_router = router
    driver.set_router(router)
    driver.configure_swap_mode("resident")
    return t


def _lora_params(model):
    return [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]


# ── Loader manifest (the real VRAM save) ───────────────────────────────────


def _specs(expert_mode):
    loader = Wan22Loader(torch.device("cpu"), expert_mode=expert_mode)
    return {s.key: s for s in loader.get_component_manifest(_Defn())}


def test_loader_high_loads_only_transformer_as_unet():
    specs = _specs("high")
    assert "unet" in specs and "unet_low" not in specs
    assert specs["unet"].subfolder == "transformer"


def test_loader_low_loads_transformer_2_as_unet():
    specs = _specs("low")
    assert "unet" in specs and "unet_low" not in specs
    # The low expert is loaded AS the single primary "unet".
    assert specs["unet"].subfolder == "transformer_2"


def test_loader_both_loads_both_transformers():
    specs = _specs("both")
    assert specs["unet"].subfolder == "transformer"
    assert specs["unet_low"].subfolder == "transformer_2"


# ── Router pinned ──────────────────────────────────────────────────────────


def _pinned_router(expert):
    return ExpertRouter(
        boundary=0.875,
        timestep_cfg={"timestep_sampling": "uniform"},
        seed=0,
        pinned_expert=expert,
    )


def test_pinned_router_always_chooses_the_pinned_expert():
    r = _pinned_router(HIGH)
    assert r.pinned_expert == HIGH
    assert [r.choose_expert(s) for s in (0, 1, 7, 123, 999)] == [HIGH] * 5
    # p_high is the degenerate 1.0 (the unpinned MC path clamps to <1.0).
    assert r.p_high == 1.0

    r_low = _pinned_router(LOW)
    assert [r_low.choose_expert(s) for s in (0, 3, 50)] == [LOW] * 3
    assert r_low.p_high == 0.0


def test_pinned_router_truncates_timesteps_to_expert_range():
    dev = torch.device("cpu")
    cfg = {"timestep_sampling": "uniform"}
    hi = _pinned_router(HIGH).sample_timesteps_for(HIGH, 128, dev, cfg)
    lo = _pinned_router(LOW).sample_timesteps_for(LOW, 128, dev, cfg)
    assert bool((hi >= 875.0).all()), hi.min().item()
    assert bool((lo < 875.0).all()), lo.max().item()


def test_unpinned_router_phigh_is_strictly_interior():
    """Sanity contrast: a normal router estimates p_high in (0, 1)."""
    r = ExpertRouter(
        boundary=0.875,
        timestep_cfg={"timestep_sampling": "uniform"},
        seed=0,
        mc_samples=20_000,
    )
    assert r.pinned_expert is None
    assert 0.0 < r.p_high < 1.0


def test_pinned_router_state_roundtrip():
    r = _pinned_router(LOW)
    state = r.state_dict()
    assert state["pinned_expert"] == LOW
    r2 = _pinned_router(HIGH)
    r2.load_state_dict(state)
    assert r2.pinned_expert == LOW
    assert r2.choose_expert(5) == LOW


# ── Driver wiring (single transformer, other slot None) ────────────────────


def test_driver_high_mode_wires_high_only():
    d = Wan22Driver(_Defn(), torch.device("cpu"))
    d.configure_expert_mode("high")
    m = _TinyWan()
    d.assign_components({"unet": m})
    assert d.transformer_high is m
    assert d.transformer_low is None
    assert d.active_expert == HIGH
    assert d.get_primary_model() is m


def test_driver_low_mode_wires_low_only():
    d = Wan22Driver(_Defn(), torch.device("cpu"))
    d.configure_expert_mode("low")
    m = _TinyWan()
    d.assign_components({"unet": m})  # loader put transformer_2/ here
    assert d.transformer_low is m
    assert d.transformer_high is None
    assert d.active_expert == LOW
    assert d.get_primary_model() is m


# ── Trainer: only the resident adapter is wrapped / optimized ──────────────


def test_single_expert_wraps_only_active_expert():
    t = _single_expert_trainer("high")
    t._apply_peft()
    assert hasattr(t.driver.transformer_high, "peft_config")
    assert t.driver.transformer_low is None
    assert len(_lora_params(t.driver.transformer_high)) > 0


def test_single_expert_low_wraps_only_low():
    t = _single_expert_trainer("low")
    t._apply_peft()
    assert hasattr(t.driver.transformer_low, "peft_config")
    assert t.driver.transformer_high is None


def test_single_expert_optimizer_collects_one_adapter():
    t = _single_expert_trainer("high")
    t._apply_peft()
    collected = t._collect_expert_params()
    high_lora = _lora_params(t.driver.transformer_high)
    assert collected and len(collected) == len(high_lora)

    t._configure_optimization(max_train_steps=10)
    opt_ids = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
    assert all(id(p) in opt_ids for p in high_lora)


def test_single_expert_optimizer_step_is_noop():
    """Pinned router → active expert never changes across optimizer steps."""
    t = _single_expert_trainer("low")
    t._apply_peft()
    for step in range(8):
        t.driver.on_optimizer_step(step)
        assert t.driver.active_expert == LOW


# ── Saver: exactly one file with the right expert suffix ───────────────────


def _peft_tiny():
    return get_peft_model(
        _TinyWan(),
        LoraConfig(r=4, lora_alpha=4, target_modules=["attn1.to_q", "attn1.to_k"]),
    )


def test_saver_high_only_writes_one_file(tmp_path):
    out = tmp_path / "run.safetensors"
    Wan22Saver(mode="t2v").save(
        {"unet_high": _peft_tiny(), "unet_low": None, "config": {}}, out, metadata={}
    )
    assert (tmp_path / "run_high_noise.safetensors").exists()
    assert not (tmp_path / "run_low_noise.safetensors").exists()


def test_saver_low_only_writes_one_file(tmp_path):
    out = tmp_path / "run.safetensors"
    Wan22Saver(mode="t2v").save(
        {"unet_high": None, "unet_low": _peft_tiny(), "config": {}}, out, metadata={}
    )
    assert (tmp_path / "run_low_noise.safetensors").exists()
    assert not (tmp_path / "run_high_noise.safetensors").exists()
