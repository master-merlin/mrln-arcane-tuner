"""WAN 2.2 dual-expert EMA tests (W3.T10; no GPU/weights; fakes + tiny tensors).

``_configure_ema`` (``pipeline_optimization.py``) used to always bind
``EMAHandler`` to ``self._get_primary_model()`` — the single ACTIVE expert.
On a dual-expert ``both``-mode run this meant only the currently-active
expert's LoRA ever got an EMA shadow; the OTHER expert's saved file was raw
(un-EMA'd) weights — an asymmetric smoothing regime inside one logical LoRA
pair. ``Wan22Trainer._ema_parameters()`` fixes this by handing
``_configure_ema`` an explicit ``{name: param}`` mapping covering BOTH
experts, prefixed ``high.``/``low.`` so identically-named parameters from
each expert don't collide in one shadow dict.

Harnesses mirror the existing wan22 fake-transformer precedents exactly:
``test_wan22_dual_adapter.py``'s ``_make_trainer`` for the dual (``both``)
path, ``test_wan22_single_expert.py``'s ``_single_expert_trainer`` for the
single-expert (``high``/``low``) fallback path — real ``Wan22Driver`` + real
PEFT wrapping over tiny LoRA-targetable modules, no downloaded weights.
"""

import structlog
import torch
import torch.nn as nn

from app.engine.models.families.wan22.driver import Wan22Driver
from app.engine.models.families.wan22.expert_router import ExpertRouter
from app.engine.models.families.wan22.trainer import Wan22Trainer


# ── Tiny fake WAN transformer (LoRA-targetable) ───────────────────────────


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


def _dual_trainer(swap_mode: str = "resident") -> Wan22Trainer:
    """A ``Wan22Trainer`` with a real driver + BOTH tiny experts (mirrors
    ``test_wan22_dual_adapter.py``'s ``_make_trainer``)."""
    t = object.__new__(Wan22Trainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.expert_mode = "both"
    t.config = {
        "network_rank": 4,
        "network_alpha": 4,
        "timestep_sampling": "uniform",
        "expert_switch_interval": 1,
        "expert_swap_mode": swap_mode,
        "mixed_precision": "bf16",
        "optimizer_type": "AdamW",
        "learning_rate": 1e-4,
        "seed": 0,
        "ema": True,
        "ema_decay": 0.9,
    }
    t.components = {}
    driver = Wan22Driver(_Defn(), t.device)
    driver.assign_components({"unet": _TinyWan(), "unet_low": _TinyWan()})
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
    driver.configure_swap_mode(swap_mode)
    return t


def _single_expert_trainer(expert: str) -> Wan22Trainer:
    """A ``Wan22Trainer`` wired for single-expert training with ONE tiny
    expert (mirrors ``test_wan22_single_expert.py``'s ``_single_expert_trainer``)."""
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
        "ema": True,
        "ema_decay": 0.9,
    }
    t.components = {}
    driver = Wan22Driver(_Defn(), t.device)
    driver.configure_expert_mode(expert)
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


def _lora_names(model):
    return {n for n, p in model.named_parameters() if "lora_" in n and p.requires_grad}


# ── Dual-expert ('both' mode): EMA must shadow BOTH experts ───────────────


def test_ema_parameters_returns_prefixed_union_of_both_experts():
    t = _dual_trainer()
    t._apply_peft()

    params = t._ema_parameters()

    assert params is not None
    high_names = _lora_names(t.driver.transformer_high)
    low_names = _lora_names(t.driver.transformer_low)
    assert high_names and low_names, "fixture must actually produce LoRA params"

    assert {f"high.{n}" for n in high_names} <= set(params)
    assert {f"low.{n}" for n in low_names} <= set(params)
    # No cross-expert collision: every key is unambiguously prefixed.
    assert all(k.startswith("high.") or k.startswith("low.") for k in params)
    # Every returned param is actually trainable.
    assert all(p.requires_grad for p in params.values())


def test_configure_ema_shadow_covers_both_experts():
    """End-to-end: ``_configure_ema`` (the real seam) shadows BOTH experts,
    not just the active one — the actual bug this task fixes."""
    t = _dual_trainer()
    t._apply_peft()

    t._configure_ema()

    assert t.ema_handler is not None
    shadow_keys = set(t.ema_handler.shadow)
    high_names = _lora_names(t.driver.transformer_high)
    low_names = _lora_names(t.driver.transformer_low)
    assert {f"high.{n}" for n in high_names} <= shadow_keys
    assert {f"low.{n}" for n in low_names} <= shadow_keys
    # THE regression check: before the fix, only the ACTIVE expert (high, by
    # default) was shadowed — the low expert's params were entirely absent.
    assert any(k.startswith("low.") for k in shadow_keys), (
        "low expert has NO EMA shadow — the asymmetric-smoothing bug is back"
    )


def test_ema_step_updates_both_experts_shadows():
    """The shadow for BOTH experts actually tracks live param changes."""
    t = _dual_trainer()
    t._apply_peft()
    t._configure_ema()

    # Mutate one LoRA param from EACH expert directly, then step().
    high_name = next(iter(_lora_names(t.driver.transformer_high)))
    low_name = next(iter(_lora_names(t.driver.transformer_low)))
    high_param = dict(t.driver.transformer_high.named_parameters())[high_name]
    low_param = dict(t.driver.transformer_low.named_parameters())[low_name]
    high_param.data.fill_(1.0)
    low_param.data.fill_(1.0)

    before_high = t.ema_handler.shadow[f"high.{high_name}"].clone()
    before_low = t.ema_handler.shadow[f"low.{low_name}"].clone()
    t.ema_handler.step()

    assert not torch.allclose(t.ema_handler.shadow[f"high.{high_name}"], before_high)
    assert not torch.allclose(t.ema_handler.shadow[f"low.{low_name}"], before_low)


# ── Single-expert mode: unchanged (base primary-model) behavior ──────────


def test_ema_parameters_returns_none_for_single_expert_mode():
    t = _single_expert_trainer("high")
    t._apply_peft()
    assert t._ema_parameters() is None


def test_configure_ema_uses_primary_model_for_single_expert_mode():
    """Single-expert runs are byte-identical to the un-overridden base path:
    EMAHandler is bound to the primary model directly (unprefixed keys)."""
    t = _single_expert_trainer("high")
    t._apply_peft()

    t._configure_ema()

    assert t.ema_handler.model is t._get_primary_model()
    shadow_keys = set(t.ema_handler.shadow)
    primary_names = _lora_names(t._get_primary_model())
    assert shadow_keys == primary_names
    assert not any(k.startswith("high.") or k.startswith("low.") for k in shadow_keys)
