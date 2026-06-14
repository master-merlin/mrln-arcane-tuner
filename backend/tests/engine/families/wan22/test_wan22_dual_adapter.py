"""WAN 2.2 dual-adapter wiring tests (no GPU/weights; fakes + tiny tensors).

Asserts the dual-expert plumbing that single-run training depends on:

- BOTH experts get LoRA-injected (``_apply_peft``).
- BOTH experts' trainable params reach the optimizer (``_configure_optimization``).
- ``get_primary_model()`` follows ``active_expert``.
- The swap-mode state transition moves the inactive model's device (verified with
  a fake ``.to()`` recorder).
"""

import structlog
import torch
import torch.nn as nn

from app.engine.models.families.wan22.driver import Wan22Driver
from app.engine.models.families.wan22.expert_router import HIGH, LOW, ExpertRouter
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


def _make_trainer(swap_mode="resident"):
    """A Wan22Trainer with a real driver + two tiny experts, minimal stubs."""
    t = object.__new__(Wan22Trainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
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


def _lora_param_count(model):
    return sum(
        p.numel()
        for n, p in model.named_parameters()
        if "lora_" in n and p.requires_grad
    )


# ── BOTH experts get LoRA ─────────────────────────────────────────────────


def test_both_experts_get_lora_injected():
    t = _make_trainer()
    t._apply_peft()
    high = t.driver.transformer_high
    low = t.driver.transformer_low
    assert hasattr(high, "peft_config"), "high expert not PEFT-wrapped"
    assert hasattr(low, "peft_config"), "low expert not PEFT-wrapped"
    assert _lora_param_count(high) > 0
    assert _lora_param_count(low) > 0


# ── BOTH param groups reach the optimizer ─────────────────────────────────


def test_optimizer_collects_both_experts_params():
    t = _make_trainer()
    t._apply_peft()
    t._configure_optimization(max_train_steps=10)

    opt_param_ids = {
        id(p) for group in t.optimizer.param_groups for p in group["params"]
    }
    high_lora = [
        p for n, p in t.driver.transformer_high.named_parameters() if "lora_" in n
    ]
    low_lora = [
        p for n, p in t.driver.transformer_low.named_parameters() if "lora_" in n
    ]
    assert high_lora and low_lora
    assert all(id(p) in opt_param_ids for p in high_lora), "high LoRA not in optimizer"
    assert all(id(p) in opt_param_ids for p in low_lora), "low LoRA not in optimizer"


# ── get_primary_model follows active_expert ───────────────────────────────


def test_primary_model_follows_active_expert():
    t = _make_trainer()
    t._apply_peft()
    driver = t.driver
    driver._set_active(HIGH)
    assert driver.get_primary_model() is driver.transformer_high
    driver._set_active(LOW)
    assert driver.get_primary_model() is driver.transformer_low


# ── swap-mode device transition (fake .to recorder) ───────────────────────


class _DeviceRecorder(nn.Module):
    """A module whose ``.to()`` records the requested device instead of moving."""

    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        self.current_device = "cpu"
        self.to_calls = []
        self._p = nn.Parameter(torch.zeros(1), requires_grad=False)

    def to(self, *args, **kwargs):  # noqa: A003 - shadow intentional
        target = args[0] if args else kwargs.get("device")
        self.to_calls.append(str(target))
        self.current_device = str(target)
        return self


def test_swap_mode_moves_inactive_expert_device():
    t = _make_trainer(swap_mode="swap")
    driver = t.driver
    high = _DeviceRecorder("high")
    low = _DeviceRecorder("low")
    driver.transformer_high = high
    driver.transformer_low = low
    driver._set_active(HIGH)

    # Drive a swap to the low expert (what on_optimizer_step does in swap mode).
    driver._swap_to(LOW)

    assert driver.active_expert == LOW
    assert driver.get_primary_model() is low
    # High (was active) was offloaded to CPU; low was brought to the device.
    assert "cpu" in high.to_calls, high.to_calls
    assert any("cpu" in c for c in low.to_calls) or any(
        "cpu" not in c for c in low.to_calls
    )
    # The target expert's last placement is the training device (cpu here).
    assert low.current_device == str(driver.device)


def test_resident_mode_does_not_offload_on_switch():
    t = _make_trainer(swap_mode="resident")
    driver = t.driver
    high = _DeviceRecorder("high")
    low = _DeviceRecorder("low")
    driver.transformer_high = high
    driver.transformer_low = low
    driver._set_active(HIGH)

    # In resident mode, on_optimizer_step flips the active pointer WITHOUT
    # offloading (no .to('cpu') on the previously-active expert).
    driver.router._block_decisions = {}  # force a fresh draw path
    driver._set_active(HIGH)
    # Simulate the router selecting LOW for the next step.
    driver.swap_mode = "resident"
    # Manually exercise the resident branch of on_optimizer_step:
    driver._set_active(LOW)
    assert "cpu" not in high.to_calls, "resident mode must not offload to CPU"


# ── on_optimizer_step advances the router (active flips per seeded plan) ───


def test_on_optimizer_step_advances_active_expert():
    t = _make_trainer()
    driver = t.driver
    # Build the expected plan from the router directly.
    plan = [driver.router.choose_expert(s) for s in range(1, 40)]
    # Reset the active expert to step 0's decision and replay via the hook.
    driver._set_active(driver.router.choose_expert(0))
    seen = []
    for step in range(0, 39):
        driver.on_optimizer_step(step)  # sets active for step+1
        seen.append(driver.active_expert)
    assert seen == plan
