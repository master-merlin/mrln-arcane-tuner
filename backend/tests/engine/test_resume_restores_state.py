"""A resume brings back the scaler's and the EMA's state exactly, and a bf16 run steps on after it.

Every save and load goes through the real ``CheckpointManager``; the bf16 run resumes through the
trainer's real ``_resume_if_needed``. All of it runs on the CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from app.engine.components.checkpoints import CheckpointManager
from app.engine.strategies.ema import EMAHandler
from tests.engine.families.wan22.test_wan22_router_resume import (
    CPU,
    HIGH,
    WAN22,
    expert_params,
    make_training_trainer,
    resume,
    save_training,
    weights,
)


def _tiny_lora(fill: float = 0.5) -> Any:
    torch.manual_seed(0)
    model = get_peft_model(nn.Sequential(nn.Linear(4, 4)), LoraConfig(r=2, lora_alpha=2, target_modules=["0"]))
    with torch.no_grad():
        model.base_model.model[0].lora_B["default"].weight.fill_(fill)
    return model


def _trainable(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


# ── The GradScaler ──


def test_a_grad_scalers_state_comes_back_exactly(tmp_path):
    """An enabled scaler that backed off once and then grew its tracker: the load gives back
    that scale and that tracker, not a fresh scaler's."""
    model = _tiny_lora()
    params = [p for _, p in _trainable(model)]
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0**10, growth_interval=1000)
    assert scaler.is_enabled()
    for step in range(4):
        scaler.scale(model(torch.ones(2, 4)).pow(2).mean()).backward()
        if step == 0:
            params[0].grad.fill_(float("inf"))  # an overflow: the step is skipped and the scale backs off
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    saved = scaler.state_dict()
    assert saved["scale"] == 2.0**9 and saved["_growth_tracker"] == 3
    default = torch.amp.GradScaler("cpu").state_dict()
    assert saved["scale"] != default["scale"] and saved["_growth_tracker"] != default["_growth_tracker"]

    manager = CheckpointManager(str(tmp_path / "run"))
    folder = manager.save_checkpoint(
        step=4, components={"unet": model}, optimizer=optimizer, scaler=scaler, config={"lora_name": "t"}
    )
    fresh = torch.amp.GradScaler("cpu")
    state = manager.load_checkpoint(folder, scaler=fresh)

    assert "scaler" in state.components_loaded
    assert fresh.state_dict() == saved


# ── A bf16 run ──


def _bf16(t: Any) -> None:
    t.driver.transformer_high.to(torch.bfloat16)


def _bf16_step(t: Any, step: int) -> None:
    t.global_step = step
    model = t.driver.get_primary_model()
    model(torch.ones(2, 4, dtype=torch.bfloat16)).float().pow(2).mean().backward()
    t.optimizer.step()
    t.lr_scheduler.step()
    t.optimizer.zero_grad(set_to_none=True)
    t.driver.on_optimizer_step(step)


def test_a_bf16_run_resumes_and_steps(tmp_path):
    config = {"mixed_precision": "bf16"}
    first = make_training_trainer(
        WAN22, config, out=tmp_path / "run", device=CPU, expert_mode=HIGH, before_optimizer=_bf16
    )
    assert expert_params(first, HIGH) and all(p.dtype == torch.bfloat16 for p in expert_params(first, HIGH))
    first.max_train_steps = 100
    for step in range(3):
        _bf16_step(first, step)
    folder = save_training(first, 2)

    resumed = make_training_trainer(
        WAN22, config, out=tmp_path / "run", device=CPU, expert_mode=HIGH, before_optimizer=_bf16
    )
    assert resume(resumed, folder) == 3
    before = weights(resumed, HIGH)

    _bf16_step(resumed, 3)

    for now, then in zip(weights(resumed, HIGH), before):
        assert not torch.equal(now, then), "the step after the resume moved this bf16 weight"


# ── EMA ──


def _ema_source(kind: str, model: nn.Module) -> Any:
    """The handler's two bindings: a model, or the dual-expert trainers' prefixed name map."""
    if kind == "module":
        return model
    return {f"high.{name}": p for name, p in _trainable(model)}


def _ema_round_trip(tmp_path: Path, kind: str) -> tuple[EMAHandler, dict[str, torch.Tensor], EMAHandler, Any]:
    model = _tiny_lora()
    ema = EMAHandler(_ema_source(kind, model), decay=0.9)
    optimizer = torch.optim.AdamW([p for _, p in _trainable(model)], lr=1e-1)
    for _ in range(3):
        model(torch.ones(2, 4)).pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.step()
    saved = {name: shadow.clone() for name, shadow in ema.shadow.items()}
    live = dict(ema._named_parameters())
    assert saved and all(not torch.equal(saved[name], live[name].detach()) for name in saved), "shadows lag"

    manager = CheckpointManager(str(tmp_path / "run"))
    folder = manager.save_checkpoint(step=3, components={"unet": model}, ema_handler=ema, config={"lora_name": "t"})

    fresh = EMAHandler(_ema_source(kind, _tiny_lora()), decay=0.9)
    assert any(not torch.equal(fresh.shadow[name], saved[name]) for name in saved), "the load has work to do"
    state = manager.load_checkpoint(folder, ema_handler=fresh)
    return ema, saved, fresh, state


@pytest.mark.parametrize("kind", ["module", "named-params"])
def test_the_ema_shadow_comes_back_exactly(tmp_path, kind):
    _ema, saved, fresh, state = _ema_round_trip(tmp_path, kind)

    assert "ema" in state.components_loaded
    assert fresh.shadow.keys() == saved.keys()
    assert all(torch.equal(fresh.shadow[name], saved[name]) for name in saved)


@pytest.mark.parametrize("kind", ["module", "named-params"])
def test_the_ema_step_count_and_decay_come_back(tmp_path, kind):
    ema, _saved, fresh, _state = _ema_round_trip(tmp_path, kind)

    assert fresh.decay == ema.decay
    assert fresh._step_count == ema._step_count


@pytest.mark.parametrize("kind", ["module", "named-params"])
def test_a_resume_that_restores_every_shadow_warns_of_no_mismatch(tmp_path, kind, monkeypatch):
    """The shadow holds the trainable params only, so frozen base weights are not "missing"."""
    import app.engine.strategies.ema as ema_module

    warned: list[tuple[str, dict]] = []

    class _Logger:
        def __getattr__(self, name):
            return getattr(real, name)

        def warning(self, event, **kw):
            warned.append((event, kw))

    real = ema_module.logger
    monkeypatch.setattr(ema_module, "logger", _Logger())
    _ema_round_trip(tmp_path, kind)

    assert [event for event, _ in warned if event == "ema_shadow_key_mismatch"] == []


def test_a_resume_missing_a_trainable_shadow_still_warns(tmp_path, monkeypatch):
    import app.engine.strategies.ema as ema_module

    warned: list[str] = []

    class _Logger:
        def __getattr__(self, name):
            return getattr(real, name)

        def warning(self, event, **kw):
            warned.append(event)

    real = ema_module.logger
    monkeypatch.setattr(ema_module, "logger", _Logger())
    fresh = EMAHandler(_tiny_lora(), decay=0.9)
    one = next(iter(fresh.shadow))
    fresh.load_state_dict({one: fresh.shadow[one].clone()})

    assert warned == ["ema_shadow_key_mismatch"]
