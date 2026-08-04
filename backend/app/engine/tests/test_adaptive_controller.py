"""Freeze-mode behavior of AdaptiveTargetingController (spec §5, §7).

Every assertion is on observable state — ``requires_grad`` flags, ``p.grad``,
the bytes of the run-dir history file — never on "a method was called".

Shared fixtures (``make_adaptive_controller``, ``train_step``, ``lora_params``,
``make_peft_tiny``, ``FakeLogWriter``) live in this directory's ``conftest.py``.
"""

import json
import os

import torch
import torch.nn as nn

from app.engine.components.adaptive_targeting import AdaptiveTargetingController
from app.engine.models.adaptive import AdaptiveTargetingConfig


def test_discovers_all_lora_modules(make_adaptive_controller):
    _model, ctl, _writer = make_adaptive_controller()
    assert ctl.enabled
    assert ctl.total_count == 8  # 4 blocks × (to_q, to_v)
    assert ctl.active_count == 8


def test_event_freezes_cold_modules_observably(
    make_adaptive_controller, train_step, lora_params
):
    model, ctl, writer = make_adaptive_controller(energy_threshold=0.90)
    hot = ["blocks.0.to_q", "blocks.1.to_q"]
    for step in range(1, 25):
        train_step(model, hot)
        ctl.on_optimizer_step(step)
    # Warmup 10% of 100 = step 10; interval 10 → event fires at 20.
    cold = lora_params(model, "blocks.3.to_v")
    assert cold and all(p.requires_grad is False for p in cold)
    for name in hot:
        assert all(p.requires_grad for p in lora_params(model, name))
    assert ctl.active_count < 8
    kinds = [d["kind"] for t, d in writer.events if t == "adapt"]
    assert "narrow" in kinds


def test_frozen_params_receive_no_grad_on_next_backward(
    make_adaptive_controller, train_step, lora_params
):
    model, ctl, _writer = make_adaptive_controller(energy_threshold=0.90)
    for step in range(1, 25):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    model.zero_grad(set_to_none=True)
    model(torch.randn(2, 16)).sum().backward()
    assert all(p.grad is None for p in lora_params(model, "blocks.3.to_v"))


def test_never_freezes_below_floor(make_adaptive_controller, train_step):
    model, ctl, _writer = make_adaptive_controller(
        energy_threshold=0.5, min_active_pct=0.5
    )
    for step in range(1, 45):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    assert ctl.active_count >= 4  # ceil(0.5 * 8)


def test_freeze_mode_is_monotonic(make_adaptive_controller, train_step, lora_params):
    model, ctl, _writer = make_adaptive_controller(energy_threshold=0.90)
    for step in range(1, 25):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    narrowed = ctl.active_count
    assert narrowed < 8
    # Later, a previously-cold module gets hot — freeze mode must NOT re-admit it.
    for step in range(25, 45):
        train_step(model, ["blocks.3.to_v"])
        ctl.on_optimizer_step(step)
    assert ctl.active_count <= narrowed
    assert all(not p.requires_grad for p in lora_params(model, "blocks.3.to_v"))


def test_respects_manually_frozen_universe(
    tmp_path, make_peft_tiny, fake_log_writer, train_step, lora_params
):
    """Modules the user froze via targeted_layers stay OUT of the universe."""
    model = make_peft_tiny(4)
    for p in lora_params(model, "blocks.2"):
        p.requires_grad_(False)  # simulates TargetedLayerManager manual freeze
    ctl = AdaptiveTargetingController(
        model=model,
        config=AdaptiveTargetingConfig(
            warmup_pct=0.1, interval_steps=10, probe_steps=5
        ),
        total_steps=100,
        log_writer=fake_log_writer,
        output_dir=str(tmp_path),
    )
    assert ctl.total_count == 6  # blocks.2's two modules excluded
    for step in range(1, 45):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    assert all(not p.requires_grad for p in lora_params(model, "blocks.2"))


def test_zero_heat_window_skips_freezing(make_adaptive_controller):
    _model, ctl, writer = make_adaptive_controller()
    for step in range(1, 25):
        ctl.on_optimizer_step(step)  # no train_step → zero deltas
    assert ctl.active_count == 8
    kinds = [d["kind"] for t, d in writer.events if t == "adapt"]
    assert "narrow" not in kinds


def test_no_events_before_warmup(make_adaptive_controller, train_step):
    model, ctl, writer = make_adaptive_controller(warmup_pct=0.5)
    for step in range(1, 40):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    assert not [e for t, e in writer.events if t == "adapt"]


def test_json_history_written_atomically_and_parseable(
    tmp_path, make_adaptive_controller, train_step
):
    model, ctl, _writer = make_adaptive_controller(energy_threshold=0.90)
    for step in range(1, 25):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    path = os.path.join(str(tmp_path), "adaptive_targeting.json")
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["events"] and payload["events"][0]["kind"] == "narrow"
    assert payload["modules"] and len(payload["modules"]) == ctl.total_count
    assert not os.path.exists(path + ".tmp")


def test_three_consecutive_failures_disable_feature(
    make_adaptive_controller, train_step, monkeypatch
):
    model, ctl, writer = make_adaptive_controller()
    import app.engine.components.adaptive_targeting as mod

    monkeypatch.setattr(
        mod,
        "delta_frobenius_sq",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    for step in range(1, 60):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    assert ctl.enabled is False
    warnings = [d for t, d in writer.events if t == "warning"]
    assert any("boom" in str(w) for w in warnings)  # the reason is surfaced
    assert any("adaptive_targeting_disabled" in str(w) for w in warnings)


def test_zero_lora_modules_disables_with_warning(tmp_path, fake_log_writer):
    ctl = AdaptiveTargetingController(
        model=nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16)),  # no PEFT wrap
        config=AdaptiveTargetingConfig(),
        total_steps=100,
        log_writer=fake_log_writer,
        output_dir=str(tmp_path),
    )
    assert ctl.enabled is False
    assert any(
        "adaptive_targeting_disabled" in str(d)
        for t, d in fake_log_writer.events
        if t == "warning"
    )


def test_keep_patterns_match_targeted_layer_manager(
    make_adaptive_controller, train_step
):
    """The emitted patterns must actually re-select the active set via the
    production freeze mechanism (rebuild restarts depend on this)."""
    from app.engine.core.optimization.targeted_training import TargetedLayerManager

    model, ctl, _writer = make_adaptive_controller(energy_threshold=0.90)
    for step in range(1, 25):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    active_before = {
        n for n, p in model.named_parameters() if ".lora_" in n and p.requires_grad
    }
    assert active_before  # the patterns are non-trivial
    for name, p in model.named_parameters():  # scramble
        if ".lora_" in name:
            p.requires_grad_(True)
    TargetedLayerManager(ctl.keep_patterns()).apply(model)
    active_after = {
        n for n, p in model.named_parameters() if ".lora_" in n and p.requires_grad
    }
    assert active_after == active_before


def test_state_round_trip_restores_active_set(
    make_adaptive_controller, train_step, lora_params
):
    """A resumed run must reinstate the narrowed set on a freshly built model,
    otherwise the resume silently un-freezes everything the run had learned."""
    model, ctl, _writer = make_adaptive_controller(energy_threshold=0.90)
    for step in range(1, 25):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    state = ctl.get_state()
    assert state["active_modules"] and len(state["active_modules"]) < 8

    resumed_model, resumed, _w = make_adaptive_controller(energy_threshold=0.90)
    assert resumed.active_count == 8  # fresh controller starts wide open
    resumed.restore_state(state)

    assert resumed.keep_patterns() == ctl.keep_patterns()
    assert resumed.event_index == ctl.event_index
    every_module = [f"blocks.{i}.{proj}" for i in range(4) for proj in ("to_q", "to_v")]
    frozen_names = [n for n in every_module if n not in state["active_modules"]]
    assert frozen_names
    for name in frozen_names:
        assert all(not p.requires_grad for p in lora_params(resumed_model, name))
