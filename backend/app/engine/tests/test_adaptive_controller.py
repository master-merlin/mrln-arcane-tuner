"""Freeze-mode behavior of AdaptiveTargetingController (spec §5, §7).

Every assertion is on observable state — ``requires_grad`` flags, ``p.grad``,
the bytes of the run-dir history file — never on "a method was called".

Shared fixtures (``make_adaptive_controller``, ``train_step``, ``lora_params``,
``make_peft_tiny``, ``FakeLogWriter``) live in this directory's ``conftest.py``.
"""

import json
import os

import pytest
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
    # The module's hard invariant: base model params are NEVER touched. PEFT
    # already froze them, so only an exact before/after comparison can catch a
    # regression that flips one — every other assertion here filters on lora_.
    base_before = {
        n: p.requires_grad for n, p in model.named_parameters() if ".lora_" not in n
    }
    assert base_before  # the comparison must not be vacuous
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
    base_after = {
        n: p.requires_grad for n, p in model.named_parameters() if ".lora_" not in n
    }
    assert base_after == base_before


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


def test_freeze_mode_is_monotonic(
    make_adaptive_controller, train_step, force_update, lora_params
):
    model, ctl, _writer = make_adaptive_controller(energy_threshold=0.90)
    for step in range(1, 25):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    before = ctl.get_state()["active_modules"]
    assert len(before) < 8
    assert "blocks.3.to_v" not in before

    # A frozen module now moves MORE than anything still active — the metric
    # sees genuine heat on it. train_step cannot produce this (it skips frozen
    # params, so driving it through train_step would prove nothing: the module
    # would be re-admitted by no implementation at all). Freeze mode must still
    # refuse; reactivation is probe-gated (Task 4).
    for step in range(25, 45):
        train_step(model, ["blocks.0.to_q"])
        force_update(model, "blocks.3.to_v", scale=5.0)
        ctl.on_optimizer_step(step)

    after = ctl.get_state()["active_modules"]
    # Set containment, not a count: freezing one module while un-freezing
    # another keeps the count equal but is exactly what must not happen.
    assert set(after) <= set(before)
    assert "blocks.3.to_v" not in after
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

    # Persisted heat/hot for a user-frozen module must not survive a resume
    # either: it is outside the universe, so reporting it in hot_count and
    # top_modules would advertise a module the controller never manages.
    ctl.restore_state(
        {"heat": {"blocks.2.to_q": 9.0}, "hot_modules": ["blocks.2.to_q"]}
    )
    assert ctl.hot_count == 0
    assert "blocks.2.to_q" not in ctl.get_state()["heat"]


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


def _patch_flaky_heat(monkeypatch):
    """Make ``delta_frobenius_sq`` fail on demand; returns the switch dict."""
    import app.engine.components.adaptive_targeting as mod

    real = mod.delta_frobenius_sq
    switch = {"failing": True}

    def flaky(*args, **kwargs):
        if switch["failing"]:
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "delta_frobenius_sq", flaky)
    return switch


def _run(ctl, model, train_step, first, last):
    for step in range(first, last):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)


def test_three_consecutive_failures_disable_feature(
    make_adaptive_controller, train_step, monkeypatch
):
    model, ctl, writer = make_adaptive_controller()
    _patch_flaky_heat(monkeypatch)

    # Events land at 20, 30, 40 (warmup 10 + interval 10).
    _run(ctl, model, train_step, 1, 31)
    assert ctl.enabled is True  # two strikes is not three
    _run(ctl, model, train_step, 31, 41)
    assert ctl.enabled is False

    warnings = [d for t, d in writer.events if t == "warning"]
    assert any("boom" in str(w) for w in warnings)  # the reason is surfaced
    assert any("adaptive_targeting_disabled" in str(w) for w in warnings)


def test_failed_event_does_not_advance_the_heat_state(
    make_adaptive_controller, train_step, monkeypatch
):
    """An event that dies part-way must leave the EMA and the snapshot it was
    measured against consistent. Advancing the heat and then failing before the
    snapshot is replaced makes the next window count this interval twice."""
    import app.engine.components.adaptive_targeting as mod

    model, ctl, _writer = make_adaptive_controller()
    for step in range(1, 21):  # event at 20 succeeds and records heat
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    heat_before = ctl.get_state()["heat"]
    assert heat_before

    monkeypatch.setattr(
        mod,
        "select_active",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    for step in range(21, 31):  # event at 30 dies after heat was computed
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)

    assert ctl.enabled is True  # one failure is survivable
    assert ctl.get_state()["heat"] == heat_before


def test_successful_event_resets_the_failure_streak(
    make_adaptive_controller, train_step, monkeypatch
):
    """The budget is three CONSECUTIVE failures, not three failures ever —
    otherwise a long run with occasional transient errors disables itself."""
    model, ctl, _writer = make_adaptive_controller()
    switch = _patch_flaky_heat(monkeypatch)

    _run(ctl, model, train_step, 1, 30)  # event 20 fails
    switch["failing"] = False
    _run(ctl, model, train_step, 30, 40)  # event 30 succeeds → streak reset
    switch["failing"] = True
    _run(ctl, model, train_step, 40, 60)  # events 40, 50 fail
    assert ctl.enabled is True  # three failures total, only two in a row
    _run(ctl, model, train_step, 60, 70)  # event 60 fails → three in a row
    assert ctl.enabled is False


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


def test_floor_padding_keeps_the_hottest_module_not_a_positional_prefix(
    make_adaptive_controller, train_step
):
    """When the monotonic intersect drops the keep-set below the floor, the
    padding must re-admit the HOTTEST still-active modules. Padding with a
    positional prefix of the active list instead would freeze the one module
    that is still learning, whenever it sorts late in module order.

    ``heat_ema=0`` (no smoothing) makes the second window read exactly zero for
    modules that stopped moving, which is what puts the frozen, module-ordered
    filler modules ahead of them in ``select_active``'s own floor padding.
    """
    model, ctl, _writer = make_adaptive_controller(
        energy_threshold=0.99, min_active_pct=0.13, heat_ema=0.0
    )
    # Event at step 20: three modules learn, none of them early in module order.
    # The loop stops ON the event, so no post-event step leaks into window two.
    warm = ["blocks.0.to_v", "blocks.1.to_q", "blocks.3.to_v"]
    for step in range(1, 21):
        train_step(model, warm)
        ctl.on_optimizer_step(step)
    assert set(ctl.get_state()["active_modules"]) == set(warm)

    # Event at step 30: only the LAST of the three still learns. select_active
    # pads its own floor with zero-heat modules in module order — all frozen —
    # so the intersect with the active set falls to one, below the floor of 2.
    for step in range(21, 31):
        train_step(model, ["blocks.3.to_v"])
        ctl.on_optimizer_step(step)

    active = ctl.get_state()["active_modules"]
    assert len(active) == 2  # ceil(0.13 * 8), the floor — padding ran
    assert "blocks.3.to_v" in active  # a positional prefix would have evicted it


def test_restore_state_warns_when_persisted_modules_do_not_match(
    make_adaptive_controller,
):
    """A state written against a different module graph must not quietly leave
    the controller wide open — the narrowing decision would vanish in silence."""
    _model, ctl, writer = make_adaptive_controller()
    ctl.restore_state(
        {"active_modules": ["blocks.99.to_q", "somewhere.else"], "event_index": 4}
    )
    assert ctl.active_count == 8  # deliberately open — and said so
    warnings = [str(d) for t, d in writer.events if t == "warning"]
    assert any(
        "resumed state lists 2" in w and "only 0" in w for w in warnings
    )  # both counts named


def test_unmanaged_adapters_are_reported_not_silently_dropped(
    tmp_path, fake_log_writer
):
    """A peft Embedding keeps its adapter in ``lora_embedding_A``/``_B``, which
    this controller cannot measure — but TargetedLayerManager DOES manage those
    suffixes, so an unreported skip is a module ``keep_patterns()`` would
    silently freeze on a rebuild restart."""
    from peft import LoraConfig, get_peft_model

    class _EmbeddingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(8, 16)
            self.to_q = nn.Linear(16, 16)

        def forward(self, idx):
            return self.to_q(self.tok_emb(idx))

    model = get_peft_model(
        _EmbeddingModel(),
        LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "tok_emb"]),
    )
    ctl = AdaptiveTargetingController(
        model=model,
        config=AdaptiveTargetingConfig(),
        total_steps=100,
        log_writer=fake_log_writer,
        output_dir=str(tmp_path),
    )
    assert ctl.enabled is True  # keeps operating
    assert ctl.total_count == 1  # only to_q is measurable
    warnings = [str(d) for t, d in fake_log_writer.events if t == "warning"]
    assert len(warnings) == 1  # ONE line, not one per module
    assert "tok_emb" in warnings[0]


def test_history_heat_survives_tiny_values_and_non_finite(
    tmp_path, make_adaptive_controller
):
    """Heat is stored with six SIGNIFICANT figures: real per-window ‖ΔW‖² runs
    far below 1e-6, where decimal rounding would write the whole map as zeros.
    A non-finite value must serialize as null, never as a plausible 0.0 —
    JSON.parse rejects NaN, and 0.0 would read as 'this layer never learned'."""
    _model, ctl, _writer = make_adaptive_controller()
    ctl.restore_state(
        {"heat": {"blocks.0.to_q": 3.21e-11, "blocks.1.to_q": float("nan")}}
    )
    ctl._write_history()

    path = os.path.join(str(tmp_path), "adaptive_targeting.json")
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    assert "NaN" not in raw and "Infinity" not in raw
    heat = json.loads(raw)["heat"]
    assert heat["blocks.0.to_q"] == 3.21e-11  # not flattened to 0.0
    assert heat["blocks.1.to_q"] is None  # not a plausible 0.0


class _CompiledShim(nn.Module):
    """Stand-in for ``torch._dynamo.OptimizedModule``.

    ``torch.compile`` returns a wrapper that holds the real module under
    ``_orig_mod``, so every submodule name gains that segment. Simulated rather
    than compiled for real: the fp8 compile path needs a GPU + Triton, and the
    only thing under test here is the NAME shape it produces.
    """

    def __init__(self, inner) -> None:
        super().__init__()
        self._orig_mod = inner

    def forward(self, *args, **kwargs):
        return self._orig_mod(*args, **kwargs)


def _controller(model, tmp_path, writer, **cfg):
    return AdaptiveTargetingController(
        model=model,
        config=AdaptiveTargetingConfig(
            warmup_pct=0.1, interval_steps=10, probe_steps=5, min_active_pct=0.13, **cfg
        ),
        total_steps=100,
        log_writer=writer,
        output_dir=str(tmp_path),
    )


def test_compile_wrapper_segment_is_stripped_from_module_names(
    tmp_path, make_peft_tiny, fake_log_writer, train_step
):
    """A name carrying ``_orig_mod`` would make every emitted pattern miss.

    The patterns are persisted as ``targeted_layers`` and re-applied by a
    DIFFERENT process whose compile state may differ (the fp8 family compiles,
    a plain resume of the same job may not). Names must therefore be
    compile-state independent, or the restart matches nothing and trains
    nothing.
    """
    from app.engine.core.optimization.targeted_training import TargetedLayerManager

    model = _CompiledShim(make_peft_tiny(4))
    ctl = _controller(model, tmp_path, fake_log_writer, energy_threshold=0.90)
    assert ctl.total_count == 8  # discovery still finds every module

    hot = ["blocks.1.to_v", "blocks.3.to_q"]
    for step in range(1, 21):
        train_step(model, hot)
        ctl.on_optimizer_step(step)
    assert set(ctl.get_state()["active_modules"]) == set(hot)
    assert not any("_orig_mod" in pattern for pattern in ctl.keep_patterns())

    # The contract that matters: a different, UNCOMPILED process re-applies them.
    fresh = make_peft_tiny(4)
    TargetedLayerManager(ctl.keep_patterns()).apply(fresh)
    trainable = {
        n for n, p in fresh.named_parameters() if ".lora_" in n and p.requires_grad
    }
    assert trainable  # not vacuous: something stayed trainable
    assert trainable == {
        n
        for n, _ in fresh.named_parameters()
        if ".lora_" in n and any(module in n for module in hot)
    }


def test_targeted_layer_manager_matches_a_compiled_model(make_peft_tiny):
    """The mirror direction of the same defect: patterns captured WITHOUT the
    compile wrapper must still select the right modules in a process that has
    one. ``_configure_targeted_training`` runs after ``_compile_if_quantized``,
    so this is the path a rebuild restart on the fp8 family actually takes."""
    from app.engine.core.optimization.targeted_training import TargetedLayerManager

    compiled = _CompiledShim(make_peft_tiny(4))
    TargetedLayerManager(["^blocks\\.1\\.to_v$"]).apply(compiled)
    trainable = {
        n for n, p in compiled.named_parameters() if ".lora_" in n and p.requires_grad
    }
    assert trainable == {
        n
        for n, _ in compiled.named_parameters()
        if ".lora_" in n and "blocks.1.to_v" in n
    }


def test_restore_clamps_next_event_to_the_resume_point(
    make_adaptive_controller, train_step
):
    """A checkpoint whose ``next_event`` predates the resume point must not fire
    an event on the next step: that measurement window would be one step long,
    and ranking modules on it is exactly the freeze-on-noise the zero-heat guard
    exists to prevent."""
    model, ctl, _writer = make_adaptive_controller(energy_threshold=0.90)
    for step in range(1, 21):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    state = ctl.get_state()
    assert state["next_event"] == 30  # the schedule this run was on

    resumed_model, resumed, writer = make_adaptive_controller(energy_threshold=0.90)
    resumed.restore_state(state, current_step=500)
    for step in range(501, 510):  # a full interval must elapse first
        train_step(resumed_model, ["blocks.0.to_q"])
        resumed.on_optimizer_step(step)
    assert not [d for t, d in writer.events if t == "adapt"]

    train_step(resumed_model, ["blocks.0.to_q"])
    resumed.on_optimizer_step(510)  # resume point + interval_steps
    assert [d["kind"] for t, d in writer.events if t == "adapt"] == ["narrow"]


def test_restore_without_a_step_keeps_the_saved_schedule(make_adaptive_controller):
    """Backward-compatible signature: callers that cannot name a resume step
    (and every existing one) keep the persisted schedule verbatim."""
    _model, ctl, _writer = make_adaptive_controller()
    ctl.restore_state({"next_event": 77})
    assert ctl.get_state()["next_event"] == 77


def _segment_two_controller(model, tmp_path, writer, **cfg):
    """A controller for the NEXT rebuild segment over an already-narrowed model."""
    return AdaptiveTargetingController(
        model=model,
        config=AdaptiveTargetingConfig(
            warmup_pct=0.1, interval_steps=10, probe_steps=5, **cfg
        ),
        total_steps=100,
        log_writer=writer,
        output_dir=str(tmp_path),
    )


def test_floor_is_anchored_on_the_original_universe_across_a_rebuild(
    tmp_path, make_adaptive_controller, make_peft_tiny, fake_log_writer, train_step
):
    """``min_active_pct`` is a share of the run's ORIGINAL universe, not of
    whatever the previous rebuild segment narrowed it to.

    A rebuild restart re-applies the keep-set as the process's manual
    ``targeted_layers``, so the next controller's universe IS the previous
    keep-set. Recomputing the floor from that already-narrowed universe makes
    the guarantee a fraction of a fraction — over the capped rebuild cycles it
    collapses toward a single module, while the UI still promises "never leave
    fewer than this share of LoRA modules active".
    """
    from app.engine.core.optimization.targeted_training import TargetedLayerManager

    model, ctl, _writer = make_adaptive_controller(
        energy_threshold=0.5, min_active_pct=0.5
    )
    for step in range(1, 45):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    state = ctl.get_state()
    assert len(state["active_modules"]) == 4  # ceil(0.5 * 8), segment one's floor
    assert state["original_total"] == 8  # carried across the restart

    # Segment two: the relaunched process freezes everything outside the
    # keep-set FIRST, so its universe is only those four modules.
    narrowed = make_peft_tiny(4)
    TargetedLayerManager(ctl.keep_patterns()).apply(narrowed)
    ctl2 = _segment_two_controller(
        narrowed, tmp_path, fake_log_writer,
        energy_threshold=0.5, min_active_pct=0.5,
    )
    assert ctl2.total_count == 4
    ctl2.restore_state(state)

    # Segment two picks the schedule up where segment one left it (next event
    # at 50), so these steps really do run analysis events — a loop that ends
    # before the restored next_event would assert nothing at all.
    for step in range(41, 75):
        train_step(narrowed, ["blocks.0.to_q"])
        ctl2.on_optimizer_step(step)
    assert [d["kind"] for t, d in fake_log_writer.events if t == "adapt"]
    # A floor recomputed over the narrowed universe would be ceil(0.5*4) = 2.
    assert ctl2.active_count == 4


def test_restore_seeds_prior_events_and_prunes_the_discarded_future(
    tmp_path, make_adaptive_controller, train_step
):
    """The run-dir history file is the durable record, and every process starts
    with an empty event list — so without seeding, the first event after a
    resume (and after EVERY rebuild restart) rewrites the file with only the
    current segment. Events past the resume point are dropped: rewinding to an
    earlier checkpoint discards those steps, and keeping their events would
    advertise decisions this run never made."""
    path = tmp_path / "adaptive_targeting.json"
    path.write_text(
        json.dumps({
            "events": [
                {"step": 100, "event_index": 0, "kind": "narrow"},
                {"step": 200, "event_index": 1, "kind": "rebuild_request"},
                {"step": 900, "event_index": 2, "kind": "narrow"},
            ],
            "modules": [],
            "heat": {},
        }),
        encoding="utf-8",
    )

    model, ctl, _writer = make_adaptive_controller(energy_threshold=0.90)
    ctl.restore_state({"event_index": 2}, current_step=200)
    for step in range(201, 215):  # next_event is clamped to 210
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)

    with open(path, encoding="utf-8") as fh:
        events = json.load(fh)["events"]
    steps = [e["step"] for e in events]
    assert steps[:2] == [100, 200]  # the earlier segments survived
    assert 900 not in steps  # the discarded future did not
    assert len(steps) > 2  # …and this segment appended to them


def test_a_rebuild_request_that_never_shipped_is_not_persisted(
    tmp_path, make_adaptive_controller, train_step
):
    """The pipeline only restarts on an ``adapt`` payload it actually received.
    An event recorded before a failing emit would still be written to the
    history file by the NEXT event — a rebuild row for a restart that never
    happened, in the file the Stats table reads."""
    model, ctl, writer = make_adaptive_controller(
        energy_threshold=0.90, action="rebuild", rebuild_min_shrink_pct=1.0
    )
    for step in range(1, 21):
        train_step(model, ["blocks.0.to_q"])
        assert ctl.on_optimizer_step(step) in (None, "rebuild_request")
    assert ctl._pending_rebuild_step == 20  # the event asked for a rebuild

    working_emit = writer.emit
    writer.emit = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("channel down"))
    with pytest.raises(RuntimeError):
        ctl.notify_rebuild_checkpoint("checkpoint-000020")
    writer.emit = working_emit

    for step in range(21, 31):  # the next event rewrites the history file
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)

    with open(os.path.join(str(tmp_path), "adaptive_targeting.json"), encoding="utf-8") as fh:
        kinds = [e["kind"] for e in json.load(fh)["events"]]
    assert kinds  # not vacuous: the surviving events were written
    assert "rebuild_request" not in kinds


def test_unreadable_history_file_warns_and_still_restores(
    tmp_path, make_adaptive_controller
):
    """A corrupt history file must degrade to "history restarts here" with the
    reason surfaced — never take the resume (and the run) down with it."""
    (tmp_path / "adaptive_targeting.json").write_text("{ not json", encoding="utf-8")
    _model, ctl, writer = make_adaptive_controller()
    ctl.restore_state({"event_index": 3, "next_event": 400}, current_step=200)

    assert ctl.enabled is True
    assert ctl.event_index == 3  # the rest of the restore still happened
    warnings = [str(d) for t, d in writer.events if t == "warning"]
    assert any("adaptive_targeting.json" in w for w in warnings)


def test_a_cold_projection_type_keeps_a_representative(
    make_adaptive_controller, train_step, lora_params
):
    """End-to-end guard for the grouping fix, on real requires_grad flags.

    The fixture's universe is 4 blocks x (to_q, to_v). Training EVERY to_q and
    no to_v is the shape of the real defect: one projection carries all the
    measured movement, so a single global ranking fills the whole keep-set —
    floor included — from that projection and retires the other pathway
    entirely. Per-group selection gives to_v its own floor.
    """
    model, ctl, _writer = make_adaptive_controller(energy_threshold=0.90)
    hot = [f"blocks.{i}.to_q" for i in range(4)]
    for step in range(1, 21):
        train_step(model, hot)
        ctl.on_optimizer_step(step)

    assert ctl.active_count < 8  # it did narrow
    every = [f"blocks.{i}.{p}" for i in range(4) for p in ("to_q", "to_v")]
    live = [
        n for n in every if all(p.requires_grad for p in lora_params(model, n))
    ]
    assert any("to_v" in n for n in live), "the whole to_v pathway was frozen"
    assert any("to_q" in n for n in live)


@pytest.mark.parametrize("document", ["[]", "null", "3", '"events"'])
def test_history_that_is_not_an_object_degrades_instead_of_disabling(
    tmp_path, make_adaptive_controller, document
):
    """Valid JSON that is not an object is the same class of problem as a
    truncated file and must degrade the same way. Reaching for ``.get`` on it
    raises AttributeError, which no except clause here names — it escapes to
    the pipeline's setup guard and turns a cosmetic file into "adaptive
    targeting is off for this whole segment"."""
    (tmp_path / "adaptive_targeting.json").write_text(document, encoding="utf-8")
    _model, ctl, writer = make_adaptive_controller()
    ctl.restore_state({"event_index": 3, "next_event": 400}, current_step=200)

    assert ctl.enabled is True
    assert ctl.event_index == 3  # the rest of the restore still happened
    warnings = [str(d) for t, d in writer.events if t == "warning"]
    assert any("adaptive_targeting.json" in w for w in warnings)


def test_a_failed_seed_does_not_discard_the_rest_of_the_restore(
    tmp_path, make_adaptive_controller, train_step, lora_params, monkeypatch
):
    """Re-adopting the event history is the least critical part of a restore
    and the only part that reads a file this process did not write. If it runs
    first, an error nobody anticipated takes the narrowed active set, the heat
    and the schedule clamp down with it, and the run resumes wide open."""
    model, ctl, _writer = make_adaptive_controller(energy_threshold=0.90)
    for step in range(1, 25):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    state = ctl.get_state()
    frozen = [
        f"blocks.{i}.{proj}"
        for i in range(4)
        for proj in ("to_q", "to_v")
        if f"blocks.{i}.{proj}" not in state["active_modules"]
    ]
    assert frozen  # not vacuous

    (tmp_path / "adaptive_targeting.json").write_text("{}", encoding="utf-8")
    # Break the layer BELOW the seeder with an error type its except clause
    # deliberately does not name, standing in for the unforeseen failure.
    monkeypatch.setattr(json, "load", _boom)

    resumed_model, resumed, _w = make_adaptive_controller(energy_threshold=0.90)
    with pytest.raises(RecursionError):
        resumed.restore_state(state, current_step=24)

    assert resumed.event_index == ctl.event_index
    assert resumed.keep_patterns() == ctl.keep_patterns()
    for name in frozen:
        assert all(not p.requires_grad for p in lora_params(resumed_model, name))


def _boom(*_args, **_kwargs):
    raise RecursionError("unforeseen")


def test_an_applied_narrowing_survives_a_dead_log_channel(
    tmp_path, make_adaptive_controller, train_step
):
    """The mirror of the rebuild-handoff rule above. A handoff is only real
    once it ships, so a failed emit must record nothing — but a narrowing is
    already applied to the model by the time we emit, and the history file is
    the only durable record of it. Losing it there makes the Stats timeline
    disagree with what the run actually trained."""
    model, ctl, writer = make_adaptive_controller(energy_threshold=0.90)
    working_emit = writer.emit

    def _adapt_channel_down(msg_type, payload):
        # Only the adapt payload fails. Killing the writer wholesale would also
        # kill the warning that reports the failure, which is a property of the
        # test double, not of the log writer this stands in for.
        if msg_type == "adapt":
            raise RuntimeError("channel down")
        working_emit(msg_type, payload)

    writer.emit = _adapt_channel_down
    for step in range(1, 21):
        train_step(model, ["blocks.0.to_q"])
        assert ctl.on_optimizer_step(step) is None  # the failure never kills the run
    assert ctl.active_count < 8  # the freeze WAS applied

    with open(os.path.join(str(tmp_path), "adaptive_targeting.json"), encoding="utf-8") as fh:
        events = json.load(fh)["events"]
    assert [e["kind"] for e in events] == ["narrow"]
    assert events[0]["active_count"] == ctl.active_count
    warnings = [str(d) for t, d in writer.events if t == "warning"]
    assert any("channel down" in w for w in warnings)


def test_history_modules_span_the_run_not_only_this_segment(
    tmp_path, make_adaptive_controller, train_step
):
    """A rebuild restart discovers an already-narrowed universe. Writing that
    as the file's module list would name fewer modules than the file's own
    events refer to, so anything reading the two together (a post-run report,
    a future Stats view) would resolve a frozen module to nothing."""
    path = tmp_path / "adaptive_targeting.json"
    path.write_text(
        json.dumps({
            "events": [{"step": 100, "event_index": 0, "kind": "narrow"}],
            "modules": ["blocks.0.to_q", "retired.in.an.earlier.segment"],
            "heat": {},
        }),
        encoding="utf-8",
    )

    model, ctl, _writer = make_adaptive_controller(energy_threshold=0.90)
    ctl.restore_state({"event_index": 1}, current_step=100)
    for step in range(101, 125):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)

    modules = json.loads(path.read_text(encoding="utf-8"))["modules"]
    assert "retired.in.an.earlier.segment" in modules  # the earlier segment's
    assert "blocks.3.to_v" in modules  # …and this one's
    assert len(modules) == len(set(modules))  # union, not concatenation


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
