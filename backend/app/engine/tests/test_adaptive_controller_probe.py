"""Re-activation (probe window) run path of AdaptiveTargetingController (spec §5).

Freeze mode is monotonic; a probe window deliberately is not — that is its whole
point. Every ``probe_every``-th event reopens the FULL universe for a bounded
number of optimizer steps, measures heat over exactly that window, and applies a
keep-set that MAY re-admit a module an earlier event froze.

Every assertion is on observable state — ``requires_grad`` flags, the emitted
``adapt`` payloads, ``get_state()`` — never on "a method was called".

Shared fixtures (``make_adaptive_controller``, ``train_step``, ``lora_params``,
``make_peft_tiny``, ``fake_log_writer``) live in this directory's ``conftest.py``.
"""

from app.engine.components.adaptive_targeting import AdaptiveTargetingController
from app.engine.models.adaptive import AdaptiveTargetingConfig


def _cfg(**overrides) -> dict:
    """Re-activation knobs: probe on every 2nd event, over a 4-step window."""
    return {
        "reactivation": True,
        "probe_every": 2,
        "probe_steps": 4,
        "energy_threshold": 0.90,
        **overrides,
    }


def _adapt(writer) -> list[dict]:
    """The ``adapt`` payloads the UI would have received, in emission order."""
    return [data for msg_type, data in writer.events if msg_type == "adapt"]


def test_probe_reopens_full_universe_then_reapplies(
    make_adaptive_controller, train_step, lora_params
):
    """The end-to-end re-activation path: narrow, probe, re-admit."""
    model, ctl, writer = make_adaptive_controller(**_cfg())

    # Warmup 10% of 100 = step 10, interval 10 → event 1 at step 20, which is
    # always a regular narrow (a probe never opens before the universe has been
    # narrowed at least once). Each loop stops ON its event step: a step past it
    # would leak into the next window and shift what that window measures.
    for step in range(1, 21):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    assert ctl.active_count < 8
    cold = lora_params(model, "blocks.3.to_v")
    assert cold and all(not p.requires_grad for p in cold)

    # Event 2 is the probe: the whole universe goes temporarily trainable.
    for step in range(21, 31):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    assert [event["kind"] for event in _adapt(writer)][-1] == "probe_open"
    assert ctl.active_count == 8
    assert all(p.requires_grad for p in cold)  # the SAME parameter objects

    # Only blocks.3.to_v learns during the probe window. train_step skips frozen
    # params, so this measures real heat exactly because the probe unfroze it.
    for step in range(31, 35):  # probe_steps=4 → the window closes ON step 34
        train_step(model, ["blocks.3.to_v"])
        ctl.on_optimizer_step(step)

    applied = _adapt(writer)[-1]
    assert applied["kind"] == "probe_apply"
    assert applied["reactivated_this_event"] >= 1  # counted, not just applied
    # Re-activation: freeze mode's monotonic intersect could never do this.
    assert all(p.requires_grad for p in lora_params(model, "blocks.3.to_v"))
    assert "blocks.3.to_v" in ctl.get_state()["active_modules"]


def test_probe_never_unfreezes_outside_manual_universe(
    tmp_path, make_peft_tiny, fake_log_writer, train_step, lora_params
):
    """A probe reopens the UNIVERSE, which excludes the user's manual freeze.

    Checked on every step, not just at the end: a probe that unfroze a
    user-frozen module and a later event that re-froze it would leave no trace
    in a final-state assertion, but the run would still have trained it.
    """
    model = make_peft_tiny(4)
    for p in lora_params(model, "blocks.2"):
        p.requires_grad_(False)  # simulates a TargetedLayerManager manual freeze
    ctl = AdaptiveTargetingController(
        model=model,
        config=AdaptiveTargetingConfig(
            warmup_pct=0.1, interval_steps=10, min_active_pct=0.13, **_cfg()
        ),
        total_steps=100,
        log_writer=fake_log_writer,
        output_dir=str(tmp_path),
    )
    assert ctl.total_count == 6  # blocks.2's two modules are outside the universe

    for step in range(1, 35):  # narrow at 20, probe opens at 30 and closes at 34
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
        assert all(not p.requires_grad for p in lora_params(model, "blocks.2"))

    kinds = [event["kind"] for event in _adapt(fake_log_writer)]
    assert "probe_open" in kinds and "probe_apply" in kinds  # not vacuous


def test_interval_clock_pauses_during_probe(make_adaptive_controller, train_step):
    """No regular narrowing event may fire while a probe window is open.

    One firing mid-probe would truncate the probe's own measurement window AND
    apply a keep-set ranked over a universe that is only temporarily wide open.
    """
    model, ctl, writer = make_adaptive_controller(**_cfg(probe_steps=8))
    for step in range(1, 40):
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)

    events = _adapt(writer)
    opens = [e["step"] for e in events if e["kind"] == "probe_open"]
    applies = [e["step"] for e in events if e["kind"] == "probe_apply"]
    assert opens and applies  # the window really did open and close
    for open_step, apply_step in zip(opens, applies):
        assert apply_step > open_step
        assert not [e for e in events if open_step < e["step"] < apply_step]


def test_probe_without_signal_restores_the_pre_probe_set(
    make_adaptive_controller, train_step, lora_params
):
    """A probe that measured nothing restores the PRE-probe active set.

    ``self._active`` during a probe IS the wide-open universe, so "restoring"
    that would silently undo every narrowing decision the run had made.
    ``heat_ema=0`` drops the carried-over EMA, which is what lets the window read
    exactly zero when nothing moves.
    """
    model, ctl, writer = make_adaptive_controller(**_cfg(heat_ema=0.0))
    for step in range(1, 21):  # event 1: narrow
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    pre_probe = ctl.get_state()["active_modules"]
    assert 0 < len(pre_probe) < 8

    for step in range(21, 31):  # event 2: probe opens at step 30
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    assert ctl.active_count == 8

    # Nothing moves for the whole window (e.g. accumulation-only steps).
    for step in range(31, 35):
        ctl.on_optimizer_step(step)

    assert ctl.get_state()["active_modules"] == pre_probe
    assert all(not p.requires_grad for p in lora_params(model, "blocks.3.to_v"))
    assert "probe_apply" not in [event["kind"] for event in _adapt(writer)]
    assert any(  # never silent
        "no learning signal" in str(data)
        for msg_type, data in writer.events
        if msg_type == "log"
    )


def test_interrupted_probe_is_abandoned_on_resume(make_adaptive_controller, train_step):
    """A probe window cannot survive a stop: the baseline snapshot it was being
    measured against died with the process, so half a window would be compared
    to the resume point and read as near-zero heat across the board."""
    model, ctl, _writer = make_adaptive_controller(**_cfg())
    for step in range(1, 31):  # narrow at 20, probe opens at 30
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    state = ctl.get_state()
    assert state["probe_open_step"] == 30  # persisted, for diagnostics
    assert len(state["active_modules"]) == 8  # the probe holds it wide open

    resumed_model, resumed, resumed_writer = make_adaptive_controller(**_cfg())
    resumed.restore_state(state)
    assert resumed.get_state()["probe_open_step"] is None
    assert any(
        "probe window was open" in str(data)
        for msg_type, data in resumed_writer.events
        if msg_type == "log"
    )

    for step in range(31, 45):
        train_step(resumed_model, ["blocks.1.to_v"])
        resumed.on_optimizer_step(step)
    kinds = [event["kind"] for event in _adapt(resumed_writer)]
    assert kinds  # the resumed run really did keep running events
    assert "probe_apply" not in kinds  # nothing "closed" a probe it never opened


def test_failed_probe_close_releases_the_window(
    make_adaptive_controller, train_step, monkeypatch
):
    """A close that dies part-way must still release the window.

    A probe left open pauses the interval clock forever: every later event of
    the run is lost and it finishes with the universe wide open and unmeasured
    — a permanent wedge, not a one-off skipped event.
    """
    import app.engine.components.adaptive_targeting as mod

    model, ctl, writer = make_adaptive_controller(**_cfg())
    real_select = mod.select_active
    failing = {"on": False}

    def flaky(*args, **kwargs):
        if failing["on"]:
            raise RuntimeError("boom")
        return real_select(*args, **kwargs)

    monkeypatch.setattr(mod, "select_active", flaky)

    for step in range(1, 31):  # narrow at 20, probe opens at 30
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    failing["on"] = True
    for step in range(31, 35):  # the close at step 34 raises
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)

    assert ctl.enabled is True  # one failure is survivable
    assert ctl.get_state()["probe_open_step"] is None  # the window was released
    assert any(  # and the reason reached the log channel
        "boom" in str(data) for msg_type, data in writer.events if msg_type == "warning"
    )

    failing["on"] = False
    for step in range(35, 45):  # the interval clock runs again → event at 44
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)
    assert [event["kind"] for event in _adapt(writer)][-1] == "narrow"


def test_freeze_only_mode_never_opens_a_probe(make_adaptive_controller, train_step):
    """``reactivation`` is off by default — the probe path must stay dormant even
    when the probe knobs are set, or freeze mode silently becomes non-monotonic."""
    model, ctl, writer = make_adaptive_controller(
        energy_threshold=0.90, probe_every=2, probe_steps=4
    )
    for step in range(1, 60):  # events at 20, 30, 40, 50
        train_step(model, ["blocks.0.to_q"])
        ctl.on_optimizer_step(step)

    assert {event["kind"] for event in _adapt(writer)} == {"narrow"}
    assert ctl.get_state()["probe_open_step"] is None
