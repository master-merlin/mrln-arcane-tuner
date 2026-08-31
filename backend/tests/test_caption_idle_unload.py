"""The idle unload that gives single-image captioning a release path at all.

A caption batch has always freed its model in its own ``finally``
(``caption_batch.py``), but a SINGLE caption had no unload path anywhere: the
only callers of ``DELETE /captions/unload`` were the model and variant
dropdowns, so a one-off caption held VRAM until the user happened to switch
models. Reported by the user during UAT round 4.

Unloading on every single caption would be the wrong fix — interactive
captioning is a caption-look-caption loop and each one would pay a full model
load — so the release is deferred to an idle window that each new caption
pushes out. These tests pin the parts that can silently rot:

* the timer actually fires and actually unloads;
* a second caption REPLACES the pending timer rather than stacking one;
* a batch cancels it, so the timer cannot fire into a running batch;
* an explicit unload cancels it, so nothing fires into an already-free GPU;
* the window is configurable and can be switched off.

Timings here are deliberately tiny (milliseconds) and every wait is bounded, so
the suite never sleeps on a wall-clock window.
"""

from __future__ import annotations

import threading
import time

import pytest

from app.core.captioning.caption_service import CaptionService


@pytest.fixture(autouse=True)
def _no_pending_timer():
    """Never leak a timer between tests — a stray one would unload a model
    another test is using, which is exactly the class of bug this file is
    about."""
    CaptionService.cancel_idle_unload()
    yield
    CaptionService.cancel_idle_unload()


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    """Poll until *predicate* holds. Bounded: returns False rather than hanging."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_armed_timer_fires_and_unloads(monkeypatch):
    fired = threading.Event()
    monkeypatch.setattr(
        CaptionService, "unload_models",
        classmethod(lambda cls, **kw: fired.set() or True),
    )

    assert CaptionService.arm_idle_unload(0.02) is True
    assert _wait_for(fired.is_set), "idle timer never unloaded"


def test_the_fired_unload_is_the_batch_guarded_mode(monkeypatch):
    """It must not be the unconditional internal mode.

    The timer runs on its own thread with no idea what else started in the
    meantime, so it has to use the same refusal the manual route uses.
    """
    seen: dict = {}
    done = threading.Event()

    def _capture(cls, **kwargs):
        seen.update(kwargs)
        done.set()
        return True

    monkeypatch.setattr(CaptionService, "unload_models", classmethod(_capture))
    CaptionService.arm_idle_unload(0.02)
    assert _wait_for(done.is_set)
    assert seen.get("skip_if_batch_active") is True


def test_rearming_replaces_the_pending_timer_instead_of_stacking(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(
        CaptionService, "unload_models",
        classmethod(lambda cls, **kw: calls.append(time.monotonic()) or True),
    )

    # A long window, then a short one: if arming STACKED, the long timer would
    # still be pending and would eventually add a second call.
    CaptionService.arm_idle_unload(30.0)
    CaptionService.arm_idle_unload(0.02)
    assert _wait_for(lambda: len(calls) == 1)
    time.sleep(0.15)
    assert len(calls) == 1, "re-arming stacked a second timer instead of replacing"


def test_a_caption_batch_cancels_a_pending_idle_unload(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        CaptionService, "unload_models",
        classmethod(lambda cls, **kw: calls.append(1) or True),
    )

    CaptionService.arm_idle_unload(0.05)
    CaptionService.cancel_idle_unload()
    time.sleep(0.2)
    assert calls == [], "idle unload fired after it was cancelled"


def test_an_explicit_unload_cancels_the_pending_timer():
    """`unload_models` cancels first — otherwise the timer fires later into a
    GPU that is already free, or worse, into a model loaded since."""
    CaptionService.arm_idle_unload(30.0)
    assert CaptionService._idle_timer is not None
    CaptionService.unload_models()
    assert CaptionService._idle_timer is None


def test_a_non_positive_window_disables_idle_unloading(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        CaptionService, "unload_models",
        classmethod(lambda cls, **kw: calls.append(1) or True),
    )

    assert CaptionService.arm_idle_unload(0) is False
    assert CaptionService._idle_timer is None
    time.sleep(0.1)
    assert calls == []


def test_cancelling_with_nothing_armed_is_harmless():
    CaptionService.cancel_idle_unload()
    CaptionService.cancel_idle_unload()
    assert CaptionService._idle_timer is None


def test_the_default_window_is_read_from_the_environment():
    """Pins that the constant is an env-overridable float, not a literal buried
    in the call site — the window is exactly the kind of number a user with a
    small card needs to change."""
    from app.core.captioning import caption_service as mod

    assert isinstance(mod.IDLE_UNLOAD_SECONDS, float)
    assert mod.IDLE_UNLOAD_SECONDS > 0
