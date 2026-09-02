# backend/tests/test_refine_guard_probe_bound.py
"""LANE-70 — the readiness probe is bounded, in ONE place, to a few seconds.

The user, signing UAT round 7 (7.4): *"it takes a while until 'Generate'
becomes available when switching from OpenAI to ollama."* The CTA waits on
``refine_guard.endpoint_readiness``; before this lane the refine probe ran on
``OllamaClient``'s 120 s inference timeout and the caption probe on its own
10 s, so an endpoint that ACCEPTS the connection and never answers held the
button for that long. A model listing answers in well under a second or it
is not an endpoint the CTA should wait on — the sentence says "unreachable".

The endpoint under test is a REAL socket that listens and never accepts
(connections complete in the kernel backlog and then hang), the shape a
wedged server or a firewall's black-hole presents. Both entry points are
exercised — they share the one ``asyncio.wait_for`` in ``endpoint_readiness``
and nothing else bounds them. Mutation that turns this red: drop the
``wait_for`` (the test's own 3 s deadline fails, instead of hanging 120 s).
"""

from __future__ import annotations

import asyncio
import socket
import time

import pytest

from app.core.llm import provider_settings, refine_guard
from app.core.llm.ollama_client import OllamaClient

#: Patched-in bound for the behavioural tests: fast, and still a real wait.
_TEST_BOUND_S = 0.5
#: The test's own deadline — a removed bound fails HERE, never hangs the suite.
_DEADLINE_S = 3.0


@pytest.fixture
def silent_port():
    """A loopback port whose listener never reads or answers."""
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        yield srv.getsockname()[1]


class _FakeSettingsManager:
    def __init__(self, modules: dict[str, dict]) -> None:
        self.modules = modules

    def get_module_settings(self, module):
        return self.modules.get(module, {})

    def update_module_settings(self, module, settings):
        self.modules.setdefault(module, {}).update(settings)


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, _DEADLINE_S))


def test_shipped_bound_is_a_few_seconds() -> None:
    """The constant the UI waits on: single digits, never the client's 120 s."""
    assert 0 < refine_guard.PROBE_TIMEOUT_S <= 5.0


def test_refine_probe_of_a_silent_endpoint_is_unreachable_within_the_bound(
        monkeypatch, silent_port) -> None:
    monkeypatch.setattr(refine_guard, "PROBE_TIMEOUT_S", _TEST_BOUND_S)
    url = f"http://127.0.0.1:{silent_port}"

    t0 = time.perf_counter()
    ready = _run(refine_guard.refine_readiness(OllamaClient(base_url=url)))
    elapsed = time.perf_counter() - t0

    assert ready.available is False
    assert ready.reason == refine_guard.unreachable_reason(url), ready.reason
    assert elapsed < _TEST_BOUND_S + 1.5, f"probe took {elapsed:.2f}s"


def test_caption_probe_of_a_silent_endpoint_is_unreachable_within_the_bound(
        monkeypatch, silent_port) -> None:
    monkeypatch.setattr(refine_guard, "PROBE_TIMEOUT_S", _TEST_BOUND_S)
    base = f"http://127.0.0.1:{silent_port}/v1"
    mgr = _FakeSettingsManager(
        {provider_settings.MODULE: {"providers": {"custom": {"base_url": base}}}})
    monkeypatch.setattr(provider_settings, "_manager", lambda: mgr)

    t0 = time.perf_counter()
    ready = _run(refine_guard.caption_provider_readiness("custom", "llava:13b"))
    elapsed = time.perf_counter() - t0

    assert ready.available is False
    assert ready.reason == refine_guard.unreachable_reason(
        base, refine_guard.CAPTION_SURFACE), ready.reason
    assert elapsed < _TEST_BOUND_S + 1.5, f"probe took {elapsed:.2f}s"


def test_positive_control_an_answering_endpoint_is_judged_not_timed_out(
        monkeypatch, fake_ollama) -> None:
    """The bound must not eat a live answer: same bound, real listing, verdict
    on the model — proves the timeout wraps the call rather than replacing it."""
    monkeypatch.setattr(refine_guard, "PROBE_TIMEOUT_S", _TEST_BOUND_S)
    fake_ollama.models[:] = ["qwen2.5:7b-instruct"]

    ready = _run(refine_guard.refine_readiness(
        OllamaClient(base_url=fake_ollama.url), "qwen2.5:7b-instruct"))
    assert ready.available is True and ready.reason is None

    missing = _run(refine_guard.refine_readiness(
        OllamaClient(base_url=fake_ollama.url), "gemma3:12b"))
    assert missing.reason == refine_guard.model_missing_reason(
        "gemma3:12b", fake_ollama.url)
