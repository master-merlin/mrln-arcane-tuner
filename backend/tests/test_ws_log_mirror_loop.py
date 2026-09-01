"""Regression: the log->WebSocket mirror must never feed itself.

uvicorn hands its own ``uvicorn.error`` logger to the ``websockets`` protocol
(``uvicorn/protocols/websockets/websockets_impl.py``), so ``websockets`` writes
one ``> TEXT '…' [N bytes]`` DEBUG record *per frame it sends* under the name
``uvicorn.error`` — a name none of the ``websockets``-prefixed suppressions in
``app.core.logger`` match.

:class:`WebSocketLogHandler` mirrors log records to every client. Mirroring a
frame-trace record emits another frame, which logs another trace, which mirrors
again: an unbounded feedback loop that fills server.log at event-loop speed
(observed: 1.19M lines / 114 MB, 100% frame traces, with a single client).

The ``_in_ws_log`` ContextVar guard used to stop this because ``broadcast()``
awaited ``send_text`` inline, inside the guarded context that
``run_coroutine_threadsafe`` propagates. W4.T9 (per-connection send queues)
moved the send into a sender task created back in ``connect()`` — outside the
guard — so the cycle re-opened. These tests pin both halves of the fix.
"""
import asyncio
import logging
from unittest.mock import MagicMock, patch

from app.core.events import EventManager
from app.core.logger import WebSocketLogHandler, _in_ws_log, config_log_level


def _drain(loop: asyncio.AbstractEventLoop) -> None:
    loop.run_until_complete(asyncio.sleep(0.05))


def _teardown(loop: asyncio.AbstractEventLoop, em: EventManager, *connections) -> None:
    for ws in connections:
        loop.run_until_complete(em.disconnect(ws))
    # Let each sender task's cancellation actually land before closing the loop,
    # otherwise it is destroyed while pending and asyncio complains on stderr.
    loop.run_until_complete(asyncio.sleep(0.05))
    loop.close()


class TestSendPathIsGuarded:
    """Every socket write must run inside the log-mirror guard."""

    def test_sender_loop_send_runs_inside_guard(self):
        """The per-connection sender task must mark its ``send_text`` as a
        WS-send scope, or records logged by the send get mirrored back."""
        em = EventManager()
        seen = []

        class FakeWS:
            async def accept(self):
                pass

            async def send_text(self, msg):
                seen.append(_in_ws_log.get())

        ws = FakeWS()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(em.connect(ws))
        loop.run_until_complete(em.broadcast("server_log", {"message": "hi"}))
        _drain(loop)

        assert seen, "sender task never delivered the message"
        assert seen[0] is True, (
            "send_text ran with the log-mirror guard OFF — a frame-trace record "
            "logged during the send will be re-broadcast (feedback loop)"
        )
        _teardown(loop, em, ws)

    def test_frame_trace_logged_during_send_is_not_mirrored(self):
        """Full loop reproduction: a ``uvicorn.error`` frame trace emitted by
        the send itself must not schedule another broadcast."""
        em = EventManager()
        uv = logging.getLogger("uvicorn.error")

        handler = WebSocketLogHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))

        scheduled = []

        def _record_schedule(coro, _loop):
            coro.close()  # never awaited in this test — avoid a warning
            scheduled.append(coro)
            return MagicMock()

        class FakeWS:
            async def accept(self):
                pass

            async def send_text(self, msg):
                # Exactly what websockets/protocol.py:755 does per frame.
                uv.debug("> TEXT %s", f"'{msg[:40]}...' [229 bytes]")

        ws = FakeWS()
        loop = asyncio.new_event_loop()
        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = False

        prev_level, prev_prop = uv.level, uv.propagate
        uv.setLevel(logging.DEBUG)  # bypass the threshold floor on purpose
        uv.propagate = False
        uv.addHandler(handler)
        try:
            loop.run_until_complete(em.connect(ws))
            with patch("app.core.events.event_manager", em), \
                 patch("app.core.logger._log_loop", mock_loop), \
                 patch("app.core.logger.asyncio.run_coroutine_threadsafe",
                       _record_schedule):
                loop.run_until_complete(
                    em.broadcast("server_log", {"message": "real app log"})
                )
                _drain(loop)

            assert scheduled == [], (
                f"{len(scheduled)} frame-trace record(s) were mirrored back to "
                "clients — each mirror emits another frame, which logs another "
                "trace: unbounded log->WS->log feedback loop"
            )
        finally:
            uv.removeHandler(handler)
            uv.setLevel(prev_level)
            uv.propagate = prev_prop
            _teardown(loop, em, ws)


class TestUvicornErrorThreshold:
    """Defense in depth: per-frame protocol traces are not application logs."""

    def test_debug_level_does_not_enable_ws_frame_traces(self):
        """App-level DEBUG must not turn on ``uvicorn.error`` DEBUG — that
        stream is the websockets per-frame trace, which is pure noise in
        server.log and unparseable as JSON (_docs/LOGGING.md golden rule 1)."""
        try:
            config_log_level("DEBUG")

            assert logging.getLogger().level == logging.DEBUG, (
                "root logger must still honour DEBUG"
            )
            assert logging.getLogger("uvicorn.error").isEnabledFor(
                logging.DEBUG
            ) is False, (
                "uvicorn.error at DEBUG emits one '> TEXT …' record per "
                "WebSocket frame; it must stay floored at INFO"
            )
        finally:
            config_log_level("INFO")

    def test_uvicorn_error_still_reports_info_and_errors(self):
        """The floor is a threshold only — real uvicorn output survives."""
        try:
            config_log_level("DEBUG")
            uv = logging.getLogger("uvicorn.error")

            assert uv.isEnabledFor(logging.INFO) is True
            assert uv.isEnabledFor(logging.ERROR) is True
        finally:
            config_log_level("INFO")
