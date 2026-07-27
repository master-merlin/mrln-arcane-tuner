"""
Tests for the EventManager WebSocket event broadcaster.
Covers: connect/disconnect, broadcast delivery, error handling.
"""

import asyncio
import json

from unittest.mock import AsyncMock

from app.core.events import EventManager


def _drain(loop: asyncio.AbstractEventLoop) -> None:
    """Let per-connection sender tasks (W4.T9) actually run and deliver.

    ``broadcast()`` only enqueues — delivery happens on each connection's own
    sender task, so tests that assert ``send_text`` was called must give that
    task a turn on the loop first.
    """
    loop.run_until_complete(asyncio.sleep(0.05))


def _teardown(loop: asyncio.AbstractEventLoop, em: EventManager, *connections) -> None:
    """Disconnect every connection and let cancellation land before closing
    the loop — otherwise each connection's sender task (W4.T9) is left
    pending and gets destroyed with the loop, which is harmless but noisy
    ("Task was destroyed but it is pending!")."""
    for ws in connections:
        loop.run_until_complete(em.disconnect(ws))
    loop.run_until_complete(asyncio.sleep(0))
    loop.close()


# ── Connection Management ────────────────────────────────────────────────


class TestEventManagerConnections:
    """Tests for connect/disconnect lifecycle."""

    def test_connect_accepts_and_tracks(self):
        """connect() should accept the websocket and add it to active_connections."""
        em = EventManager()
        ws = AsyncMock()

        loop = asyncio.new_event_loop()
        loop.run_until_complete(em.connect(ws))

        ws.accept.assert_awaited_once()
        assert ws in em.active_connections
        _teardown(loop, em, ws)

    def test_disconnect_removes_connection(self):
        """disconnect() should remove the websocket from active_connections."""
        em = EventManager()
        ws = AsyncMock()

        loop = asyncio.new_event_loop()
        loop.run_until_complete(em.connect(ws))
        loop.run_until_complete(em.disconnect(ws))

        assert ws not in em.active_connections
        _teardown(loop, em)

    def test_disconnect_ignores_unknown(self):
        """disconnect() should not raise for a websocket that was never connected."""
        em = EventManager()
        ws = AsyncMock()

        loop = asyncio.new_event_loop()
        loop.run_until_complete(em.disconnect(ws))

        assert len(em.active_connections) == 0
        _teardown(loop, em)

    def test_multiple_connections_tracked(self):
        """Multiple websockets should all be tracked."""
        em = EventManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        loop = asyncio.new_event_loop()
        loop.run_until_complete(em.connect(ws1))
        loop.run_until_complete(em.connect(ws2))

        assert len(em.active_connections) == 2
        _teardown(loop, em, ws1, ws2)


# ── Broadcast ────────────────────────────────────────────────────────────


class TestEventManagerBroadcast:
    """Tests for event broadcasting."""

    def test_broadcast_sends_to_all(self):
        """broadcast() should send to every connected client."""
        em = EventManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        loop = asyncio.new_event_loop()
        loop.run_until_complete(em.connect(ws1))
        loop.run_until_complete(em.connect(ws2))
        loop.run_until_complete(em.broadcast("test_event", {"value": 42}))
        _drain(loop)

        ws1.send_text.assert_awaited_once()
        ws2.send_text.assert_awaited_once()
        _teardown(loop, em, ws1, ws2)

    def test_broadcast_message_format(self):
        """Broadcast message should contain type, payload, and timestamp."""
        em = EventManager()
        ws = AsyncMock()

        loop = asyncio.new_event_loop()
        loop.run_until_complete(em.connect(ws))
        loop.run_until_complete(em.broadcast("training_step", {"step": 10, "loss": 0.5}))
        _drain(loop)

        sent = ws.send_text.call_args[0][0]
        msg = json.loads(sent)
        assert msg["type"] == "training_step"
        assert msg["payload"]["step"] == 10
        assert "timestamp" in msg
        _teardown(loop, em, ws)

    def test_broadcast_noop_with_no_connections(self):
        """broadcast() with no connections should not raise."""
        em = EventManager()

        loop = asyncio.new_event_loop()
        loop.run_until_complete(em.broadcast("test_event", {}))
        loop.close()

    def test_broadcast_removes_failed_connections(self):
        """Connections that fail on send should be auto-removed."""
        em = EventManager()
        good_ws = AsyncMock()
        bad_ws = AsyncMock()
        bad_ws.send_text.side_effect = ConnectionError("gone")

        loop = asyncio.new_event_loop()
        loop.run_until_complete(em.connect(good_ws))
        loop.run_until_complete(em.connect(bad_ws))
        loop.run_until_complete(em.broadcast("test_event", {"data": "hello"}))
        _drain(loop)

        # Failed connection should have been removed
        assert bad_ws not in em.active_connections
        # Good connection should remain
        assert good_ws in em.active_connections
        _teardown(loop, em, good_ws)


# ── Per-connection queues (W4.T9) ────────────────────────────────────────


class TestEventManagerPerConnectionQueues:
    """A stalled connection must never block broadcast() or another client's
    delivery — each connection gets its own outbound queue + sender task."""

    def test_broadcast_does_not_block_on_a_stalled_connection(self):
        """RED (pre-fix): broadcast() awaited `send_text` in a for-loop, so a
        connection whose send never resolves hung the whole call — every
        other client's delivery (and the mutation route awaiting the emit)
        waited behind it. GREEN (post-fix): broadcast() only enqueues, so it
        returns promptly regardless, and the healthy client still gets the
        event via its own sender task."""
        em = EventManager()
        stalled_ws = AsyncMock()
        healthy_ws = AsyncMock()

        never_releases = asyncio.Event()

        async def hang_forever(_msg):
            await never_releases.wait()

        stalled_ws.send_text.side_effect = hang_forever

        loop = asyncio.new_event_loop()
        # Connect the stalled client FIRST — under the old sequential
        # for-loop implementation this guarantees the hang is hit before the
        # healthy client is ever reached.
        loop.run_until_complete(em.connect(stalled_ws))
        loop.run_until_complete(em.connect(healthy_ws))

        loop.run_until_complete(
            asyncio.wait_for(em.broadcast("test_event", {"v": 1}), timeout=1.0)
        )

        _drain(loop)
        healthy_ws.send_text.assert_awaited_once()

        # Let the stalled sender task unwind cleanly instead of leaking.
        never_releases.set()
        _drain(loop)
        _teardown(loop, em, stalled_ws, healthy_ws)

    def test_broadcast_drops_oldest_message_when_queue_full_preserving_order(self):
        """A client that falls behind (queue full) loses its OLDEST
        undelivered messages, never gets them reordered or duplicated: the
        retained tail is always the newest messages, still in FIFO order."""
        em = EventManager()
        ws = AsyncMock()

        # Freeze delivery so every broadcast just piles up in the queue.
        gate = asyncio.Event()

        async def wait_for_gate(_msg):
            await gate.wait()

        ws.send_text.side_effect = wait_for_gate

        loop = asyncio.new_event_loop()
        loop.run_until_complete(em.connect(ws))
        queue = em._queues[ws]

        total = EventManager._QUEUE_MAXSIZE + 50
        for i in range(total):
            loop.run_until_complete(em.broadcast("evt", {"i": i}))

        # Bounded — never grew past its configured cap.
        assert queue.qsize() <= EventManager._QUEUE_MAXSIZE

        remaining = []
        while not queue.empty():
            remaining.append(json.loads(queue.get_nowait())["payload"]["i"])

        # Still ascending (no reordering) and it's a contiguous suffix of the
        # full 0..total-1 sequence (oldest dropped, never newest/no gaps).
        assert remaining == sorted(remaining)
        assert remaining == list(range(total - len(remaining), total))

        gate.set()
        _drain(loop)
        _teardown(loop, em, ws)
