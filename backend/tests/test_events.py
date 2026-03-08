"""
Tests for the EventManager WebSocket event broadcaster.
Covers: connect/disconnect, broadcast delivery, error handling.
"""

import asyncio
import json

from unittest.mock import AsyncMock

from app.core.events import EventManager


# ── Connection Management ────────────────────────────────────────────────


class TestEventManagerConnections:
    """Tests for connect/disconnect lifecycle."""

    def test_connect_accepts_and_tracks(self):
        """connect() should accept the websocket and add it to active_connections."""
        em = EventManager()
        ws = AsyncMock()

        asyncio.new_event_loop().run_until_complete(em.connect(ws))

        ws.accept.assert_awaited_once()
        assert ws in em.active_connections

    def test_disconnect_removes_connection(self):
        """disconnect() should remove the websocket from active_connections."""
        em = EventManager()
        ws = AsyncMock()

        loop = asyncio.new_event_loop()
        loop.run_until_complete(em.connect(ws))
        loop.run_until_complete(em.disconnect(ws))

        assert ws not in em.active_connections

    def test_disconnect_ignores_unknown(self):
        """disconnect() should not raise for a websocket that was never connected."""
        em = EventManager()
        ws = AsyncMock()

        asyncio.new_event_loop().run_until_complete(em.disconnect(ws))

        assert len(em.active_connections) == 0

    def test_multiple_connections_tracked(self):
        """Multiple websockets should all be tracked."""
        em = EventManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        loop = asyncio.new_event_loop()
        loop.run_until_complete(em.connect(ws1))
        loop.run_until_complete(em.connect(ws2))

        assert len(em.active_connections) == 2


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

        ws1.send_text.assert_awaited_once()
        ws2.send_text.assert_awaited_once()

    def test_broadcast_message_format(self):
        """Broadcast message should contain type, payload, and timestamp."""
        em = EventManager()
        ws = AsyncMock()

        loop = asyncio.new_event_loop()
        loop.run_until_complete(em.connect(ws))
        loop.run_until_complete(em.broadcast("training_step", {"step": 10, "loss": 0.5}))

        sent = ws.send_text.call_args[0][0]
        msg = json.loads(sent)
        assert msg["type"] == "training_step"
        assert msg["payload"]["step"] == 10
        assert "timestamp" in msg

    def test_broadcast_noop_with_no_connections(self):
        """broadcast() with no connections should not raise."""
        em = EventManager()

        asyncio.new_event_loop().run_until_complete(
            em.broadcast("test_event", {})
        )

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

        # Failed connection should have been removed
        assert bad_ws not in em.active_connections
        # Good connection should remain
        assert good_ws in em.active_connections
