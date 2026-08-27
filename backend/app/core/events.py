
import asyncio
from typing import Any
from fastapi import WebSocket
import json
from app.core.logger import get_logger, ws_send_scope

logger = get_logger(__name__)

class EventManager:
    """
    Singleton Event Manager that handles WebSocket connections and broadcasts messages.

    W4.T9: each connection owns its own bounded outbound queue plus a
    dedicated sender task. ``broadcast()`` only ever enqueues — it never
    awaits a client's socket directly, so one stalled tab (a frozen browser
    tab, a full TCP send buffer) can never block the mutation route awaiting
    the emit, nor delay delivery to any other client. Per-client ordering is
    preserved (each queue is FIFO and has exactly one sender draining it); if
    a client falls behind and its queue fills up, the OLDEST queued message
    is dropped to make room for the newest one, so a slow client sees a
    coherent, if gappy, stream rather than reordered or duplicated events.
    """

    _QUEUE_MAXSIZE = 256

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()
        self._queues: dict[WebSocket, "asyncio.Queue[str]"] = {}
        self._sender_tasks: dict[WebSocket, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
            queue: "asyncio.Queue[str]" = asyncio.Queue(maxsize=self._QUEUE_MAXSIZE)
            self._queues[websocket] = queue
            self._sender_tasks[websocket] = asyncio.create_task(
                self._sender_loop(websocket, queue)
            )
        logger.info("websocket_client_connected", total_clients=len(self.active_connections))

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            task = self._sender_tasks.pop(websocket, None)
            self._queues.pop(websocket, None)
        if task is not None:
            task.cancel()
        logger.info("websocket_client_disconnected", total_clients=len(self.active_connections))

    async def _sender_loop(self, websocket: WebSocket, queue: "asyncio.Queue[str]"):
        """Drain *queue* and forward each message to *websocket*, in order.

        Runs as its own task per connection so a slow/stalled client's
        ``send_text`` await can never block ``broadcast()`` or any other
        connection's delivery. A send failure tears this connection down
        (mirrors the error handling ``broadcast()`` used to do inline).

        The send runs inside :func:`ws_send_scope` so the per-frame DEBUG
        traces uvicorn's websockets protocol logs during ``send_text`` are not
        mirrored back to clients — mirroring one enqueues another message here,
        which logs another trace: an unbounded feedback loop. ``broadcast()``
        used to inherit that guard implicitly by awaiting the send itself; this
        task does not, so it must set the scope explicitly.
        """
        try:
            while True:
                msg = await queue.get()
                try:
                    with ws_send_scope():
                        await websocket.send_text(msg)
                except Exception as e:
                    logger.warning("websocket_send_failed", error=str(e))
                    queue.task_done()
                    await self.disconnect(websocket)
                    return
                queue.task_done()
        except asyncio.CancelledError:
            pass

    async def broadcast(self, event_type: str, payload: dict[str, Any]):
        """
        Enqueues an event for delivery to every connected client.

        Never awaits a client's socket directly (see class docstring) — safe
        to call from any mutation route without risking a stall on a slow or
        frozen client.
        """
        message = {
            "type": event_type,
            "payload": payload,
            # Always called from inside the loop (broadcast is a coroutine), so
            # get_running_loop() is both correct and the non-deprecated form.
            "timestamp": asyncio.get_running_loop().time()
        }

        # We need to serialize once
        json_msg = json.dumps(message)

        for queue in list(self._queues.values()):
            self._enqueue_drop_oldest(queue, json_msg)

    @staticmethod
    def _enqueue_drop_oldest(queue: "asyncio.Queue[str]", msg: str) -> None:
        """``put_nowait``, dropping the OLDEST queued message to make room if
        *queue* is full — a client that falls behind loses backlog, not
        order (the newest event always wins a spot)."""
        try:
            queue.put_nowait(msg)
            return
        except asyncio.QueueFull:
            pass
        try:
            queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            pass  # lost a race with the sender draining it — fine, retry below
        try:
            queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass  # sender refilled it between our get and put — drop this one too

# Global Instance
event_manager = EventManager()

# Public re-export so callers can `from app.core.events import emit_entity_change`
from app.core.entity_events import emit_entity_change  # noqa: E402, F401
