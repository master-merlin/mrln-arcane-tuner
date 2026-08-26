"""WebSocket endpoint for real-time event streaming and system metrics."""

import asyncio
import json
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.events import event_manager
from app.core.logger import get_logger, ws_send_scope

router = APIRouter()
logger = get_logger(__name__)

# Unique ID for this server process — changes on every restart
_SERVER_INSTANCE_ID = str(uuid.uuid4())

# Track clients that have subscribed to system metrics
_metrics_tasks: dict[int, asyncio.Task] = {}


async def _stream_metrics(websocket: WebSocket, interval_s: float = 2.0):
    """Background task: push system metrics to a single client."""
    from app.core.system_monitor import system_monitor

    try:
        while True:
            # NVML per-GPU queries + compute-process enumeration + a
            # psutil.Process(pid).name() lookup per process — on a WDDM box
            # under training load NVML can stall. Offload to a thread so a
            # slow snapshot() never blocks the event loop (W4.T10).
            snap = await asyncio.to_thread(system_monitor.snapshot)
            msg = json.dumps({
                "type": "system_metrics",
                "payload": snap.to_dict(),
            })
            # ws_send_scope: keep the frame trace this write logs out of the
            # log->WS mirror (see app.core.logger.ws_send_scope).
            with ws_send_scope():
                await websocket.send_text(msg)
            await asyncio.sleep(interval_s)
    except asyncio.CancelledError:
        # `unsubscribe_metrics` and shutdown cancel this task. That is the
        # normal exit, and it must PROPAGATE — swallowing a cancellation leaves
        # the canceller waiting on a task that reports itself still running.
        raise
    except (WebSocketDisconnect, ConnectionError, RuntimeError) as exc:
        # Expected: the client vanished mid-push, or Starlette refused a send on
        # an already-closed socket. Not a fault, so debug — but no longer
        # silent, because "the metrics stopped" and "the client left" were
        # previously indistinguishable from the outside.
        with ws_send_scope():
            logger.debug(
                "ws_metrics_stream_closed",
                client=str(websocket.client),
                error=type(exc).__name__,
            )
    except Exception as exc:
        # A real fault — a broken snapshot, a serialization error. Logged ONCE,
        # at WARNING, with the connection and the exception, then the loop
        # exits. Bounded by construction rather than by a rate limiter: there
        # is exactly one record per stream because there is no retry.
        #
        # The ws_send_scope wrapper is LOAD-BEARING and is not decoration.
        # Without it this record is mirrored to every WebSocket client, each
        # mirror emits a frame trace on `uvicorn.error`, and each trace is
        # itself mirrored — the log->WS->log loop that once wrote 114 MB /
        # 1.19M lines with a single client connected. Un-silencing this handler
        # is precisely the change that can reopen it.
        with ws_send_scope():
            logger.warning(
                "ws_metrics_stream_failed",
                client=str(websocket.client),
                error=str(exc),
                error_type=type(exc).__name__,
            )


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Accept a WebSocket connection and stream server events.

    Clients can send JSON messages to control behaviour:
    - ``{"action": "subscribe_metrics"}``  → start system metrics stream
    - ``{"action": "subscribe_metrics", "interval_s": 1.0}``  → custom interval
    - ``{"action": "unsubscribe_metrics"}`` → stop metrics stream
    """
    await event_manager.connect(websocket)
    logger.debug("ws_connected", client=str(websocket.client))
    ws_id = id(websocket)

    # Send server identity so frontend can detect restarts
    with ws_send_scope():
        await websocket.send_text(json.dumps({
            "type": "server_hello",
            "payload": {"instance_id": _SERVER_INSTANCE_ID},
        }))
    try:
        while True:
            raw = await websocket.receive_text()
            # Try to parse control messages
            try:
                msg = json.loads(raw)
                action = msg.get("action", "")

                if action == "subscribe_metrics":
                    # Cancel existing task if any
                    if ws_id in _metrics_tasks:
                        _metrics_tasks[ws_id].cancel()
                    interval = float(msg.get("interval_s", 2.0))
                    interval = max(0.5, min(interval, 30.0))  # clamp
                    _metrics_tasks[ws_id] = asyncio.create_task(
                        _stream_metrics(websocket, interval),
                    )

                elif action == "unsubscribe_metrics":
                    task = _metrics_tasks.pop(ws_id, None)
                    if task:
                        task.cancel()
            except (json.JSONDecodeError, ValueError):
                pass  # not a control message — ignore

    except WebSocketDisconnect:
        logger.debug("ws_disconnected", client=str(websocket.client))
        await event_manager.disconnect(websocket)
    except RuntimeError:
        logger.debug("ws_unexpected_close", client=str(websocket.client))
        await event_manager.disconnect(websocket)
    finally:
        # Clean up metrics task
        task = _metrics_tasks.pop(ws_id, None)
        if task:
            task.cancel()
