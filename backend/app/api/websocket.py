"""WebSocket endpoint for real-time event streaming and system metrics."""

import asyncio
import json
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.events import event_manager
from app.core.logger import get_logger

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
            await websocket.send_text(msg)
            await asyncio.sleep(interval_s)
    except Exception:
        pass  # client disconnected — task will be cleaned up


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
