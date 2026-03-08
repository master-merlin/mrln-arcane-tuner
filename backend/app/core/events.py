
import asyncio
from typing import Any
from fastapi import WebSocket
import json
from app.core.logger import get_logger

logger = get_logger(__name__)

class EventManager:
    """
    Singleton Event Manager that handles WebSocket connections and broadcasts messages.
    """
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info("websocket_client_connected", total_clients=len(self.active_connections))

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info("websocket_client_disconnected", total_clients=len(self.active_connections))

    async def broadcast(self, event_type: str, payload: dict[str, Any]):
        """
        Broadcasts an event to all connected clients.
        """
        message = {
            "type": event_type,
            "payload": payload,
            "timestamp": asyncio.get_event_loop().time() # or standard time
        }
        
        # We need to serialize once
        json_msg = json.dumps(message)
        
        to_remove = []
        
        # Determine connections to iterate over (snapshot to avoid modification during iteration issues if we yielded)
        # But here we just iterate list.
        # Note: sending over websocket is async.
        
        current_connections = list(self.active_connections)
        
        if not current_connections:
            return

        # logger.debug("broadcasting_event", type=event_type, clients=len(current_connections))

        for connection in current_connections:
            try:
                await connection.send_text(json_msg)
            except Exception as e:
                logger.warning("websocket_send_failed", error=str(e))
                to_remove.append(connection)

        if to_remove:
            async with self._lock:
                for conn in to_remove:
                    if conn in self.active_connections:
                        self.active_connections.remove(conn)

# Global Instance
event_manager = EventManager()
