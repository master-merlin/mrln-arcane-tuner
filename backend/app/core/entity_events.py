"""Typed `entity.changed` WebSocket event envelope + emit helper.

Used by domain managers (job_manager, dataset_manager, settings_manager, ...)
to broadcast generic CRUD events that the frontend EntityStore subscribes to.
"""
from typing import Any, Awaitable, Callable, Literal, TypedDict

EntityName = Literal[
    "job", "dataset", "media_item", "settings", "registry_model", "overlay",
    "project", "template",
]
EntityOp = Literal["created", "updated", "deleted", "bulk_deleted"]


class EntityChangedPayload(TypedDict):
    entity: str
    op: str
    id: str
    payload: dict[str, Any] | None


BroadcastFn = Callable[[str, dict[str, Any]], Awaitable[None]]


async def emit_entity_change(
    broadcast: BroadcastFn,
    *,
    entity: EntityName,
    op: EntityOp,
    id: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Broadcast an `entity.changed` event with a typed payload.

    Pass `event_manager.broadcast` as `broadcast`. We accept it as an
    argument (rather than importing event_manager directly) to keep this
    module testable without touching global state.
    """
    envelope: EntityChangedPayload = {
        "entity": entity,
        "op": op,
        "id": id,
        "payload": payload,
    }
    await broadcast("entity.changed", dict(envelope))
