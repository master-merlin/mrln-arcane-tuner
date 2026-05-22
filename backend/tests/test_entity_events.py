"""Unit tests for the typed entity.changed event helper."""
from unittest.mock import AsyncMock
import pytest
from app.core.entity_events import emit_entity_change


@pytest.mark.asyncio
async def test_emit_entity_change_calls_broadcast_with_envelope():
    broadcast = AsyncMock()
    await emit_entity_change(
        broadcast,
        entity="job",
        op="deleted",
        id="abc-123",
    )
    broadcast.assert_awaited_once_with(
        "entity.changed",
        {"entity": "job", "op": "deleted", "id": "abc-123", "payload": None},
    )


@pytest.mark.asyncio
async def test_emit_entity_change_includes_payload_on_create():
    broadcast = AsyncMock()
    row = {"id": "abc-123", "name": "test"}
    await emit_entity_change(broadcast, entity="dataset", op="created", id="abc-123", payload=row)
    broadcast.assert_awaited_once_with(
        "entity.changed",
        {"entity": "dataset", "op": "created", "id": "abc-123", "payload": row},
    )


@pytest.mark.asyncio
async def test_emit_entity_change_bulk_deleted_carries_ids():
    broadcast = AsyncMock()
    await emit_entity_change(
        broadcast, entity="job", op="bulk_deleted", id="", payload={"ids": ["a", "b", "c"]}
    )
    broadcast.assert_awaited_once_with(
        "entity.changed",
        {"entity": "job", "op": "bulk_deleted", "id": "", "payload": {"ids": ["a", "b", "c"]}},
    )
