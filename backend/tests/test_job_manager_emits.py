"""Job manager broadcasts entity.changed on mutations."""
import asyncio
from unittest.mock import patch, AsyncMock

import pytest

from app.core.job_manager import JobManager


@pytest.fixture
def job_manager_with_loop():
    """Build a JobManager bound to the current event loop.

    No DB setup needed: `_persist_delete` swallows exceptions, so even if
    the underlying repository call fails the broadcast still fires.
    """
    mgr = JobManager()
    mgr._loop = asyncio.get_event_loop()
    return mgr


@pytest.mark.asyncio
async def test_delete_job_broadcasts_entity_changed(job_manager_with_loop):
    mgr = job_manager_with_loop
    mock_broadcast = AsyncMock()

    with patch("app.core.job_manager.event_manager.broadcast", mock_broadcast):
        mgr.delete_job("nonexistent-job-id")
        # delete_job schedules broadcast via run_coroutine_threadsafe;
        # yield to the loop so it actually runs.
        await asyncio.sleep(0.05)

    # Filter to the entity.changed broadcast (delete_job may not emit anything
    # else, but be robust to additional broadcasts being added later).
    entity_calls = [
        c for c in mock_broadcast.await_args_list
        if c.args and c.args[0] == "entity.changed"
    ]
    assert len(entity_calls) == 1, (
        f"expected exactly one entity.changed broadcast, "
        f"got {len(entity_calls)} from {mock_broadcast.await_args_list}"
    )
    payload = entity_calls[0].args[1]
    assert payload == {
        "entity": "job",
        "op": "deleted",
        "id": "nonexistent-job-id",
        "payload": None,
    }


@pytest.mark.asyncio
async def test_create_job_broadcasts_entity_changed(job_manager_with_loop):
    """Job creation broadcasts entity.changed with op=created and the full payload."""
    mgr = job_manager_with_loop
    mock_broadcast = AsyncMock()

    with patch("app.core.job_manager.event_manager.broadcast", mock_broadcast):
        job = mgr.create_job(plugin_id="test_plugin", config={"foo": "bar"})
        # create_job schedules broadcast via run_coroutine_threadsafe;
        # yield to the loop so it actually runs.
        await asyncio.sleep(0.05)

    entity_calls = [
        c for c in mock_broadcast.await_args_list
        if c.args and c.args[0] == "entity.changed"
    ]
    created = [c for c in entity_calls if c.args[1]["op"] == "created"]
    assert len(created) == 1, (
        f"expected one created event, got {len(created)} from {entity_calls}"
    )
    envelope = created[0].args[1]
    assert envelope["entity"] == "job"
    assert envelope["id"] == job.id
    assert envelope["payload"]["id"] == job.id
