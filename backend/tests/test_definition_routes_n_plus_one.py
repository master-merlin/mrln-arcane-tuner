"""R-API-07: list_model_definitions must read settings.json at most once
per request, not once per registered model definition (N+1 regression guard).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.engine.core.definitions import ModelDefinition
from app.engine.models.registry import registry
from app.engine.utils import model_override_manager as mom


@pytest.fixture
def seeded_definitions():
    """Seed the registry with 3 throwaway definitions so the N+1 bug
    (one settings.json read per registry definition) is observable.

    Without seeding, the test environment has 0 definitions and the N+1
    loop body never executes, masking the bug.
    """
    saved_defs = dict(registry._definitions)
    saved_paths = dict(registry._paths)

    fake_ids = ["__n1_test_a", "__n1_test_b", "__n1_test_c"]
    try:
        for fid in fake_ids:
            registry._definitions[fid] = ModelDefinition(
                id=fid,
                name=fid,
                family="sdxl",
                components={},
            )
            registry._paths[fid] = f"/tmp/{fid}.yaml"
        yield fake_ids
    finally:
        registry._definitions.clear()
        registry._paths.clear()
        registry._definitions.update(saved_defs)
        registry._paths.update(saved_paths)


def test_list_model_definitions_loads_settings_at_most_once(
    client: TestClient, seeded_definitions: list[str],
) -> None:
    """R-API-07: previously this handler called ModelOverrideManager.get_override
    in a loop over registry definitions, causing N+1 settings.json reads. After
    the fix, the handler must batch-load via get_all_async at most once.

    The assertion is ``<= 1`` rather than ``== 1`` so a future caching
    optimization (e.g., in-memory override cache invalidated on write) can
    drop the count to zero without breaking this regression guard.

    We spy on ``_load`` (the ultimate disk-touching method) rather than
    ``get_override`` / ``get_all`` so the assertion is robust regardless of
    whether the handler routes through sync or async variants — every load
    of settings.json ultimately calls ``_load`` (``_load_async`` delegates
    via ``asyncio.to_thread(_load)``).
    """
    call_count = {"loads": 0}
    # ``_load`` is a staticmethod; access via __dict__ to get the descriptor,
    # then unwrap with __func__ so our spy doesn't recurse into the patched
    # version when it delegates to the real implementation.
    real_load = mom.ModelOverrideManager.__dict__["_load"].__func__

    @staticmethod
    def counting_load():
        call_count["loads"] += 1
        return real_load()

    with patch.object(mom.ModelOverrideManager, "_load", counting_load):
        response = client.get("/api/models/definitions")

    assert response.status_code == 200, (
        f"unexpected status {response.status_code}: {response.text[:200]}"
    )
    # Sanity check: the response actually contains our seeded definitions
    # (so the loop body did execute — otherwise the test would pass spuriously).
    ids_in_response = {d["id"] for d in response.json()}
    assert set(seeded_definitions).issubset(ids_in_response), (
        f"seeded definitions missing from response: "
        f"seeded={seeded_definitions}, got_ids={sorted(ids_in_response)}"
    )
    # After fix: exactly 1 load (the get_all_async at the top of the handler).
    # Before fix: 1 load per registry definition (>= 3 here with seeded data) = N+1.
    assert call_count["loads"] <= 1, (
        f"settings.json was loaded {call_count['loads']} times during "
        f"list_model_definitions; expected <= 1 (N+1 regression bug)"
    )
