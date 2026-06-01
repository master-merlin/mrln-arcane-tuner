"""T4: GET /api/models/capabilities/{definition_id} includes the archetype
capability descriptor from resolve_capabilities(), merged alongside the
existing backward-compatible keys (enriched, block_topology,
lora_targetable_modules, trainable_layers).
"""

from __future__ import annotations

import pytest

from app.engine.models.registry import registry


@pytest.fixture(scope="module", autouse=True)
def _loaded_registry():
    """Ensure definitions are loaded before endpoint tests run."""
    registry.discover_families()
    registry.load_definitions("app/engine/models/definitions")
    return registry


def test_capabilities_endpoint_includes_descriptor(client):
    r = client.get("/api/models/capabilities/hidream_o1_image")
    assert r.status_code == 200
    body = r.json()
    # existing keys preserved (backward-compat):
    assert "block_topology" in body
    assert "lora_targetable_modules" in body
    # new descriptor merged in:
    assert body["archetype"] == "unified_transformer"
    assert body["capabilities"]["has_vae"] is False
    assert body["field_visibility"]["cache_latents"]["supported"] is False
    assert body["field_visibility"]["cache_text_embeddings"]["supported"] is False
    assert body["defaults"]["learning_rate"] == 5e-6


def test_capabilities_endpoint_latent_family(client):
    r = client.get("/api/models/capabilities/sdxl_base_1.0")
    assert r.status_code == 200
    body = r.json()
    assert body["archetype"] == "latent_diffusion"
    assert body["field_visibility"]["cache_latents"]["supported"] is True
    assert (
        body["field_visibility"]["train_text_encoder"]["supported"] is True
    )  # SDXL override
    assert body["defaults"]["learning_rate"] == 1e-4


def test_capabilities_endpoint_unknown_definition(client):
    r = client.get("/api/models/capabilities/does_not_exist_xyz")
    assert r.status_code == 404
