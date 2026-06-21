"""Task 7: list_model_definitions must include caption_format for each definition.

Assertions:
- ideogram4 family definition → caption_format == "ideogram4_json"
- flat (sdxl) family definition → caption_format == "plain"
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.engine.core.definitions import ModelDefinition
from app.engine.models.registry import registry


@pytest.fixture
def seeded_caption_format_definitions():
    """Seed the registry with one ideogram4 definition and one sdxl definition."""
    saved_defs = dict(registry._definitions)
    saved_paths = dict(registry._paths)

    try:
        registry._definitions["ideogram4_fp8"] = ModelDefinition(
            id="ideogram4_fp8",
            name="Ideogram 4 FP8",
            family="ideogram4",
            components={},
        )
        registry._paths["ideogram4_fp8"] = "/tmp/ideogram4_fp8.yaml"

        registry._definitions["__cf_test_sdxl"] = ModelDefinition(
            id="__cf_test_sdxl",
            name="Test SDXL",
            family="sdxl",
            components={},
        )
        registry._paths["__cf_test_sdxl"] = "/tmp/__cf_test_sdxl.yaml"

        yield
    finally:
        registry._definitions.clear()
        registry._paths.clear()
        registry._definitions.update(saved_defs)
        registry._paths.update(saved_paths)


def test_list_definitions_includes_caption_format_ideogram4(
    client: TestClient, seeded_caption_format_definitions
) -> None:
    """ideogram4 definition must carry caption_format == 'ideogram4_json'."""
    response = client.get("/api/models/definitions")
    assert response.status_code == 200
    defs = {d["id"]: d for d in response.json()}

    assert "ideogram4_fp8" in defs, (
        f"ideogram4_fp8 missing from response; got {list(defs)}"
    )
    defn = defs["ideogram4_fp8"]
    assert "caption_format" in defn, (
        "caption_format field missing from definition payload"
    )
    assert defn["caption_format"] == "ideogram4_json", (
        f"expected 'ideogram4_json', got {defn['caption_format']!r}"
    )


def test_list_definitions_includes_caption_format_plain(
    client: TestClient, seeded_caption_format_definitions
) -> None:
    """Non-structured (sdxl) definition must carry caption_format == 'plain'."""
    response = client.get("/api/models/definitions")
    assert response.status_code == 200
    defs = {d["id"]: d for d in response.json()}

    assert "__cf_test_sdxl" in defs, f"sdxl test def missing; got {list(defs)}"
    defn = defs["__cf_test_sdxl"]
    assert "caption_format" in defn, (
        "caption_format field missing from definition payload"
    )
    assert defn["caption_format"] == "plain", (
        f"expected 'plain', got {defn['caption_format']!r}"
    )
