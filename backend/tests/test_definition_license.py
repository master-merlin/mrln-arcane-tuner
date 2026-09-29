"""Optional definition ``license`` field: schema, picker data, LoRA metadata.

Covers:
- ``ModelDefinition.license`` field (defaults to ``None``)
- ``enrich_schema`` injecting the definition_id ``license_map`` — present
  only for definitions that declare a licence, absent otherwise (same
  reasoning as ``trigger_metadata``'s empty-dict return: an absent key is
  unambiguous, an empty string is not)
- ``lora_metadata.license_metadata`` writing ``modelspec.license``
"""

from __future__ import annotations

from app.engine.core.definitions import ModelDefinition
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.registry import registry
from app.engine.models.training_plugin import StandardPlugin
from app.engine.utils.lora_metadata import LICENSE_KEY, license_metadata


class TestLicenseField:
    def test_defaults_none(self):
        d = ModelDefinition(id="x", family="flux1", name="X")
        assert d.license is None

    def test_settable(self):
        d = ModelDefinition(
            id="x",
            family="flux1",
            name="X",
            license="qwen-research (non-commercial)",
        )
        assert d.license == "qwen-research (non-commercial)"


class TestEnrichSchemaLicenseMap:
    def test_license_map_absent_for_unlicensed_definition(self):
        registry.initialize()
        schema = BaseTrainingConfig.model_json_schema()
        enriched = StandardPlugin().enrich_schema(schema)
        license_map = enriched["properties"]["definition_id"].get("license_map")
        assert isinstance(license_map, dict)
        assert "flux1-dev" not in license_map

    def test_license_map_present_for_licensed_definition(self, monkeypatch):
        registry.initialize()
        defn = registry.get_definition("flux1-dev")
        monkeypatch.setattr(defn, "license", "qwen-research (non-commercial)")
        schema = BaseTrainingConfig.model_json_schema()
        enriched = StandardPlugin().enrich_schema(schema)
        license_map = enriched["properties"]["definition_id"].get("license_map")
        assert license_map["flux1-dev"] == "qwen-research (non-commercial)"


class TestLicenseMetadata:
    def test_absent_when_no_license(self):
        assert license_metadata(None) == {}

    def test_absent_when_blank(self):
        assert license_metadata("   ") == {}

    def test_present_with_text(self):
        meta = license_metadata("qwen-research (non-commercial)")
        assert meta == {LICENSE_KEY: "qwen-research (non-commercial)"}

    def test_key_name(self):
        assert LICENSE_KEY == "modelspec.license"
