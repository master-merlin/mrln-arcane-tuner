"""PR4 — training capability plumbing for paired edit models.

Covers:
- ``ModelDefinition.control_inputs`` field
- ``resolve_capabilities`` surfacing control_inputs + is_edit + the
  edit-gated field_visibility rules (hide flips/masking, show control_resolution)
- ``BaseTrainingConfig.control_resolution`` + ``SamplePromptConfig.control_images``
- ``enrich_schema`` injecting the definition_id ``edit_map``
- ``validate_edit_config`` static run-config rules (kind match, flips, masking)
"""

from __future__ import annotations

import pytest

from app.engine.core.definitions import ModelDefinition
from app.engine.core.archetypes import resolve_capabilities, build_field_visibility
from app.engine.core.edit_validation import validate_edit_config, EditConfigReport
from app.engine.models.base import BaseTrainingConfig, SamplePromptConfig
from app.engine.models.training_plugin import StandardPlugin
from app.engine.models.registry import registry


# ── ModelDefinition.control_inputs ───────────────────────────────────────


class TestControlInputsField:
    def test_defaults_zero(self):
        d = ModelDefinition(id="x", family="flux1", name="X")
        assert d.control_inputs == 0

    def test_settable(self):
        d = ModelDefinition(id="x", family="flux1", name="X", control_inputs=1)
        assert d.control_inputs == 1

    def test_shipped_definitions_control_inputs(self):
        # Standard T2I definitions are 0; edit definitions (Kontext, PR6) are >0.
        # EDIT_FIRST_IDS: unified generate+edit models whose PRIMARY
        # definition is edit-capable without "edit" in the id (OmniGen2's
        # whole point is instruction edit; T2I is its no-control fallback).
        edit_first_ids = {
            "omnigen2",
            "bernini-r-1.3b",
            "bernini-r-14b",
        }  # bernini = v2v edit-first renderer; control = source video
        registry.initialize()
        for mid in registry.list_models():
            ci = registry.get_definition(mid).control_inputs
            if "kontext" in mid or "edit" in mid or mid in edit_first_ids:
                assert ci >= 1, f"{mid} is an edit definition but control_inputs={ci}"
            else:
                assert ci == 0, f"{mid} is standard but control_inputs={ci}"


# ── Capability descriptor ────────────────────────────────────────────────


class TestEditCapabilities:
    def test_standard_definition_not_edit(self):
        registry.initialize()
        r = resolve_capabilities(registry.get_definition("flux1-dev"))
        caps = r["capabilities"]
        assert caps["control_inputs"] == 0
        assert caps["is_edit"] is False
        # Flips/masking shown, control_resolution hidden for non-edit.
        fv = r["field_visibility"]
        assert fv["h_flip"]["supported"] is True
        assert fv["v_flip"]["supported"] is True
        assert fv["masking_enabled"]["supported"] is True
        assert fv["control_resolution"]["supported"] is False

    def test_edit_definition_gates_fields(self):
        # Synthesize an edit definition (no edit family ships until PR6).
        defn = ModelDefinition(
            id="flux1-dev", family="flux1", name="Edit", control_inputs=1
        )
        r = resolve_capabilities(defn)
        caps = r["capabilities"]
        assert caps["control_inputs"] == 1
        assert caps["is_edit"] is True
        fv = r["field_visibility"]
        assert fv["h_flip"]["supported"] is False
        assert fv["v_flip"]["supported"] is False
        assert fv["masking_enabled"]["supported"] is False
        assert fv["control_resolution"]["supported"] is True

    def test_field_visibility_helper_with_derived_flags(self):
        fv = build_field_visibility(
            {
                "is_edit": True,
                "supports_augmentation": False,
                "supports_masking_variants": False,
            }
        )
        assert fv["h_flip"]["supported"] is False
        assert "reason" in fv["h_flip"]
        assert fv["control_resolution"]["supported"] is True


# ── Config schema fields ─────────────────────────────────────────────────


class TestConfigSchemaFields:
    def test_control_resolution_default(self):
        cfg = BaseTrainingConfig(datasets=[{"dataset_name": "d"}])
        assert cfg.control_resolution == 0

    def test_control_resolution_group_strategy(self):
        schema = BaseTrainingConfig.model_json_schema()
        extra = schema["properties"]["control_resolution"].get("group")
        assert extra == "STRATEGY"

    def test_sample_prompt_control_images_default(self):
        p = SamplePromptConfig()
        assert p.control_images == []

    def test_sample_prompt_control_images_settable(self):
        p = SamplePromptConfig(control_images=["control/a.jpg"])
        assert p.control_images == ["control/a.jpg"]


# ── enrich_schema edit_map ───────────────────────────────────────────────


class TestEnrichSchemaEditMap:
    def test_edit_map_present_on_definition_id(self):
        registry.initialize()
        schema = BaseTrainingConfig.model_json_schema()
        enriched = StandardPlugin().enrich_schema(schema)
        edit_map = enriched["properties"]["definition_id"].get("edit_map")
        assert isinstance(edit_map, dict)
        # Standard definitions map to 0; the Kontext edit definition to 1.
        assert edit_map["flux1-dev"] == 0
        assert edit_map["flux1-kontext-dev"] == 1


# ── validate_edit_config ─────────────────────────────────────────────────


def _defn(control_inputs: int) -> ModelDefinition:
    return ModelDefinition(
        id="x", family="flux1", name="X", control_inputs=control_inputs
    )


class TestValidateEditConfig:
    def test_edit_model_requires_edit_datasets(self):
        report = validate_edit_config(
            _defn(1),
            {"datasets": [{"dataset_name": "std"}]},
            kind_of=lambda n: "standard",
        )
        assert not report.ok
        assert any("requires an edit dataset" in e for e in report.errors)

    def test_edit_model_with_edit_dataset_ok(self):
        report = validate_edit_config(
            _defn(1),
            {"datasets": [{"dataset_name": "ed"}], "h_flip": False, "v_flip": False},
            kind_of=lambda n: "edit",
        )
        assert report.ok
        assert report.warnings == []

    def test_edit_model_rejects_flips(self):
        report = validate_edit_config(
            _defn(1),
            {"datasets": [{"dataset_name": "ed"}], "h_flip": True},
            kind_of=lambda n: "edit",
        )
        assert not report.ok
        assert any("flip" in e.lower() for e in report.errors)

    def test_edit_model_rejects_masking(self):
        report = validate_edit_config(
            _defn(1),
            {"datasets": [{"dataset_name": "ed", "masking_enabled": True}]},
            kind_of=lambda n: "edit",
        )
        assert not report.ok
        assert any("mask" in e.lower() for e in report.errors)

    def test_standard_model_warns_on_edit_dataset(self):
        report = validate_edit_config(
            _defn(0),
            {"datasets": [{"dataset_name": "ed"}]},
            kind_of=lambda n: "edit",
        )
        assert report.ok  # warnings don't block
        assert any("ignored" in w for w in report.warnings)

    def test_standard_model_standard_dataset_clean(self):
        report = validate_edit_config(
            _defn(0),
            {"datasets": [{"dataset_name": "std"}]},
            kind_of=lambda n: "standard",
        )
        assert report.ok
        assert report.warnings == []

    def test_report_is_dataclass(self):
        assert isinstance(EditConfigReport(), EditConfigReport)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
