"""
Tests for ModelRegistry — covers registration, YAML loading, definition CRUD, save/update.
"""

import yaml
import pytest

from app.engine.models.registry import ModelRegistry
from app.engine.core.definitions import ModelDefinition, ModelFamily


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset registry state before each test."""
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False
    yield
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False


class TestRegisterFamily:
    def test_register_and_retrieve(self):
        class DummyFamily(ModelFamily):
            family_id = "dummy"
        ModelRegistry.register_family("dummy", DummyFamily)
        assert ModelRegistry.get_family_class("dummy") is DummyFamily

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="not found"):
            ModelRegistry.get_family_class("nonexistent")


class TestLoadDefinition:
    def test_load_from_yaml(self, tmp_path):
        data = {"id": "test/sdxl", "family": "sdxl", "name": "Test SDXL"}
        path = str(tmp_path / "test.yaml")
        with open(path, "w") as f:
            yaml.dump(data, f)

        defn = ModelRegistry.load_definition(path)
        assert defn.id == "test/sdxl"
        assert defn.family == "sdxl"

    def test_shorthand_component_normalization(self, tmp_path):
        data = {
            "id": "test/norm",
            "family": "sdxl",
            "name": "Norm",
            "components": {"unet": "/path/to/unet"},
        }
        path = str(tmp_path / "norm.yaml")
        with open(path, "w") as f:
            yaml.dump(data, f)

        defn = ModelRegistry.load_definition(path)
        assert defn.components["unet"].path == "/path/to/unet"

    def test_loaded_definition_in_registry(self, tmp_path):
        data = {"id": "inreg", "family": "sdxl", "name": "InReg"}
        path = str(tmp_path / "inreg.yaml")
        with open(path, "w") as f:
            yaml.dump(data, f)

        ModelRegistry.load_definition(path)
        assert ModelRegistry.get_definition("inreg") is not None


class TestDefinitionCRUD:
    def _add_definition(self, tmp_path, def_id="crud/test"):
        data = {"id": def_id, "family": "sdxl", "name": "Crud Test", "version": "1.0"}
        path = str(tmp_path / f"{def_id.replace('/', '_')}.yaml")
        with open(path, "w") as f:
            yaml.dump(data, f)
        return ModelRegistry.load_definition(path)

    def test_get_definition_returns_none_for_missing(self):
        assert ModelRegistry.get_definition("ghost") is None

    def test_list_models(self, tmp_path):
        self._add_definition(tmp_path, "a")
        self._add_definition(tmp_path, "b")
        ids = ModelRegistry.list_models()
        assert set(ids) == {"a", "b"}

    def test_count(self, tmp_path):
        """Public accessor for len(_definitions) — B-ARCH-6, replaces the
        system_routes.py private reach-through `len(registry._definitions)`."""
        assert ModelRegistry.count() == 0
        self._add_definition(tmp_path, "a")
        self._add_definition(tmp_path, "b")
        assert ModelRegistry.count() == 2

    def test_update_definition(self, tmp_path):
        self._add_definition(tmp_path)
        ModelRegistry.update_definition("crud/test", {"version": "2.0"})
        assert ModelRegistry.get_definition("crud/test").version == "2.0"

    def test_update_nonexistent_raises(self):
        with pytest.raises(ValueError, match="not found"):
            ModelRegistry.update_definition("ghost", {"version": "2.0"})

    def test_save_definition(self, tmp_path):
        self._add_definition(tmp_path)
        ModelRegistry.update_definition("crud/test", {"version": "3.0"})
        ModelRegistry.save_definition("crud/test")

        # Re-read the YAML to verify persistence
        path = ModelRegistry._paths["crud/test"]
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        assert data["version"] == "3.0"

    def test_save_definition_preserves_comments_and_layout(self, tmp_path):
        """Enrichment saves must round-trip the hand-written YAML: comments
        and key order survive, only changed keys are rewritten, and fields
        the author never wrote don't explode into the file (the old
        yaml.dump(model_dump()) full rewrite destroyed all three)."""
        text = (
            "id: crud/commented\n"
            "family: sdxl\n"
            "# Chosen revision: the diffusers branch - do not change.\n"
            "name: Commented Model\n"
            "version: '1.0'\n"
            "architecture_params:\n"
            "  # attention_head_dim actually means NUM HEADS here.\n"
            "  depth: 2\n"
            "  # GQA: to_k/to_v width = num_kv_heads * head_dim.\n"
            "  num_kv_heads: 1\n"
        )
        path = str(tmp_path / "commented.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        ModelRegistry.load_definition(path)

        # Enrichment-style update: merge one harvested key into arch params.
        ModelRegistry.update_definition(
            "crud/commented",
            {"architecture_params": {"depth": 2, "num_kv_heads": 1, "num_heads": 8}},
        )
        ModelRegistry.save_definition("crud/commented")

        with open(path, encoding="utf-8") as f:
            saved = f.read()
        assert "# Chosen revision: the diffusers branch - do not change." in saved
        # Comments on UNCHANGED siblings inside the merged mapping survive too.
        assert "# attention_head_dim actually means NUM HEADS here." in saved
        assert "# GQA: to_k/to_v width = num_kv_heads * head_dim." in saved
        data = yaml.safe_load(saved)
        assert data["architecture_params"] == {"depth": 2, "num_kv_heads": 1, "num_heads": 8}
        # Fields the author never wrote (still at their defaults) stay absent.
        assert "lora_targetable_modules" not in saved
        assert "block_topology" not in saved
        # Hand-written key order intact.
        assert saved.index("family:") < saved.index("name:") < saved.index("version:")

    def test_save_definition_appends_newly_enriched_fields(self, tmp_path):
        """Keys absent from the file but set to non-default values (fresh
        enrichment data) are appended without disturbing the rest."""
        text = (
            "id: crud/fresh\n"
            "family: sdxl\n"
            "# keep me\n"
            "name: Fresh\n"
        )
        path = str(tmp_path / "fresh.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        ModelRegistry.load_definition(path)

        ModelRegistry.update_definition(
            "crud/fresh",
            {"detected_precision": {"unet": "torch.bfloat16"}},
        )
        ModelRegistry.save_definition("crud/fresh")

        with open(path, encoding="utf-8") as f:
            saved = f.read()
        assert "# keep me" in saved
        assert yaml.safe_load(saved)["detected_precision"] == {"unet": "torch.bfloat16"}

    def test_save_definition_scientific_floats_survive_pyyaml_reload(self, tmp_path):
        """ruamel's default `1e-05` float form is re-read by PyYAML as a
        STRING (YAML 1.1 wants a mantissa dot). Enrichment writes such floats
        (norm_eps) — the save must emit a form PyYAML loads back as float."""
        text = "id: crud/floats\nfamily: sdxl\nname: Floats\n"
        path = str(tmp_path / "floats.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        ModelRegistry.load_definition(path)

        ModelRegistry.update_definition(
            "crud/floats",
            {"architecture_params": {"transformer.norm_eps": 1e-05}},
        )
        ModelRegistry.save_definition("crud/floats")

        with open(path, encoding="utf-8") as f:
            reloaded = yaml.safe_load(f)
        val = reloaded["architecture_params"]["transformer.norm_eps"]
        assert isinstance(val, float), f"norm_eps came back as {type(val).__name__}: {val!r}"
        assert val == 1e-05

    def test_save_nonexistent_raises(self):
        with pytest.raises(ValueError, match="not found"):
            ModelRegistry.save_definition("ghost")

    def test_save_without_path_raises(self, tmp_path):
        data = {"id": "nopath", "family": "sdxl", "name": "No Path"}
        defn = ModelDefinition(**data)
        ModelRegistry._definitions["nopath"] = defn
        # No path entry
        with pytest.raises(ValueError, match="No file path"):
            ModelRegistry.save_definition("nopath")
