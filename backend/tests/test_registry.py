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
