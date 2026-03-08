"""
Tests for engine core definitions — ModelComponent, ModelDefinition, ModelFamily.
"""

from app.engine.core.definitions import ModelComponent, ModelDefinition, ModelFamily


class TestModelComponent:
    def test_defaults(self):
        mc = ModelComponent(path="/weights/vae.safetensors")
        assert mc.type == "diffusers"
        assert mc.params == {}

    def test_custom_type(self):
        mc = ModelComponent(path="/w.safetensors", type="checkpoint", params={"custom": True})
        assert mc.type == "checkpoint"
        assert mc.params["custom"] is True


class TestModelDefinition:
    def test_minimal(self):
        md = ModelDefinition(id="test/sdxl", family="sdxl", name="Test SDXL")
        assert md.version == "1.0"
        assert md.defaults == {}
        assert md.components == {}

    def test_full(self):
        md = ModelDefinition(
            id="test/sdxl",
            family="sdxl",
            name="Test SDXL",
            version="2.0",
            defaults={"lr": 0.001},
            components={"unet": ModelComponent(path="/unet")},
        )
        assert md.version == "2.0"
        assert "unet" in md.components
        assert md.defaults["lr"] == 0.001

    def test_introspection_defaults_empty(self):
        md = ModelDefinition(id="x", family="x", name="X")
        assert md.detected_precision == {}
        assert md.architecture_params == {}
        assert md.lora_targetable_modules == []

    def test_model_copy_update(self):
        md = ModelDefinition(id="x", family="x", name="X", version="1.0")
        md2 = md.model_copy(update={"version": "2.0"})
        assert md2.version == "2.0"
        assert md.version == "1.0"  # Original unchanged

    def test_model_dump_roundtrip(self):
        md = ModelDefinition(id="test", family="sdxl", name="Test")
        data = md.model_dump()
        md2 = ModelDefinition(**data)
        assert md2.id == md.id


class TestModelFamily:
    def test_is_abstract(self):
        """ModelFamily cannot be instantiated directly (missing abstract methods)."""
        # ModelFamily is ABC-derived but doesn't declare abstract methods,
        # so it CAN be instantiated conceptually. We just test the default properties.
        class ConcreteFamily(ModelFamily):
            family_id = "test"

        md = ModelDefinition(id="x", family="test", name="X")
        cf = ConcreteFamily(md, {"lr": 0.001})
        assert cf.tokenizer_count == 1
        assert cf.text_encoder_count == 1
        assert cf.definition.id == "x"
        assert cf.config["lr"] == 0.001
