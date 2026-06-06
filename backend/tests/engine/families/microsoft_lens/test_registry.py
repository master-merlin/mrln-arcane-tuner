"""microsoft_lens family registration tests."""
import pytest

from app.engine.models.registry import ModelRegistry


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset registry class-level state before and after each test."""
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


def test_microsoft_lens_family_is_discovered():
    registry = ModelRegistry()
    registry.discover_families()
    fam_cls = registry.get_family_class("microsoft_lens")
    assert fam_cls.family_name == "microsoft_lens"


def test_microsoft_lens_trainer_class_resolves():
    from app.engine.models.families.microsoft_lens.family import (
        MicrosoftLensFamily,
    )
    from app.engine.models.families.microsoft_lens.trainer import (
        MicrosoftLensTrainer,
    )
    assert MicrosoftLensFamily.family_name == "microsoft_lens"
    instance = object.__new__(MicrosoftLensFamily)
    assert instance.get_trainer_class() is MicrosoftLensTrainer


def test_vendored_transformer_imports_and_constructs_tiny():
    from app.engine.models.families.microsoft_lens.vendor.transformer import (
        LensTransformer2DModel,
    )
    # Tiny config -- proves the class constructs without real weights.
    model = LensTransformer2DModel(
        patch_size=2, in_channels=128, out_channels=32, num_layers=1,
        attention_head_dim=8, num_attention_heads=2, inner_dim=16,
        enc_hidden_dim=2880, axes_dims_rope=(2, 2, 4),
        gate_mlp=True, rms_norm=True, multi_layer_encoder_feature=True,
        selected_layer_index=(5, 11, 17, 23),
    )
    # PeftAdapterMixin present -> LoRA-attachable.
    assert any("Peft" in b.__name__ for b in type(model).__mro__)
    assert model.config.num_attention_heads == 2


def test_lens_base_definition_loads():
    registry = ModelRegistry()
    registry.initialize()
    defn = registry.get_definition("microsoft-lens-base")
    assert defn.family == "microsoft_lens"
    assert "Lens-Base" in defn.components["repo"].path
    assert defn.architecture_params["transformer.num_layers"] == 48
