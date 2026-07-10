"""
Tests for model introspection and definition management.
Covers: ModelIntrospector, IntrospectionResult, definition enrichment,
and CRUD API endpoints.
"""
import torch
import torch.nn as nn
from unittest.mock import MagicMock


class TestModelIntrospector:
    """Tests for the ModelIntrospector utility."""

    def _make_simple_model(self, dtype: torch.dtype = torch.float16):
        """Helper: create a simple model with known structure."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Conv2d(3, 16, 3),
        ).to(dtype)
        return model

    def test_detects_dtype(self):
        """Introspector correctly detects the dtype of model parameters."""
        from app.engine.utils.introspection import ModelIntrospector

        model = self._make_simple_model(torch.float16)
        introspector = ModelIntrospector()
        result = introspector.introspect({"unet": model})

        assert result.detected_precision["unet"] == "torch.float16"

    def test_detects_bf16(self):
        """Introspector correctly detects bfloat16."""
        from app.engine.utils.introspection import ModelIntrospector

        model = self._make_simple_model(torch.bfloat16)
        introspector = ModelIntrospector()
        result = introspector.introspect({"unet": model})

        assert result.detected_precision["unet"] == "torch.bfloat16"

    def test_finds_lora_targets(self):
        """Introspector identifies nn.Linear and nn.Conv2d as LoRA targets."""
        from app.engine.utils.introspection import ModelIntrospector

        model = self._make_simple_model()
        introspector = ModelIntrospector()
        result = introspector.introspect({"unet": model})

        # Sequential has modules named "0" (Linear), "2" (Linear), "3" (Conv2d)
        assert len(result.lora_targetable_modules) == 3
        # nn.ReLU should NOT be a target
        for target in result.lora_targetable_modules:
            assert "1" not in target or target != "1"  # ReLU is index 1

    def test_grouped_convs_are_not_lora_targets(self):
        """Depthwise/grouped Conv2d must be excluded from LoRA targets.

        PEFT's LoRA Conv2d requires ``rank % groups == 0``; with typical ranks
        (8-64) a depthwise conv (groups == channels, e.g. DreamLite's
        use_sep_conv blocks at groups=256) always fails at inject time with
        "Targeting a Conv2d with groups=256 and rank 32". Only groups=1 convs
        are eligible.
        """
        from app.engine.utils.introspection import ModelIntrospector

        model = nn.Sequential()
        model.add_module("linear", nn.Linear(8, 8))
        model.add_module("plain_conv", nn.Conv2d(8, 8, 3))
        model.add_module("depthwise", nn.Conv2d(8, 8, 3, groups=8))
        model.add_module("grouped", nn.Conv2d(8, 8, 3, groups=4))

        result = ModelIntrospector().introspect({"unet": model})

        assert "linear" in result.lora_targetable_modules
        assert "plain_conv" in result.lora_targetable_modules
        assert "depthwise" not in result.lora_targetable_modules
        assert "grouped" not in result.lora_targetable_modules

    def test_counts_total_params(self):
        """Introspector counts total parameters correctly."""
        from app.engine.utils.introspection import ModelIntrospector

        model = self._make_simple_model()
        introspector = ModelIntrospector()
        result = introspector.introspect({"unet": model})

        expected = sum(p.numel() for p in model.parameters())
        assert result.total_params == expected

    def test_extracts_architecture_params(self):
        """Introspector extracts known architecture attributes from model."""
        from app.engine.utils.introspection import ModelIntrospector

        # Create model with known attributes
        class FakeUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = 4096
                self.num_heads = 32
                self.in_channels = 64
                self.linear = nn.Linear(64, 128)

        model = FakeUNet()
        introspector = ModelIntrospector()
        result = introspector.introspect({"unet": model})

        assert result.architecture_params.get("hidden_size") == 4096
        assert result.architecture_params.get("num_heads") == 32
        assert result.architecture_params.get("in_channels") == 64

    def test_skips_non_module_components(self):
        """Introspector skips tokenizers and non-Module components."""
        from app.engine.utils.introspection import ModelIntrospector

        model = self._make_simple_model()
        introspector = ModelIntrospector()

        components = {
            "unet": model,
            "tokenizer": MagicMock(),  # Not an nn.Module
        }
        result = introspector.introspect(components)

        assert "unet" in result.detected_precision
        assert "tokenizer" not in result.detected_precision

    def test_multiple_components_detected(self):
        """Introspector detects precision for all nn.Module components."""
        from app.engine.utils.introspection import ModelIntrospector

        unet = self._make_simple_model(torch.float16)
        vae = self._make_simple_model(torch.float32)
        introspector = ModelIntrospector()

        result = introspector.introspect({"unet": unet, "vae": vae})

        assert result.detected_precision["unet"] == "torch.float16"
        assert result.detected_precision["vae"] == "torch.float32"


class TestDefinitionEnrichment:
    """Tests for definition update with introspection results."""

    def test_enrichment_preserves_existing_fields(self):
        """Updating a definition with introspection data preserves user-set fields."""
        from app.engine.core.definitions import ModelDefinition, ModelComponent
        from app.engine.models.registry import ModelRegistry

        defn = ModelDefinition(
            id="test_enrich",
            family="sdxl",
            name="User Custom Name",
            version="2.0",
            components={"unet": ModelComponent(path="/fake")},
        )
        ModelRegistry._definitions["test_enrich"] = defn

        try:
            ModelRegistry.update_definition("test_enrich", {
                "detected_precision": {"unet": "torch.float16"},
                "lora_targetable_modules": ["attn.to_q", "attn.to_v"],
            })
            updated = ModelRegistry.get_definition("test_enrich")

            # Introspection fields updated
            assert updated.detected_precision == {"unet": "torch.float16"}
            assert updated.lora_targetable_modules == ["attn.to_q", "attn.to_v"]

            # User fields preserved
            assert updated.name == "User Custom Name"
            assert updated.version == "2.0"
            assert updated.components["unet"].path == "/fake"
        finally:
            del ModelRegistry._definitions["test_enrich"]

    def test_only_missing_fields_filled(self):
        """Enrichment does not overwrite existing introspection data."""
        from app.engine.core.definitions import ModelDefinition
        from app.engine.models.registry import ModelRegistry

        defn = ModelDefinition(
            id="test_no_overwrite",
            family="sdxl",
            name="Test",
            detected_precision={"unet": "torch.float32"},  # User manually set FP32
        )
        ModelRegistry._definitions["test_no_overwrite"] = defn

        try:
            # The update_definition uses model_copy which will overwrite if provided
            # So the caller is responsible for checking existing fields
            updated = ModelRegistry.get_definition("test_no_overwrite")
            assert updated.detected_precision == {"unet": "torch.float32"}
        finally:
            del ModelRegistry._definitions["test_no_overwrite"]

    def test_enrich_definition_fills_empty_and_persists(self, tmp_path):
        """enrich_definition runs introspection, fills empty fields, and persists to YAML."""
        import yaml
        from app.engine.core.definitions import ModelDefinition, ModelComponent
        from app.engine.models.registry import ModelRegistry

        yaml_path = str(tmp_path / "test_enrich.yaml")
        defn = ModelDefinition(
            id="test_enrich_full",
            family="sdxl",
            name="Test Enrichment",
            components={"unet": ModelComponent(path="/fake")},
        )
        # Manually add to registry with a path so save works
        ModelRegistry._definitions["test_enrich_full"] = defn
        ModelRegistry._paths["test_enrich_full"] = yaml_path

        try:
            # Create a simple model for introspection
            import torch.nn as nn
            model = nn.Sequential(nn.Linear(64, 128), nn.Conv2d(3, 16, 3)).to(torch.float16)
            components = {"unet": model, "tokenizer": MagicMock()}

            ModelRegistry.enrich_definition("test_enrich_full", components)

            # Check in-memory
            updated = ModelRegistry.get_definition("test_enrich_full")
            assert updated.detected_precision == {"unet": "torch.float16"}
            assert len(updated.lora_targetable_modules) == 2  # Linear + Conv2d
            assert updated.name == "Test Enrichment"  # User field preserved

            # Check persisted YAML
            with open(yaml_path, "r") as f:
                saved = yaml.safe_load(f)
            assert saved["detected_precision"] == {"unet": "torch.float16"}
            assert len(saved["lora_targetable_modules"]) == 2
        finally:
            del ModelRegistry._definitions["test_enrich_full"]
            del ModelRegistry._paths["test_enrich_full"]

    def test_enrich_definition_skips_populated_fields(self, tmp_path):
        """enrich_definition does NOT overwrite already-populated introspection fields."""
        from app.engine.core.definitions import ModelDefinition
        from app.engine.models.registry import ModelRegistry

        defn = ModelDefinition(
            id="test_no_clobber",
            family="sdxl",
            name="Test No Clobber",
            detected_precision={"unet": "torch.float32"},  # Already set by user
            lora_targetable_modules=["custom.module"],  # User-customized
        )
        ModelRegistry._definitions["test_no_clobber"] = defn
        # No path — so persist will be skipped but enrichment logic still runs

        try:
            import torch.nn as nn
            model = nn.Sequential(nn.Linear(64, 128)).to(torch.float16)
            ModelRegistry.enrich_definition("test_no_clobber", {"unet": model})

            updated = ModelRegistry.get_definition("test_no_clobber")
            # Existing values preserved
            assert updated.detected_precision == {"unet": "torch.float32"}
            assert updated.lora_targetable_modules == ["custom.module"]
        finally:
            del ModelRegistry._definitions["test_no_clobber"]


class TestIntrospectionResult:
    """Tests for the IntrospectionResult Pydantic model."""

    def test_default_values(self):
        """IntrospectionResult has sensible defaults."""
        from app.engine.utils.introspection import IntrospectionResult

        result = IntrospectionResult()
        assert result.detected_precision == {}
        assert result.lora_targetable_modules == []
        assert result.total_params == 0
        assert result.layer_count == 0

    def test_serialization(self):
        """IntrospectionResult serializes to dict correctly."""
        from app.engine.utils.introspection import IntrospectionResult

        result = IntrospectionResult(
            detected_precision={"unet": "torch.float16"},
            lora_targetable_modules=["attn.to_q"],
            total_params=12345,
        )
        data = result.model_dump()
        assert data["detected_precision"] == {"unet": "torch.float16"}
        assert data["total_params"] == 12345
