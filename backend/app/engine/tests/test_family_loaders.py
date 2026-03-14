"""Tests for family-specific loaders — manifest correctness and edge cases."""

from unittest.mock import MagicMock

import torch

from app.engine.core.definitions import ModelDefinition


def _make_definition(**kwargs) -> MagicMock:
    """Build a mock ModelDefinition."""
    definition = MagicMock(spec=ModelDefinition)
    definition.family = kwargs.get("family", "test")
    definition.id = "test-id"
    definition.components = {}
    definition.architecture_params = kwargs.get("architecture_params", {})
    return definition


# ---------------------------------------------------------------------------
# Flux2Loader
# ---------------------------------------------------------------------------

class TestFlux2Loader:
    """Verify Flux2 manifest is correct for Klein (Qwen3) and Dev (Mistral3)."""

    def test_manifest_qwen3(self):
        from app.engine.models.families.flux2.loader import Flux2Loader

        loader = Flux2Loader(torch.device("cpu"))
        definition = _make_definition(
            architecture_params={"te.model_type": "qwen3"},
        )
        manifest = loader.get_component_manifest(definition)

        keys = [s.key for s in manifest]
        assert "tokenizer" in keys
        assert "text_encoder" in keys
        assert "vae" in keys
        assert "unet" in keys

        te_spec = next(s for s in manifest if s.key == "text_encoder")
        assert "AutoModelForCausalLM" in te_spec.hf_class

    def test_manifest_mistral3(self):
        from app.engine.models.families.flux2.loader import Flux2Loader

        loader = Flux2Loader(torch.device("cpu"))
        definition = _make_definition(
            architecture_params={"te.model_type": "mistral3"},
        )
        manifest = loader.get_component_manifest(definition)

        te_spec = next(s for s in manifest if s.key == "text_encoder")
        assert "Mistral3ForConditionalGeneration" in te_spec.hf_class

        tok_spec = next(s for s in manifest if s.key == "tokenizer")
        assert "AutoProcessor" in tok_spec.hf_class

    def test_guidance_zeroing_hook(self):
        """post_load_hook removed — diffusers 0.37 handles guidance zeroing internally."""
        from app.engine.models.families.flux2.loader import Flux2Loader

        loader = Flux2Loader(torch.device("cpu"))
        definition = _make_definition()
        manifest = loader.get_component_manifest(definition)

        unet_spec = next(s for s in manifest if s.key == "unet")
        assert unet_spec.post_load_hook is None


# ---------------------------------------------------------------------------
# Flux1Loader
# ---------------------------------------------------------------------------

class TestFlux1Loader:
    """Verify Flux1 dual-TE manifest."""

    def test_dual_te_manifest(self):
        from app.engine.models.families.flux1.loader import Flux1Loader

        loader = Flux1Loader(torch.device("cpu"))
        definition = _make_definition()
        manifest = loader.get_component_manifest(definition)

        keys = [s.key for s in manifest]
        assert "text_encoder" in keys, "CLIP TE missing"
        assert "text_encoder_2" in keys, "T5 TE missing"
        assert "tokenizer" in keys
        assert "tokenizer_2" in keys

        clip_spec = next(s for s in manifest if s.key == "text_encoder")
        assert "CLIPTextModel" in clip_spec.hf_class

        t5_spec = next(s for s in manifest if s.key == "text_encoder_2")
        assert "T5EncoderModel" in t5_spec.hf_class


# ---------------------------------------------------------------------------
# SDXLLoader
# ---------------------------------------------------------------------------

class TestSDXLLoader:
    """Verify SDXL manifest uses root_key and separate_repo."""

    def test_manifest_structure(self):
        from app.engine.models.families.sdxl.loader import SDXLLoader

        loader = SDXLLoader(torch.device("cpu"))
        definition = _make_definition()
        manifest = loader.get_component_manifest(definition)

        keys = [s.key for s in manifest]
        assert "tokenizer_1" in keys
        assert "tokenizer_2" in keys
        assert "text_encoder_1" in keys
        assert "text_encoder_2" in keys
        assert "vae" in keys
        assert "unet" in keys

    def test_root_key_set(self):
        """First spec declares root_key='unet'."""
        from app.engine.models.families.sdxl.loader import SDXLLoader

        loader = SDXLLoader(torch.device("cpu"))
        definition = _make_definition()
        manifest = loader.get_component_manifest(definition)

        first_spec = manifest[0]
        assert first_spec.root_key == "unet"

    def test_vae_separate_repo(self):
        """VAE spec has separate_repo=True."""
        from app.engine.models.families.sdxl.loader import SDXLLoader

        loader = SDXLLoader(torch.device("cpu"))
        definition = _make_definition()
        manifest = loader.get_component_manifest(definition)

        vae_spec = next(s for s in manifest if s.key == "vae")
        assert vae_spec.separate_repo is True
        assert vae_spec.definition_key == "vae"
        assert vae_spec.dtype_override == torch.float32

    def test_use_subfolder_kwarg(self):
        """All SDXL specs use subfolder kwarg (single repo with subfolders)."""
        from app.engine.models.families.sdxl.loader import SDXLLoader

        loader = SDXLLoader(torch.device("cpu"))
        definition = _make_definition()
        manifest = loader.get_component_manifest(definition)

        for spec in manifest:
            assert spec.use_subfolder_kwarg is True, (
                f"{spec.key} should use subfolder kwarg"
            )


# ---------------------------------------------------------------------------
# QwenImageLoader
# ---------------------------------------------------------------------------

class TestQwenImageLoader:
    """Verify QwenImage manifest and VAE fallback."""

    def test_manifest_structure(self):
        from app.engine.models.families.qwen_image.loader import QwenImageLoader

        loader = QwenImageLoader(torch.device("cpu"))
        definition = _make_definition()
        manifest = loader.get_component_manifest(definition)

        keys = [s.key for s in manifest]
        assert "tokenizer" in keys
        assert "text_encoder" in keys
        assert "vae" in keys
        assert "unet" in keys

        te_spec = next(s for s in manifest if s.key == "text_encoder")
        assert "Qwen2_5_VL" in te_spec.hf_class


# ---------------------------------------------------------------------------
# ZImageLoader
# ---------------------------------------------------------------------------

class TestZImageLoader:
    """Verify ZImage manifest."""

    def test_manifest_structure(self):
        from app.engine.models.families.zimage.loader import ZImageLoader

        loader = ZImageLoader(torch.device("cpu"))
        definition = _make_definition()
        manifest = loader.get_component_manifest(definition)

        keys = [s.key for s in manifest]
        assert "tokenizer" in keys
        assert "text_encoder" in keys
        assert "vae" in keys
        assert "unet" in keys

        te_spec = next(s for s in manifest if s.key == "text_encoder")
        assert "AutoModelForCausalLM" in te_spec.hf_class
