"""microsoft_lens loader manifest tests (no weights downloaded)."""
import torch

from app.engine.models.families.microsoft_lens.loader import MicrosoftLensLoader
from app.engine.core.definitions import ModelDefinition


def _defn():
    return ModelDefinition(
        id="microsoft-lens-base", family="microsoft_lens", name="Lens Base",
        defaults={}, components={},
    )


def test_manifest_declares_stock_components():
    loader = MicrosoftLensLoader(torch.device("cpu"))
    specs = {s.key: s for s in loader.get_component_manifest(_defn())}
    assert specs["tokenizer"].hf_class == "transformers.PreTrainedTokenizerFast"
    assert specs["tokenizer"].is_torch_model is False
    assert specs["text_encoder"].hf_class == "transformers.GptOssForCausalLM"
    assert specs["vae"].hf_class == "diffusers.AutoencoderKLFlux2"
    # The DiT is NOT in the manifest -- it is loaded by the overridden load().
    assert "unet" not in specs
