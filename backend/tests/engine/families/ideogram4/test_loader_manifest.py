"""ideogram4 loader manifest tests (no weights downloaded)."""
import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.ideogram4.loader import IdeogramV4Loader


def _defn():
    return ModelDefinition(
        id="ideogram4-fp8", family="ideogram4", name="Ideogram 4",
        defaults={}, components={},
    )


def test_manifest_declares_stock_components():
    loader = IdeogramV4Loader(torch.device("cpu"))
    specs = {s.key: s for s in loader.get_component_manifest(_defn())}

    # Text encoder: Qwen3-VL in a SEPARATE repo, declared in the YAML under the
    # "text_encoder" definition key.
    assert specs["text_encoder"].hf_class == "transformers.AutoModel"
    assert specs["text_encoder"].separate_repo is True
    assert specs["text_encoder"].definition_key == "text_encoder"

    # Tokenizer: shares the Qwen3-VL repo (separate repo, same definition key).
    assert specs["tokenizer"].hf_class == "transformers.AutoTokenizer"
    assert specs["tokenizer"].is_torch_model is False
    assert specs["tokenizer"].separate_repo is True
    assert specs["tokenizer"].definition_key == "text_encoder"

    # VAE: upstream uses a CUSTOM `AutoEncoder` (ideogram4/autoencoder.py), NOT a
    # diffusers-native class, so the generic from_pretrained manifest path can't
    # build it. It is now vendored (vendor/autoencoder_ideogram4.py) and loaded
    # BY HAND in the overridden load(), so it is NOT in the manifest.
    assert "vae" not in specs

    # The DiT is NOT in the manifest -- loaded by the overridden load().
    assert "unet" not in specs
