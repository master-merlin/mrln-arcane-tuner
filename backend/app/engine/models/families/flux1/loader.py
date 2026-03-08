"""FLUX.1 model loader — manifest-driven via GenericComponentLoader.

Components:
- Transformer: ``FluxTransformer2DModel`` (multi-shard safetensors)
- VAE: ``AutoencoderKL`` (standard diffusers)
- Text encoder 1: ``CLIPTextModel`` (pooled embeddings)
- Text encoder 2: ``T5EncoderModel`` (sequence embeddings)
- Tokenizers: ``CLIPTokenizer``, ``T5TokenizerFast``
"""

from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader
from app.engine.core.definitions import ModelDefinition


class Flux1Loader(GenericComponentLoader):
    """Load FLUX.1 Dev / Schnell from a diffusers-format repository."""

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        return [
            # -- Tokenizers --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.CLIPTokenizer",
                subfolder="tokenizer",
                is_torch_model=False,
                fallback_to_root=True,
            ),
            ComponentSpec(
                key="tokenizer_2",
                hf_class="transformers.T5TokenizerFast",
                subfolder="tokenizer_2",
                is_torch_model=False,
                fallback_to_root=True,
            ),
            # -- Text Encoders --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.CLIPTextModel",
                subfolder="text_encoder",
                fallback_to_root=True,
            ),
            ComponentSpec(
                key="text_encoder_2",
                hf_class="transformers.T5EncoderModel",
                subfolder="text_encoder_2",
                fallback_to_root=True,
            ),
            # -- VAE --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKL",
                subfolder="vae",
                fallback_to_root=True,
            ),
            # -- Transformer → mapped to "unet" for pipeline contract --
            ComponentSpec(
                key="unet",
                hf_class="diffusers.FluxTransformer2DModel",
                subfolder="transformer",
                fallback_to_root=True,
            ),
        ]
