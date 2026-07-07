"""Ovis-Image model loader — manifest-driven via GenericComponentLoader.

All four components are diffusers-0.39/transformers-native and load via
``from_pretrained`` (no vendoring, no config translation):
- ``OvisImageTransformer2DModel`` (Flux-style 6 double + 27 single MMDiT)
- ``transformers.Qwen3Model`` text encoder (text-only, hidden 2048)
- ``AutoTokenizer`` (resolves the checkpoint's fast Qwen2TokenizerFast)
- ``AutoencoderKL`` (Flux-style 16-channel VAE)
"""

from app.engine.core.pipeline.loader_base import (
    ComponentSpec,
    GenericComponentLoader,
)
from app.engine.core.definitions import ModelDefinition


class OvisImageLoader(GenericComponentLoader):
    """Load Ovis-Image components — tokenizer, TE, VAE, transformer."""

    def get_component_manifest(
        self,
        definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        return [
            # -- Tokenizer --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoTokenizer",
                subfolder="tokenizer",
                candidates=["tokenizer"],
                is_torch_model=False,
            ),
            # -- Text Encoder --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Qwen3Model",
                subfolder="text_encoder",
                candidates=["text_encoder"],
            ),
            # -- VAE --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKL",
                subfolder="vae",
                candidates=["vae"],
            ),
            # -- Transformer → mapped to "unet" --
            # ``definition_key="transformer"`` + ``separate_repo=True`` let a
            # future definition pin the transformer to its own repo; with no
            # ``transformer`` component declared (ovis-image-base) this falls
            # back to discovering ``transformer/`` inside the repo root.
            ComponentSpec(
                key="unet",
                hf_class="diffusers.models.OvisImageTransformer2DModel",
                subfolder="transformer",
                candidates=["transformer"],
                definition_key="transformer",
                separate_repo=True,
            ),
        ]
