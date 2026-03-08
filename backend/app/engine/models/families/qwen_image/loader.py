"""Qwen-Image model loader — manifest-driven via GenericComponentLoader.

Uses diffusers ``from_pretrained`` for all components:
- ``QwenImageTransformer2DModel`` (60 layers, 24 heads, 20B params)
- ``Qwen2_5_VLForConditionalGeneration`` (text encoder — VL model in text-only mode)
- ``AutoencoderKLQwenImage`` (custom VAE, falls back to AutoencoderKL)
"""

from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader
from app.engine.core.definitions import ModelDefinition


class QwenImageLoader(GenericComponentLoader):
    """Load Qwen-Image (2512) components — tokenizer, TE, VAE, transformer."""

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        # Determine best VAE class — try QwenImage-specific, fall back
        vae_class = self._detect_vae_class()

        return [
            # -- Tokenizer --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoTokenizer",
                subfolder="tokenizer",
                candidates=["tokenizer"],
                is_torch_model=False,
            ),
            # -- Text Encoder (Qwen2.5-VL) --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Qwen2_5_VLForConditionalGeneration",
                subfolder="text_encoder",
                candidates=["text_encoder"],
            ),
            # -- VAE --
            ComponentSpec(
                key="vae",
                hf_class=vae_class,
                subfolder="vae",
                candidates=["vae"],
            ),
            # -- Transformer → mapped to "unet" --
            ComponentSpec(
                key="unet",
                hf_class="diffusers.models.QwenImageTransformer2DModel",
                subfolder="transformer",
                candidates=["transformer"],
            ),
        ]

    @staticmethod
    def _detect_vae_class() -> str:
        """Detect whether the QwenImage-specific VAE class is available."""
        try:
            from diffusers.models import AutoencoderKLQwenImage  # noqa: F401
            return "diffusers.models.AutoencoderKLQwenImage"
        except ImportError:
            return "diffusers.AutoencoderKL"
