"""LongCat-Image model loader — manifest-driven via GenericComponentLoader.

Uses diffusers ``from_pretrained`` for all components:
- ``LongCatImageTransformer2DModel`` (Flux-like double+single DiT, native 0.39)
- ``Qwen2_5_VLForConditionalGeneration`` (text encoder — same class as qwen_image)
- ``Qwen2VLProcessor`` (``text_processor`` — extra pipeline component vs zimage;
  used by the reference pipeline's prompt-rewrite step)
- ``AutoencoderKL`` (standard 16-channel diffusers VAE)
"""

from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader
from app.engine.core.definitions import ModelDefinition


class LongCatImageLoader(GenericComponentLoader):
    """Load LongCat-Image components — tokenizer, processor, TE, VAE, transformer."""

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        return [
            # -- Tokenizer --
            # AutoTokenizer (not the slow Qwen2Tokenizer): resolves the fast
            # tokenizer from tokenizer.json — same policy as qwen_image, which
            # shares this exact TE/tokenizer stack.
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoTokenizer",
                subfolder="tokenizer",
                candidates=["tokenizer"],
                is_torch_model=False,
            ),
            # -- Text Processor (Qwen2VLProcessor) --
            # Registered pipeline component (LongCatImagePipeline.text_processor).
            # Only the prompt-rewrite step consumes it; loaded for component
            # parity so the checkpoint contract stays complete.
            ComponentSpec(
                key="text_processor",
                hf_class="transformers.Qwen2VLProcessor",
                subfolder="text_processor",
                candidates=["text_processor"],
                is_torch_model=False,
            ),
            # -- Text Encoder (Qwen2.5-VL, text-only mode) --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Qwen2_5_VLForConditionalGeneration",
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
            ComponentSpec(
                key="unet",
                hf_class="diffusers.models.LongCatImageTransformer2DModel",
                subfolder="transformer",
                candidates=["transformer"],
            ),
        ]
