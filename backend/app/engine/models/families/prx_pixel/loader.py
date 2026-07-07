"""PRX Pixel model loader — manifest-driven via GenericComponentLoader.

Mirrors exactly what ``PRXPixelPipeline.from_pretrained`` materializes from
the ``Photoroom/prxpixel-t2i`` checkpoint (model_index.json, verified
2026-07-08):

- ``PRXTransformer2DModel`` (pixel variant: in_channels=3, bottleneck
  img_in, resolution_embeds — native diffusers 0.39 class)
- ``Qwen3VLTextModel`` (text encoder — the Qwen3-VL TEXT backbone, no
  vision tower; a proper top-level ``transformers`` export, unlike the
  latent sibling's T5GemmaEncoder full-path quirk)
- ``AutoTokenizer`` (resolves the checkpoint's Qwen2TokenizerFast)

NO VAE: PRXPixel denoises raw RGB (``vae_scale_factor`` is hardcoded 1 in
the pipeline) — the manifest must not declare one.
"""

from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader
from app.engine.core.definitions import ModelDefinition


class PRXPixelLoader(GenericComponentLoader):
    """Load PRX Pixel components — tokenizer, TE, transformer (no VAE)."""

    def get_component_manifest(
        self,
        definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        return [
            # -- Tokenizer --
            # AutoTokenizer resolves the fast Qwen2TokenizerFast from
            # tokenizer.json (same policy as qwen_image / prx).
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoTokenizer",
                subfolder="tokenizer",
                candidates=["tokenizer"],
                is_torch_model=False,
            ),
            # -- Text Encoder (Qwen3VLTextModel — model_index.json declares
            #    ["transformers", "Qwen3VLTextModel"]) --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Qwen3VLTextModel",
                subfolder="text_encoder",
                candidates=["text_encoder"],
            ),
            # -- Transformer → mapped to "unet" --
            ComponentSpec(
                key="unet",
                hf_class="diffusers.PRXTransformer2DModel",
                subfolder="transformer",
                candidates=["transformer"],
            ),
        ]
