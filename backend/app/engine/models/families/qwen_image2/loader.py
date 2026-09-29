"""Qwen-Image 2 loader -- manifest-driven via GenericComponentLoader.

Every class is pinned to what ``Qwen/Qwen-Image-2.1``'s ``model_index.json``
declares (@ ``790c92633540``):

- ``processor``    -> ``Qwen3VLProcessor`` (the checkpoint ships a
  ``processor/`` folder, not ``tokenizer/``; upstream's ``encode_prompt`` calls
  ``processor(text=...)`` and ``processor.apply_chat_template``, so a bare
  tokenizer would not reproduce its tokenization);
- ``text_encoder`` -> ``Qwen3VLForConditionalGeneration`` -- pinned, never
  ``AutoModel``: ``AutoModel`` builds ``Qwen3VLModel`` whose parameter prefix
  misses the checkpoint's ``model.`` and random-initialises the TE with only a
  warning (the ideogram4 B1 bug class; ``tests/engine/test_te_loading_contracts.py``);
- ``vae`` / ``transformer`` -> the VENDORED ``AutoencoderKLQwenImage21`` and
  ``QwenImage21Transformer2DModel`` (diffusers 0.40, the installed release,
  exports neither; ``vendor/`` retires when a release does).
"""

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader

_VENDOR = "app.engine.models.families.qwen_image2.vendor"


class QwenImage2Loader(GenericComponentLoader):
    """Load Qwen-Image 2.x components -- processor, TE, VAE, transformer."""

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        return [
            # -- Processor (tokenizer + chat template) --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.Qwen3VLProcessor",
                subfolder="processor",
                candidates=["processor"],
                is_torch_model=False,
            ),
            # -- Text Encoder (Qwen3-VL, text-only use) --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Qwen3VLForConditionalGeneration",
                subfolder="text_encoder",
                candidates=["text_encoder"],
            ),
            # -- VAE (vendored) --
            ComponentSpec(
                key="vae",
                hf_class=f"{_VENDOR}.autoencoder_kl_qwenimage21.AutoencoderKLQwenImage21",
                subfolder="vae",
                candidates=["vae"],
            ),
            # -- Transformer (vendored) -> mapped to "unet" --
            ComponentSpec(
                key="unet",
                hf_class=f"{_VENDOR}.transformer_qwenimage21.QwenImage21Transformer2DModel",
                subfolder="transformer",
                candidates=["transformer"],
            ),
        ]
