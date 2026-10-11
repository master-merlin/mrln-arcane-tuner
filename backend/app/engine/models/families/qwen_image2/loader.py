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
- ``vae`` / ``transformer`` -> diffusers' own ``AutoencoderKLQwenImage21`` and
  ``QwenImage21Transformer2DModel`` (shipped since 0.41.0; LANE-150 retired the
  vendored copy). They are resolved lazily from these ``hf_class`` strings:
  the release transformer module guards its optional flex_attention import with
  ``except ImportError`` only, so it must never be imported at startup.
"""

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader


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
            # -- VAE --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKLQwenImage21",
                subfolder="vae",
                candidates=["vae"],
            ),
            # -- Transformer -> mapped to "unet" --
            ComponentSpec(
                key="unet",
                hf_class="diffusers.QwenImage21Transformer2DModel",
                subfolder="transformer",
                candidates=["transformer"],
            ),
        ]
