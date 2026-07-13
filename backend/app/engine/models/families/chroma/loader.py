"""Chroma model loader — manifest-driven via GenericComponentLoader.

Components (verified against ``lodestones/Chroma1-HD``'s / ``Chroma1-Base``'s
``model_index.json`` + per-subfolder ``config.json`` — both ship a full
diffusers-layout repo, not a single-file checkpoint):

- Transformer: ``ChromaTransformer2DModel`` (multi-shard safetensors,
  ``transformer/`` subfolder).
- VAE: ``AutoencoderKL`` — the FLUX.1-schnell VAE verbatim
  (``vae/config.json``: ``_name_or_path: "/home/ubuntu/FLUX.1-schnell"``).
- Text encoder: ``T5EncoderModel`` (T5-XXL, ``text_encoder/`` subfolder,
  2-shard safetensors). Chroma drops CLIP entirely — T5 is the ONLY encoder.
- Tokenizer: ``T5Tokenizer`` (SLOW/SentencePiece) — NOT ``T5TokenizerFast``.
  ``model_index.json`` pins ``["transformers", "T5Tokenizer"]`` and the repo's
  ``tokenizer/`` folder ships only ``spiece.model`` (no ``tokenizer.json``),
  confirming the pipeline authors intentionally register the slow class even
  though ``ChromaPipeline.__init__``'s type hint says ``T5TokenizerFast``
  (a docstring/type-hint-only aspiration, not what ``from_pretrained``
  actually instantiates from ``model_index.json``).
"""

from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader
from app.engine.core.definitions import ModelDefinition


class ChromaLoader(GenericComponentLoader):
    """Load Chroma (Chroma1-Base / Chroma1-HD) from a diffusers-format repo."""

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        return [
            # -- Tokenizer (slow T5Tokenizer — see module docstring) --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.T5Tokenizer",
                subfolder="tokenizer",
                is_torch_model=False,
                fallback_to_root=True,
            ),
            # -- Text Encoder (T5-XXL, only encoder — no CLIP) --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.T5EncoderModel",
                subfolder="text_encoder",
                fallback_to_root=True,
            ),
            # -- VAE (FLUX.1-schnell AutoencoderKL) --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKL",
                subfolder="vae",
                fallback_to_root=True,
            ),
            # -- Transformer → mapped to "unet" for pipeline contract --
            ComponentSpec(
                key="unet",
                hf_class="diffusers.ChromaTransformer2DModel",
                subfolder="transformer",
                fallback_to_root=True,
            ),
        ]
