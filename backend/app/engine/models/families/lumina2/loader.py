"""Lumina-Image-2.0 model loader — manifest-driven via GenericComponentLoader.

All four components are diffusers-0.39/transformers-native and load via
``from_pretrained`` (no vendoring, no config translation):
- ``Lumina2Transformer2DModel`` (26-layer + 2 context-refiner + 2 noise-
  refiner blocks, ``diffusers.Lumina2Transformer2DModel`` — verified
  importable top-level, ``venv/Lib/site-packages/diffusers/__init__.py``
  lines 287/1155)
- ``transformers.Gemma2Model`` text encoder (``model_index.json``:
  ``["transformers", "Gemma2Model"]`` — the bare encoder, NOT
  ``Gemma2ForCausalLM``)
- ``AutoTokenizer`` (resolves the checkpoint's ``GemmaTokenizerFast`` — the
  repo ships ``tokenizer/tokenizer.json``, so the fast tokenizer loads)
- ``AutoencoderKL`` (the FLUX.1-dev VAE verbatim — ``vae/config.json``:
  ``"_name_or_path": "black-forest-labs/FLUX.1-dev"``, 16-channel)
"""

from app.engine.core.pipeline.loader_base import (
    ComponentSpec,
    GenericComponentLoader,
)
from app.engine.core.definitions import ModelDefinition


class Lumina2Loader(GenericComponentLoader):
    """Load Lumina-Image-2.0 components — tokenizer, TE, VAE, transformer."""

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
                hf_class="transformers.Gemma2Model",
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
                hf_class="diffusers.Lumina2Transformer2DModel",
                subfolder="transformer",
                candidates=["transformer"],
                definition_key="transformer",
                separate_repo=True,
            ),
        ]
