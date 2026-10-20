"""MOSS-Image loader -- manifest-driven via GenericComponentLoader (LANE-165).

Components (diffusers-format repo; Evidence: ``model_index.json`` of OpenMOSS-Team/MOSS-Image-Base
@ 1ceede80, identical in MOSS-Image-SFT @ fc5de0d5):
- Tokenizer: ``transformers.AutoTokenizer`` from ``tokenizer/`` (``Qwen2Tokenizer`` + the
  ``chat_template.jinja`` the text recipe applies).
- Text encoder: ``transformers.Qwen3_5Model`` -- the concrete class of transformers 5.14.1, never an
  Auto* mapping (the B1 random-TE bug class, ``tests/engine/test_te_loading_contracts.py``).
- VAE: ``diffusers.AutoencoderKLFlux2`` from ``vae/`` (the FLUX.2 small decoder; latents through the
  ``flux2/utils.py`` patchify + BatchNorm helpers).
- Transformer -> ``unet``: the VENDORED ``MOSSImageTransformer`` (``vendor/README.md``). The upstream
  card loads it with remote code; this loader never does -- every class it names is either a stock
  diffusers/transformers class or the reviewed vendored copy.

Both definitions download through the same HF path as every family, at the 40-hex revision pinned in
their ``components.repo.path`` (``huggingface:<repo>@<sha>``, ``app/engine/utils/model_utils.py``).
"""

from __future__ import annotations

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader

VENDORED_TRANSFORMER = (
    "app.engine.models.families.moss_image.vendor.transformer_moss_image.MOSSImageTransformer"
)


class MossImageLoader(GenericComponentLoader):
    """Load MOSS-Image (Base or SFT) from its diffusers-format repository."""

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        return [
            # -- Tokenizer (Qwen2 + chat template) --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoTokenizer",
                subfolder="tokenizer",
                is_torch_model=False,
            ),
            # -- Text encoder: Qwen3.5-9B, concrete class --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Qwen3_5Model",
                subfolder="text_encoder",
            ),
            # -- VAE --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKLFlux2",
                subfolder="vae",
            ),
            # -- Transformer -> "unet" (vendored) --
            ComponentSpec(
                key="unet",
                hf_class=VENDORED_TRANSFORMER,
                subfolder="transformer",
            ),
        ]
