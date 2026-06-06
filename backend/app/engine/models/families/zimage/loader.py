"""Z-Image model loader — manifest-driven via GenericComponentLoader.

Uses diffusers ``from_pretrained`` for all components:
- ``ZImageTransformer2DModel`` (S3-DiT, single-stream architecture)
- Text encoder via ``AutoModelForCausalLM`` / ``AutoTokenizer``
- ``AutoencoderKL`` (standard diffusers VAE)
"""

from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader
from app.engine.core.definitions import ModelDefinition


class ZImageLoader(GenericComponentLoader):
    """Load Z-Image Base components — tokenizer, TE, VAE, transformer."""

    def get_component_manifest(
        self, definition: ModelDefinition,
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
                hf_class="transformers.AutoModelForCausalLM",
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
            # definition pin the transformer to its own repo (e.g. the
            # ostris De-Turbo, which ships only ``transformer/`` and reuses
            # the base repo for VAE/TE/tokenizer). When no ``transformer``
            # component is declared (e.g. zimage-base) this falls back to
            # discovering ``transformer/`` inside the repo root, unchanged.
            ComponentSpec(
                key="unet",
                hf_class="diffusers.models.ZImageTransformer2DModel",
                subfolder="transformer",
                candidates=["transformer"],
                definition_key="transformer",
                separate_repo=True,
            ),
        ]
