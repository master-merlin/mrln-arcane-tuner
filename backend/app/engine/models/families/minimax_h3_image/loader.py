"""MiniMax-H3-Image loader, manifest-driven through ``GenericComponentLoader``.

Repo layout (MiniMaxAI/MiniMax-H3-Image at fbde899f): ``transformer_diffusers/``
(66.28 GB, diffusers keys), ``vae_diffusers/``, ``text_encoder/`` (the H3
Qwen3-VL), ``tokenizer/``, ``processor/``.

- **Transformer**: the VENDORED ``MiniMaxH3ImageTransformer2DModel`` (diffusers
  0.41.0 ships none; ``vendor/README.md``) from ``transformer_diffusers/``. Its
  6 per-step output heads are ``pdd_proj_out.0..5`` in that checkpoint (the
  shard index names all twelve weight and bias tensors), so ``from_pretrained``
  loads them into ``pdd_proj_out``; ``_post_load_transformer`` refuses a model
  whose head count is not the definition's head grid. The transformer is
  DEFERRED out of the batch load: the 66.7 GB encoder and the 66.28 GB DiT never
  coexist, so the trainer caches every caption, releases the encoder and only
  then calls :meth:`load_transformer`.
- **VAE**: the vendored ``AutoencoderKLMiniMaxH3Image`` from ``vae_diffusers/``,
  wrapped ONCE in the shared ``H3PixelAdaptedVAE`` (the engine's ``[-1, 1]``
  pixels <-> the VAE's ImageNet space).
- **Text encoder**: ``transformers.Qwen3VLForConditionalGeneration`` (a concrete
  class, never an Auto mapping), with its ``AutoProcessor``.
"""

from __future__ import annotations

from typing import Any

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader

_VENDOR = "app.engine.models.families.minimax_h3_image.vendor"
VENDORED_TRANSFORMER = f"{_VENDOR}.transformer_minimax_h3_image.MiniMaxH3ImageTransformer2DModel"
VENDORED_VAE = f"{_VENDOR}.autoencoder_kl_minimax_h3_image.AutoencoderKLMiniMaxH3Image"


class MiniMaxH3ImageLoader(GenericComponentLoader):
    """Load MiniMax-H3-Image: processor, text encoder, VAE, and the deferred DiT."""

    def __init__(self, device, *, defer_transformer: bool = False) -> None:
        super().__init__(device)
        self.defer_transformer = bool(defer_transformer)

    @staticmethod
    def _post_load_vae(model: Any, definition: ModelDefinition) -> Any:
        from app.engine.models.families.minimax_h3_shared.pixel_adapter import H3PixelAdaptedVAE

        return H3PixelAdaptedVAE(model)

    @staticmethod
    def _post_load_transformer(model: Any, definition: ModelDefinition) -> Any:
        """One output head per grid sigma, or the run would train a head the
        sampler never uses (or index past the last one)."""
        from .settings import head_sigmas

        grid = head_sigmas(definition)
        heads = len(getattr(model, "pdd_proj_out", None) or [])
        if heads != len(grid):
            raise ValueError(
                f"minimax_h3_image {definition.id}: the transformer has {heads} output heads "
                f"(pdd_proj_out) but the definition's head grid has {len(grid)} sigmas"
            )
        return model

    @staticmethod
    def _transformer_spec(definition: ModelDefinition) -> ComponentSpec:
        return ComponentSpec(
            key="transformer",
            hf_class=VENDORED_TRANSFORMER,
            subfolder="transformer_diffusers",
            post_load_hook="_post_load_transformer",
        )

    def load_transformer(
        self,
        definition: ModelDefinition,
        torch_dtype: torch.dtype,
        initial_device: str = "cpu",
    ) -> Any:
        """Materialise the deferred DiT through the same single-spec path the
        batch load uses (root resolution, ``from_pretrained``, the post-load hook)."""
        spec = self._transformer_spec(definition)
        root_path = getattr(self, "_root_path", None) or self._resolve_root(definition)
        self.logger.info(
            "minimax_h3_image_deferred_transformer_materializing",
            subfolder=spec.subfolder,
            dtype=str(torch_dtype),
            device=str(initial_device),
        )
        return self._load_single_spec(spec, definition, root_path, torch_dtype, initial_device)

    def get_component_manifest(self, definition: ModelDefinition) -> list[ComponentSpec]:
        manifest = [
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoProcessor",
                subfolder="processor",
                candidates=["processor", "tokenizer"],
                is_torch_model=False,
            ),
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Qwen3VLForConditionalGeneration",
                subfolder="text_encoder",
            ),
            ComponentSpec(
                key="vae",
                hf_class=VENDORED_VAE,
                subfolder="vae_diffusers",
                post_load_hook="_post_load_vae",
            ),
        ]
        if not getattr(self, "defer_transformer", False):
            manifest.append(self._transformer_spec(definition))
        return manifest
