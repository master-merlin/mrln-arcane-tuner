"""Krea2 model loader — manifest-driven via GenericComponentLoader.

Stock components (tokenizer, text_encoder, vae) load through the generic
manifest path.  The transformer is NOT in the manifest because
``Krea2Transformer2DModel`` is a vendored class absent from the
``diffusers 0.38`` namespace; the generic ``from_pretrained`` string
resolver cannot build it.

Instead, :meth:`Krea2Loader.load` overrides the base ``load()`` and loads
the vendored transformer by calling
``Krea2Transformer2DModel.from_pretrained(transformer_dir, ...)`` directly
(it is a clean ``ModelMixin`` with ``register_to_config`` — no fp8 dequant
or manual shard stitching required, unlike Ideogram 4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader


class Krea2Loader(GenericComponentLoader):
    """Load Krea2 components; transformer via vendored from_pretrained override."""

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        """Return component manifest for tokenizer, text_encoder, and vae.

        The transformer is loaded by the overridden :meth:`load` (not in the
        manifest), because ``Krea2Transformer2DModel`` is vendored and is not
        registered in the ``diffusers 0.38`` namespace.
        """
        vae_class = self._detect_vae_class()

        return [
            # -- Tokenizer --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.Qwen2Tokenizer",
                subfolder="tokenizer",
                candidates=["tokenizer"],
                is_torch_model=False,
            ),
            # -- Text Encoder (Qwen3-VL base model) --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Qwen3VLModel",
                subfolder="text_encoder",
                candidates=["text_encoder"],
            ),
            # -- VAE (AutoencoderKLQwenImage — same as qwen_image) --
            ComponentSpec(
                key="vae",
                hf_class=vae_class,
                subfolder="vae",
                candidates=["vae"],
            ),
            # NOTE: transformer is NOT listed here — loaded by hand in load().
        ]

    @staticmethod
    def _detect_vae_class() -> str:
        """Detect whether the QwenImage-specific VAE class is available."""
        try:
            from diffusers.models import AutoencoderKLQwenImage  # noqa: F401
            return "diffusers.models.AutoencoderKLQwenImage"
        except ImportError:
            return "diffusers.AutoencoderKL"

    async def load(
        self,
        definition: ModelDefinition,
        torch_dtype: torch.dtype | None = None,
        initial_device: str | None = None,
    ) -> dict[str, Any]:
        """Load Krea2 components: stock via manifest, transformer via vendored class.

        Loads tokenizer / text_encoder / vae through the generic manifest path,
        then loads the vendored ``Krea2Transformer2DModel`` directly via its own
        ``from_pretrained(transformer_dir, torch_dtype=...)`` call (clean ModelMixin,
        no fp8 dequant required).

        Args:
            definition: Model definition with component paths / repo IDs.
            torch_dtype: Dtype for the transformer weights. Defaults to ``bfloat16``.
            initial_device: Device to place the transformer on after load. ``None``
                defaults to ``self.device``.

        Returns:
            Dict of loaded components including ``"unet"`` for the transformer.
        """
        # 1. Load stock components (tokenizer, text_encoder, vae) via the
        #    generic manifest path.  This also sets self._root_path.
        components = await super().load(definition, torch_dtype, initial_device)

        # 2. Load the vendored transformer by hand.
        dtype = torch_dtype or torch.bfloat16
        target_device = torch.device(
            initial_device if initial_device is not None else str(self.device),
        )
        root = Path(self._root_path)
        transformer_dir = root / "transformer"
        if not transformer_dir.is_dir():
            if not (root / "config.json").is_file():
                raise FileNotFoundError(
                    f"No 'transformer/' subfolder and no config.json at root: {root}. "
                    "Place the Krea-2 transformer weights in a 'transformer/' subdirectory."
                )
            self.logger.warning(
                "krea2.transformer_dir_fallback",
                root=str(root),
                message="transformer/ subfolder not found; falling back to root.",
            )
            transformer_dir = root

        from app.engine.models.families.krea2.vendor.transformer_krea2 import (
            Krea2Transformer2DModel,
        )

        self.logger.info(
            "krea2.loading_transformer",
            path=str(transformer_dir),
            dtype=str(dtype),
        )
        model = Krea2Transformer2DModel.from_pretrained(
            str(transformer_dir),
            torch_dtype=dtype,
        )
        model = model.to(target_device)
        model.eval()

        components["unet"] = model

        self.logger.info(
            "krea2.load.complete",
            components=list(components.keys()),
        )
        return components
