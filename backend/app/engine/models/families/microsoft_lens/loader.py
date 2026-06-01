"""Microsoft Lens loader.

Stock components (tokenizer / text_encoder / vae) load through the generic
manifest path. The vendored DiT loads via the HiDream-O1 direct-safetensors
pattern (init_empty_weights + per-shard load_file + load_state_dict), because
``LensTransformer2DModel`` is not registered in the ``diffusers`` namespace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import (
    ComponentSpec,
    GenericComponentLoader,
)


class MicrosoftLensLoader(GenericComponentLoader):
    """Load Lens components; DiT via direct-safetensors into the vendored class."""

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.PreTrainedTokenizerFast",
                subfolder="tokenizer",
                is_torch_model=False,
                fallback_to_root=True,
            ),
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.GptOssForCausalLM",
                subfolder="text_encoder",
                fallback_to_root=True,
                # Dequantize MXFP4 -> bf16 so we never need Triton MoE kernels
                # on Windows. 96 GB VRAM holds the dequantized 20B encoder.
                load_kwargs={"quantization_config": None},
            ),
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKLFlux2",
                subfolder="vae",
                fallback_to_root=True,
            ),
        ]

    async def load(
        self,
        definition: ModelDefinition,
        torch_dtype: torch.dtype | None = None,
        initial_device: str | None = None,
    ) -> dict[str, Any]:
        """Load Lens components and the vendored DiT.

        Loads stock components (tokenizer, text_encoder, vae) via the generic
        manifest path, then loads the vendored ``LensTransformer2DModel`` directly
        from safetensors shards using ``init_empty_weights`` + ``load_state_dict``,
        because the class is not registered in the ``diffusers`` namespace.

        Args:
            definition: Model definition with component paths/repo IDs.
            torch_dtype: Dtype for the DiT weights. Defaults to ``bfloat16``.
            initial_device: Device to place the DiT on after load. ``None``
                defaults to ``self.device``.

        Returns:
            Dict of loaded components keyed by name, including ``"unet"`` for
            the ``LensTransformer2DModel`` instance.
        """
        # 1. Load stock components via the generic path.
        components = await super().load(definition, torch_dtype, initial_device)

        # 2. Load the vendored DiT by hand (direct safetensors).
        dtype = torch_dtype or torch.bfloat16
        target_device = torch.device(
            initial_device if initial_device is not None else str(self.device),
        )
        root = Path(self._root_path)
        transformer_dir = root / "transformer"
        if not transformer_dir.is_dir():
            # Non-standard layout: only fall back to root if it actually holds a
            # transformer config. Without this guard, root-level shards from other
            # components (vae, text_encoder) would be merged into the DiT state dict.
            if not (root / "config.json").is_file():
                raise FileNotFoundError(
                    f"No 'transformer/' subfolder and no config.json at root: {root}. "
                    "Place the Lens DiT weights in a 'transformer/' subdirectory."
                )
            self.logger.warning(
                "microsoft_lens.transformer_dir_fallback",
                root=str(root),
                message="transformer/ subfolder not found; falling back to root.",
            )
            transformer_dir = root

        from app.engine.models.families.microsoft_lens.vendor.transformer import (
            LensTransformer2DModel,
        )

        # Empty-init from config.json (diffusers filters _class_name etc.).
        config = LensTransformer2DModel.load_config(str(transformer_dir))
        with init_empty_weights():
            model = LensTransformer2DModel.from_config(config)
        model = model.to(dtype=dtype)

        shard_files = sorted(transformer_dir.glob("*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors in {transformer_dir}")
        state_dict: dict[str, torch.Tensor] = {}
        for shard in shard_files:
            try:
                state_dict.update(load_file(str(shard)))
            except Exception as e:
                self.logger.error(
                    "microsoft_lens.shard_load_failed", shard=shard.name, error=str(e),
                )
                raise RuntimeError(f"failed to load shard {shard.name}: {e}") from e

        missing, unexpected = model.load_state_dict(
            state_dict, strict=False, assign=True,
        )
        if missing:
            self.warnings.append(
                f"DiT: {len(missing)} missing key(s) (first 5: {list(missing)[:5]})",
            )
        if unexpected:
            self.warnings.append(
                f"DiT: {len(unexpected)} unexpected key(s) "
                f"(first 5: {list(unexpected)[:5]})",
            )

        # Materialize any residual meta params as zeros, then move to device.
        for name, p in [
            (n, p) for n, p in model.named_parameters() if p.device.type == "meta"
        ]:
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(
                parent, parts[-1],
                torch.nn.Parameter(
                    torch.zeros(p.shape, dtype=p.dtype),
                    requires_grad=p.requires_grad,
                ),
            )
        model = model.to(target_device)
        model.eval()

        components["unet"] = model
        self.logger.info(
            "microsoft_lens.load.complete",
            components=list(components.keys()),
        )
        return components
