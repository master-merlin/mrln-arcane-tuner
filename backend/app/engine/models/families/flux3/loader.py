"""FLUX.3 Image Klein loader -- manifest-driven via GenericComponentLoader (LANE-144 spec D2).

Components (diffusers-format repo, no scheduler subfolder):
- Processor: ``transformers.AutoProcessor`` from ``text_encoder/`` (the tokenizer and the chat
  template ship inside the text-encoder folder, there is no ``tokenizer/`` subfolder).
- Text encoder: ``transformers.Qwen3VLForConditionalGeneration`` -- the concrete class, never an
  Auto* mapping (the B1 random-TE bug class, ``tests/engine/test_te_loading_contracts.py``).
- VAE: ``diffusers.AutoencoderKLFlux2`` from ``vae/`` (FLUX.2's VAE byte for byte).
- Transformer -> ``unet``: the VENDORED ``Flux3Transformer2DModel`` (diffusers PR #15046 @
  ``ec288418``; diffusers 0.41.0 ships no FLUX.3 class), see ``vendor/README.md``.

This family trains the undistilled base only. A flow-map (step-distilled) checkpoint -- its
transformer config says ``flow_map: true`` or its weights carry a ``target_timestep_embedder`` --
is refused BEFORE any component loads, with an error naming the base repository.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader

BASE_REPO_ID = "black-forest-labs/flux-3-image-klein-base"
VENDORED_TRANSFORMER = "app.engine.models.families.flux3.vendor.transformer_flux3.Flux3Transformer2DModel"
_FLOW_MAP_PREFIX = "target_timestep_embedder."


def _transformer_dir(root: Path) -> Path:
    tdir = root / "transformer"
    return tdir if tdir.is_dir() else root


def _weight_keys(tdir: Path) -> list[str]:
    """The transformer's tensor names, read from the shard index or the safetensors headers only."""
    index = tdir / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        return list(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}))
    shards = sorted(tdir.glob("*.safetensors"))
    if not shards:
        return []
    from safetensors import safe_open

    keys: list[str] = []
    for shard in shards:
        with safe_open(str(shard), framework="pt") as handle:
            keys.extend(handle.keys())
    return keys


def flow_map_reason(root: str | Path) -> str | None:
    """Why the checkpoint at ``root`` is a flow-map (distilled) transformer, or ``None`` if it is not."""
    tdir = _transformer_dir(Path(root))
    config_path = tdir / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("flow_map"):
            return f"{config_path} sets flow_map: true"
    embedder = [k for k in _weight_keys(tdir) if k.startswith(_FLOW_MAP_PREFIX)]
    if embedder:
        return f"{tdir} carries a target_timestep embedder ({embedder[0]})"
    return None


class Flux3Loader(GenericComponentLoader):
    """Load FLUX.3 Image Klein (Base) from a diffusers-format repository."""

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        return [
            # -- Processor (tokenizer + chat template live in text_encoder/) --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoProcessor",
                subfolder="text_encoder",
                is_torch_model=False,
            ),
            # -- Text encoder: concrete class, byte-identical to Qwen/Qwen3-VL-4B-Instruct --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Qwen3VLForConditionalGeneration",
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

    async def load(
        self,
        definition: ModelDefinition,
        torch_dtype: torch.dtype | None = None,
        initial_device: str | None = None,
    ) -> dict[str, Any]:
        root = self._resolve_root(definition)
        reason = flow_map_reason(root)
        if reason is not None:
            raise RuntimeError(
                f"flux3: '{definition.id}' points at a flow map (step-distilled) FLUX.3 checkpoint: "
                f"{reason}. This family trains the undistilled base only -- point the definition at "
                f"huggingface:{BASE_REPO_ID} (flux-3-image-klein-base)."
            )
        self.logger.info("flux3.transformer_check", root=str(root), flow_map=False,
                         transformer_class=VENDORED_TRANSFORMER)
        return await super().load(definition, torch_dtype, initial_device)
