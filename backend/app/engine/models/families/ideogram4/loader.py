"""Ideogram 4 loader.

Stock components load through the generic manifest path:

- ``tokenizer`` / ``text_encoder`` — Qwen3-VL, which lives in a **separate**
  repository (``Qwen/Qwen3-VL-8B-Instruct``).  The repo id is declared in the
  model definition YAML under the ``text_encoder`` component key; the specs only
  set ``separate_repo=True`` + ``definition_key="text_encoder"`` so the generic
  loader resolves that repo independently of the fp8 DiT root.
- ``vae`` — upstream uses a **custom** ``AutoEncoder`` (``ideogram4/
  autoencoder.py``), NOT a ``diffusers``-native class, and loads it via
  ``convert_diffusers_state_dict`` + manual ``load_state_dict``.  That class
  must be vendored (see the TODO below) before the VAE actually loads; vendoring
  it is out of scope for this task.  The spec declares the intended vendored
  class name so the manifest contract is locked.

The DiT is loaded by the overridden :meth:`IdeogramV4Loader.load` (not in the
manifest) via the HiDream-O1 direct-safetensors pattern — ``init_empty_weights``
+ per-shard ``load_file`` + ``load_state_dict`` — because
``Ideogram4Transformer2DModel`` is not registered in the ``diffusers`` namespace.
The fp8 repo stores each ``X.weight`` (float8) paired with ``X.weight_scale``
(float32, per-output-channel); :func:`dequantize_fp8_state_dict` reconstructs the
real weights BEFORE ``load_state_dict``.
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

from .utils import dequantize_fp8_state_dict


class IdeogramV4Loader(GenericComponentLoader):
    """Load Ideogram 4 components; DiT via direct-safetensors + fp8 dequant."""

    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoTokenizer",
                subfolder="tokenizer",
                is_torch_model=False,
                separate_repo=True,
                definition_key="text_encoder",
                fallback_to_root=True,
            ),
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.AutoModel",
                subfolder="text_encoder",
                separate_repo=True,
                definition_key="text_encoder",
                fallback_to_root=True,
            ),
            # TODO(ideogram4-vae): custom VAE needs vendoring. Upstream
            # ``ideogram4/autoencoder.py`` defines ``AutoEncoder`` (a bespoke
            # nn.Module with a BatchNorm head), loaded via
            # ``convert_diffusers_state_dict`` + manual ``load_state_dict`` from
            # ``vae/diffusion_pytorch_model.safetensors`` — NOT ``from_pretrained``
            # and NOT a diffusers-native class. Until it is vendored as
            # ``vendor/autoencoder_ideogram4.py::Ideogram4AutoEncoder`` (with a
            # custom load path mirroring the DiT below), the generic manifest path
            # cannot load it. The spec is declared here to lock the contract.
            ComponentSpec(
                key="vae",
                hf_class=(
                    "app.engine.models.families.ideogram4.vendor."
                    "autoencoder_ideogram4.Ideogram4AutoEncoder"
                ),
                subfolder="vae",
                dtype_override=torch.float32,
                fallback_to_root=True,
            ),
        ]

    async def load(
        self,
        definition: ModelDefinition,
        torch_dtype: torch.dtype | None = None,
        initial_device: str | None = None,
    ) -> dict[str, Any]:
        """Load Ideogram 4 components and the vendored fp8 DiT.

        Loads stock components (tokenizer, text_encoder, vae) via the generic
        manifest path, then loads the vendored ``Ideogram4Transformer2DModel``
        directly from the fp8 safetensors shards in the ``transformer/``
        subfolder, because the class is not registered in the ``diffusers``
        namespace.  The KEY addition vs the Lens loader: the raw shard state
        dict is passed through :func:`dequantize_fp8_state_dict` (float8 weight
        + float32 per-output-channel ``weight_scale`` -> real weight) BEFORE
        ``load_state_dict``, then any remaining tensors are cast to the load
        dtype.

        Args:
            definition: Model definition with component paths/repo IDs.
            torch_dtype: Dtype for the DiT weights. Defaults to ``bfloat16``.
            initial_device: Device to place the DiT on after load. ``None``
                defaults to ``self.device``.

        Returns:
            Dict of loaded components keyed by name, including ``"unet"`` for
            the ``Ideogram4Transformer2DModel`` instance.
        """
        # 1. Load stock components via the generic path.
        components = await super().load(definition, torch_dtype, initial_device)

        # 2. Load the vendored DiT by hand (direct safetensors + fp8 dequant).
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
                    "Place the Ideogram 4 DiT weights in a 'transformer/' subdirectory."
                )
            self.logger.warning(
                "ideogram4.transformer_dir_fallback",
                root=str(root),
                message="transformer/ subfolder not found; falling back to root.",
            )
            transformer_dir = root

        from app.engine.models.families.ideogram4.vendor.modeling_ideogram4 import (
            Ideogram4Transformer2DModel,
        )

        # Empty-init from config.json (diffusers filters _class_name etc.).
        config = Ideogram4Transformer2DModel.load_config(str(transformer_dir))
        with init_empty_weights():
            model = Ideogram4Transformer2DModel.from_config(config)

        shard_files = sorted(transformer_dir.glob("*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors in {transformer_dir}")
        raw_state_dict: dict[str, torch.Tensor] = {}
        for shard in shard_files:
            try:
                raw_state_dict.update(load_file(str(shard)))
            except Exception as e:
                self.logger.error(
                    "ideogram4.shard_load_failed", shard=shard.name, error=str(e),
                )
                raise RuntimeError(f"failed to load shard {shard.name}: {e}") from e

        # Dequantize fp8 weights (X.weight float8 + X.weight_scale float32) to
        # real float32 weights, dropping the *.weight_scale keys. Tensors without
        # a paired scale pass through untouched.
        state_dict = dequantize_fp8_state_dict(raw_state_dict)
        # Cast all tensors to the load dtype (dequant emits float32; pass-through
        # tensors keep their stored dtype). assign=True below installs them as-is.
        state_dict = {k: v.to(dtype) for k, v in state_dict.items()}

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
            "ideogram4.load.complete",
            components=list(components.keys()),
        )
        return components
