"""Ideogram 4 loader.

Stock components load through the generic manifest path:

- ``tokenizer`` / ``text_encoder`` — Qwen3-VL, which lives in a **separate**
  repository (``Qwen/Qwen3-VL-8B-Instruct``).  The repo id is declared in the
  model definition YAML under the ``text_encoder`` component key; the specs only
  set ``separate_repo=True`` + ``definition_key="text_encoder"`` so the generic
  loader resolves that repo independently of the fp8 DiT root.
- ``vae`` — the checkpoint's ``vae/`` folder is a diffusers-layout
  ``AutoencoderKLFlux2`` (its config's ``_class_name``), loaded in float32. Its
  ``bn`` head is never read: the driver normalises latents with the ported
  ``LATENT_SHIFT`` / ``LATENT_SCALE`` (utils.py).

The DiT (``unet``) is loaded by the overridden :meth:`IdeogramV4Loader.load`
(not in the manifest) as diffusers' ``Ideogram4Transformer2DModel`` via the
direct-safetensors pattern — ``init_empty_weights`` + per-shard ``load_file`` +
``load_state_dict`` — because the fp8 repo stores each ``X.weight`` (float8)
paired with ``X.weight_scale`` (float32, per-output-channel) and the ported
FUSED ``attention.qkv`` / ``attention.o`` layout:
:func:`~.utils.dequantize_fp8_state_dict` reconstructs the real weights, then
:func:`~.utils.convert_transformer_state_dict` splits them into the release's
``to_q`` / ``to_k`` / ``to_v`` / ``to_out.0``, and
:func:`~.utils.translate_transformer_config` reads either config spelling.
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

from .utils import (
    convert_transformer_state_dict,
    dequantize_fp8_state_dict,
    translate_transformer_config,
)


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
                # MUST be Qwen3VLForConditionalGeneration, NOT AutoModel.
                # The Qwen/Qwen3-VL-8B-Instruct checkpoint's architecture is
                # ``Qwen3VLForConditionalGeneration`` with weights prefixed
                # ``model.language_model.*`` / ``model.visual.*`` / ``lm_head.*``.
                # ``AutoModel`` instantiates ``Qwen3VLModel`` (params
                # ``language_model.*``, NO ``model.`` prefix) -> prefix mismatch
                # -> ``from_pretrained`` reports EVERY weight "newly initialized"
                # (random) and only WARNS, silently yielding a random TE. Verified
                # via .agent/workdir/ig4_te_load_check.py: with AutoModel
                # ``torch.allclose(loaded q_proj, checkpoint)`` is False (random);
                # with ForConditionalGeneration it is True (real weights, 0
                # "newly initialized" warnings). The driver resolves the text
                # tower via ``.model.language_model`` (a ``Qwen3VLTextModel`` with
                # 36 layers). ai-toolkit gets away with AutoModel only because its
                # transformers build maps Qwen3-VL's AutoModel to the FCG class.
                key="text_encoder",
                hf_class="transformers.Qwen3VLForConditionalGeneration",
                subfolder="text_encoder",
                separate_repo=True,
                definition_key="text_encoder",
                fallback_to_root=True,
            ),
            ComponentSpec(
                # diffusers-layout folder (``_class_name: AutoencoderKLFlux2``);
                # fp32 like upstream, whatever the DiT load dtype.
                key="vae",
                hf_class="diffusers.AutoencoderKLFlux2",
                subfolder="vae",
                dtype_override=torch.float32,
            ),
        ]

    async def load(
        self,
        definition: ModelDefinition,
        torch_dtype: torch.dtype | None = None,
        initial_device: str | None = None,
    ) -> dict[str, Any]:
        """Load the manifest components, then the fp8 DiT by hand.

        Args:
            definition: Model definition with component paths/repo IDs.
            torch_dtype: Dtype for the DiT weights. Defaults to ``bfloat16``.
            initial_device: Device to place the DiT on after load. ``None``
                defaults to ``self.device``.

        Returns:
            Dict of loaded components keyed by name, including ``"unet"`` (the
            ``Ideogram4Transformer2DModel``) and ``"vae"`` (``AutoencoderKLFlux2``).
        """
        # 1. Stock components (tokenizer, text_encoder, vae) via the generic path.
        components = await super().load(definition, torch_dtype, initial_device)

        # 2. The DiT by hand (direct safetensors + fp8 dequant + key conversion).
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

        components["unet"] = self._load_transformer(transformer_dir, dtype, target_device)
        self.logger.info(
            "ideogram4.load.complete",
            components=list(components.keys()),
        )
        return components

    def _load_transformer(
        self, transformer_dir: Path, dtype: torch.dtype, target_device: torch.device,
    ) -> torch.nn.Module:
        """Build ``Ideogram4Transformer2DModel`` from a (fp8, fused) shard folder."""
        from diffusers import Ideogram4Transformer2DModel

        # Empty-init from config.json in the release spelling (diffusers filters
        # _class_name etc.; a ported emb_dim/num_heads config is translated).
        config = translate_transformer_config(
            dict(Ideogram4Transformer2DModel.load_config(str(transformer_dir))),
        )
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
        # real float32 weights, dropping the *.weight_scale keys (tensors without
        # a paired scale pass through), THEN split the fused attention weights
        # into the release module names.
        state_dict = convert_transformer_state_dict(dequantize_fp8_state_dict(raw_state_dict))
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
        return model
