"""Bernini-R loader — component-wise off the repo root.

The HF repos (``ByteDance/Bernini-R-1.3B-Diffusers`` and ``…-Diffusers``) have
**no** ``model_index.json`` — the root ``config.json`` is a transformers-style
``bernini_renderer`` config. So this is NOT a ``DiffusionPipeline.from_pretrained``
layout; components are loaded by subfolder exactly like upstream ``GEN_Wanx22``.

Components (all stock classes present in diffusers 0.39 / transformers 4.57.x):
- ``tokenizer``    : ``AutoTokenizer`` (UMT5 ``T5Tokenizer`` / ``spiece.model``)
- ``text_encoder`` : ``UMT5EncoderModel`` — repo ships fp32; cast to bf16 at load
- ``vae``          : ``AutoencoderKLWan`` (Wan 2.1, z=16) — kept fp32
- ``unet``         : ``WanTransformer3DModel`` from ``transformer/`` — repo ships
                     fp32 shards; cast to bf16 at load

Single vs dual expert (recon §1/§3)
-----------------------------------
- 1.3B (``skip_transformer_2: true``): ONE expert. No ``transformer_2`` subfolder
  in the repo, so the manifest carries only ``transformer/`` → ``unet``. This is
  the byte-identical v1 path.
- 14B (``dual_expert: true``, ``skip_transformer_2: false``): TWO experts.
  ``transformer/`` = the HIGH-noise expert (active for t >= boundary·1000) and
  ``transformer_2/`` = the LOW-noise expert (t < boundary). Mirrors the
  :class:`Wan22Loader` MoE manifest — ``expert_mode`` selects ``both`` (high →
  ``unet``, low → ``unet_low``) or a single expert for a one-expert run.
"""

from __future__ import annotations

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader


class BerniniRLoader(GenericComponentLoader):
    """Load Bernini-R components by subfolder off the repo root.

    Args:
        device: Target device for loaded components.
        expert_mode: Dual-expert (14B) selection — ``"both"`` (default) loads
            ``transformer/`` (high → ``unet``) and ``transformer_2/`` (low →
            ``unet_low``); ``"high"``/``"low"`` load only that expert (mapped to
            ``unet``) for a single-expert run. Ignored for the single-expert
            1.3B (no ``transformer_2`` subfolder exists).
    """

    def __init__(self, device, expert_mode: str = "both") -> None:
        super().__init__(device)
        self.expert_mode = str(expert_mode or "both").lower()

    # ── Transformer specs (shared by the manifest + any deferred load) ──────

    @staticmethod
    def _high_expert_spec(key: str = "unet") -> ComponentSpec:
        """High-noise expert (``transformer/``) — primary ``unet`` key."""
        return ComponentSpec(
            key=key,
            hf_class="diffusers.WanTransformer3DModel",
            subfolder="transformer",
            candidates=["transformer"],
            fallback_to_root=True,
        )

    @staticmethod
    def _low_expert_spec(key: str = "unet_low") -> ComponentSpec:
        """Low-noise expert (``transformer_2/``); ``key`` is ``unet_low`` in
        ``both`` mode, or ``unet`` when ``low`` is the single loaded expert."""
        return ComponentSpec(
            key=key,
            hf_class="diffusers.WanTransformer3DModel",
            subfolder="transformer_2",
            candidates=["transformer_2"],
            fallback_to_root=True,
        )

    @staticmethod
    def _is_dual_expert(definition: ModelDefinition) -> bool:
        """True for the 14B MoE (``dual_expert``), False for the 1.3B."""
        arch = getattr(definition, "architecture_params", {}) or {}
        return bool(arch.get("dual_expert", False))

    def get_component_manifest(
        self, definition: ModelDefinition
    ) -> list[ComponentSpec]:
        manifest: list[ComponentSpec] = [
            # -- Tokenizer (UMT5) --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoTokenizer",
                subfolder="tokenizer",
                candidates=["tokenizer"],
                is_torch_model=False,
                fallback_to_root=True,
            ),
            # -- Text Encoder (UMT5-XXL) — repo fp32; loader casts to bf16 --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.UMT5EncoderModel",
                subfolder="text_encoder",
                candidates=["text_encoder"],
                fallback_to_root=True,
            ),
            # -- VAE (Wan 2.1, z=16) — kept fp32 for temporal-decode precision --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKLWan",
                subfolder="vae",
                candidates=["vae"],
                dtype_override=torch.float32,
                fallback_to_root=True,
            ),
        ]

        # -- Transformer(s) → "unet" (+ "unet_low" on 14B both mode) --
        if not self._is_dual_expert(definition):
            # 1.3B single expert — byte-identical to the v1 manifest.
            manifest.append(self._high_expert_spec())
        elif self.expert_mode == "high":
            manifest.append(self._high_expert_spec())
        elif self.expert_mode == "low":
            # Load transformer_2/ AS "unet" so it becomes the single primary.
            manifest.append(self._low_expert_spec(key="unet"))
        else:  # both — high → "unet", low → "unet_low"
            manifest.append(self._high_expert_spec())
            manifest.append(self._low_expert_spec())

        return manifest
