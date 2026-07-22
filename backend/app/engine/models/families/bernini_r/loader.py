"""Bernini-R loader — component-wise off the repo root.

The HF repos (``ByteDance/Bernini-R-1.3B-Diffusers`` and ``…-Diffusers``) have
**no** ``model_index.json`` — the root ``config.json`` is a transformers-style
``bernini_renderer`` config. So this is NOT a ``DiffusionPipeline.from_pretrained``
layout; components are loaded by subfolder exactly like upstream ``GEN_Wanx22``.

All components are stock Wan classes off the same subfolder layout as WAN 2.2
(UMT5 tokenizer/text encoder, Wan2.1 VAE, ``WanTransformer3DModel`` experts), so
this loader subclasses :class:`Wan22Loader` and inherits the specs, the
``expert_mode`` dispatch, AND the ``defer_second_expert`` host-RAM sequencing +
:meth:`~Wan22Loader.load_second_expert` (two ~28 GB experts must never sit on
CPU together — see the wan22 module docstring for the reported 64 GB-box hang).

Single vs dual expert (recon §1/§3)
-----------------------------------
- 1.3B (``skip_transformer_2: true``): ONE expert. No ``transformer_2`` subfolder
  in the repo, so the manifest carries only ``transformer/`` → ``unet``. This is
  the byte-identical v1 path.
- 14B (``dual_expert: true``, ``skip_transformer_2: false``): TWO experts.
  ``transformer/`` = the HIGH-noise expert (active for t >= boundary·1000) and
  ``transformer_2/`` = the LOW-noise expert (t < boundary) — the inherited
  :class:`Wan22Loader` MoE manifest verbatim, including the deferred low-expert
  load for ``both`` runs.
"""

from __future__ import annotations

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec
from app.engine.models.families.wan22.loader import Wan22Loader


class BerniniRLoader(Wan22Loader):
    """Load Bernini-R components by subfolder off the repo root.

    The 14B (``dual_expert: true``) path IS the inherited :class:`Wan22Loader`
    manifest (both experts, ``expert_mode`` selection, ``defer_second_expert``
    host-RAM sequencing). The single-expert 1.3B has no ``transformer_2``
    subfolder at all, so its manifest carries exactly one transformer regardless
    of ``expert_mode``.
    """

    SECOND_EXPERT_LOG_EVENT = "bernini_r_load_second_expert"

    @staticmethod
    def _is_dual_expert(definition: ModelDefinition) -> bool:
        """True for the 14B MoE (``dual_expert``), False for the 1.3B."""
        arch = getattr(definition, "architecture_params", {}) or {}
        return bool(arch.get("dual_expert", False))

    def get_component_manifest(
        self, definition: ModelDefinition
    ) -> list[ComponentSpec]:
        if self._is_dual_expert(definition):
            # 14B MoE — the wan22 manifest verbatim (expert_mode + deferral).
            return super().get_component_manifest(definition)
        # 1.3B single expert — byte-identical to the v1 manifest.
        return [*self._base_component_specs(), self._high_expert_spec()]
