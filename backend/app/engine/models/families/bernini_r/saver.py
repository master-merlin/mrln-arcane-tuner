"""Bernini-R LoRA saver — wan-canonical keys, ``bernini-r`` provenance label.

Bernini-R adds ZERO new weight modules (recon §9): its checkpoints are 100%
stock Wan and load into diffusers ``WanTransformer3DModel`` verbatim, so the
diffusers → ComfyUI key conversion inherited from :class:`Wan21Saver` (the shared
:func:`wan_shared.saver_base._convert_diffusers_to_comfy`) produces
BYTE-IDENTICAL ``diffusion_model.*`` tensor keys to the wan21 single-expert
export for the same module set. ComfyUI's official Bernini-R workflow loads stock
Wan LoRA keys, so the key names MUST NOT diverge — this saver deliberately reuses
the wan21 converter and only specializes the ``modelspec.architecture`` metadata
label (``bernini-r-*``) for provenance.
"""

from __future__ import annotations

from app.engine.models.families.wan21.saver import Wan21Saver
from app.engine.models.families.wan22.saver import Wan22Saver

__all__ = ["BerniniRSaver", "BerniniRDualSaver"]


class BerniniRSaver(Wan21Saver):
    """Bernini-R single-expert (1.3B) LoRA saver.

    Identical export path to :class:`Wan21Saver` (tensor keys byte-for-byte the
    same); the only difference is the ``modelspec.architecture`` label, driven by
    the overridden :pyattr:`ARCH_PREFIX`.
    """

    ARCH_PREFIX = "bernini-r"


class BerniniRDualSaver(Wan22Saver):
    """Bernini-R dual-expert (14B MoE) LoRA saver — TWO ComfyUI-format files.

    Bernini-R 14B is 100%-stock wan2.2-A14B-arch weights (recon §9), so the
    dual-expert export IS the wan2.2 one: reuses :class:`Wan22Saver` verbatim for
    the ``{stem}_high_noise`` / ``{stem}_low_noise`` filenames and the shared
    ``diffusion_model.*`` key conversion, and only relabels
    ``modelspec.architecture`` (``bernini-r-{mode}-{high,low}``) for provenance —
    exactly as the single-expert :class:`BerniniRSaver` relabels wan21's export.
    ComfyUI's official Bernini-R workflow loads stock wan2.2 per-expert LoRA keys,
    so the tensor keys MUST NOT diverge from wan22's.
    """

    ARCH_FAMILY = "bernini-r"
