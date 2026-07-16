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

__all__ = ["BerniniRSaver"]


class BerniniRSaver(Wan21Saver):
    """Bernini-R LoRA saver.

    Identical export path to :class:`Wan21Saver` (tensor keys byte-for-byte the
    same); the only difference is the ``modelspec.architecture`` label, driven by
    the overridden :pyattr:`ARCH_PREFIX`.
    """

    ARCH_PREFIX = "bernini-r"
