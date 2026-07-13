"""OmniGen2EditTrainer — image-conditioned ("edit") variant.

Subclasses :class:`OmniGen2Trainer`. Deliberately MINIMAL — the two edit
mechanisms other edit families need at the trainer level do not apply here:

1. **No forward override** (like boogu, unlike qwen_image/flux1-kontext):
   the vendored transformer consumes clean control latents natively via
   ``ref_image_hidden_states`` — wired family-wide in
   ``OmniGen2Driver.forward_pass`` / ``_build_ref_image_hidden_states``
   off ``batch["control_latents"]``.

2. **No composite (caption, control) TE-cache keys** (UNLIKE boogu — a
   deliberate, evidence-backed difference): boogu's Qwen3-VL text encoder
   ATTENDS the control pixels (its chat template interleaves the image),
   so its edit trainer must key the TE cache on (caption, control-bytes).
   OmniGen2's mllm is invoked with ``text_input_ids + attention_mask``
   ONLY — no ``pixel_values``, no vision tower — in BOTH the upstream
   pipeline (``_get_qwen2_prompt_embeds``, pipeline_omnigen2.py ~L324-328)
   and upstream training (train.py L515-519). Text embeddings are
   therefore control-independent and the base trainer's plain per-caption
   cache (and its disk pre-cache, which stays ENABLED here, unlike boogu's
   edit trainer) is exactly correct. See driver.py module docstring §1 for
   the full recon citations.

What this subclass owns: control-fed previews (``OmniGen2EditSampler``).
"""

from __future__ import annotations

import structlog

from .trainer import OmniGen2Trainer

logger = structlog.get_logger(__name__)


class OmniGen2EditTrainer(OmniGen2Trainer):
    """OmniGen2 Edit trainer — control-fed previews; everything else is the
    base trainer (see module docstring for why that is correct, not lazy)."""

    def _create_sampler(self):
        """Edit-aware previews (control latents fed each step + the
        pipeline's image-guidance CFG semantics)."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler_edit import OmniGen2EditSampler  # noqa: PLC0415

            return OmniGen2EditSampler(self)
        return None
