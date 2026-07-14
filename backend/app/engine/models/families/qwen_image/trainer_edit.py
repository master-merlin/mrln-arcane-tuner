"""Qwen-Image-Edit LoRA trainer — image-conditioned ("edit") variant.

Subclasses :class:`QwenImageTrainer`. Like FLUX.1 Kontext, the CLEAN control
image's VAE latent is patchified and sequence-concatenated AFTER the noisy
TARGET patches; the transformer's ``img_shapes`` carries one ``(F,H,W)`` entry
per image (target + each control), and the prediction is sliced back to the
target tokens before the unpatchify so ``compute_target`` (velocity on the
target latents) needs no change. This mirrors diffusers'
``QwenImageEditPipeline`` (``hidden_states = cat([latents, image_latents])``,
``noise_pred[:, :latents.size(1)]``).

Qwen-Image-Edit additionally conditions its Qwen2.5-VL text encoder on the
control image (the edit prompt template embeds a ``<|vision_start|>`` image
token, ``drop_idx=64``), so the text embeddings depend on **(caption, control
image)** jointly. We therefore key the TE cache compositely
(:func:`composite_te_key`) so the same instruction over different controls
never collides. The VL image path activates when a Qwen2VL ``processor`` is
available; absent one (e.g. weights not present) it falls back to text-only
encoding while keeping the cache keyed compositely.
"""

from __future__ import annotations

import hashlib
from typing import Any

import structlog
import torch

from .trainer import (
    PROMPT_TEMPLATE_DROP_IDX,  # noqa: F401 (kept for parity / documentation)
    QwenImageTrainer,
)

logger = structlog.get_logger(__name__)

# Edit prompt template + system-preamble drop index — copied from diffusers
# ``QwenImageEditPipeline`` (the image token makes the VL encoder attend to the
# control image). Used only when a processor + control image are available.
EDIT_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, "
    "texture, objects, background), then explain how the user's text "
    "instruction should alter or modify the image. Generate a new image that "
    "meets the user's requirements while maintaining consistency with the "
    "original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
EDIT_PROMPT_DROP_IDX = 64


# ── Pure helpers ────────────────────────────────────────────────────────────


def patchify(latent: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, int, int]:
    """``[B, C, H, W]`` → ``([B, (H/p)(W/p), C*p*p], pH, pW)`` (same as the base)."""
    B, C, H, W = latent.shape
    pH, pW = H // patch_size, W // patch_size
    x = latent.reshape(B, C, pH, patch_size, pW, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, pH * pW, C * patch_size * patch_size)
    return x, pH, pW


def unpatchify(
    x: torch.Tensor, pH: int, pW: int, out_channels: int, patch_size: int
) -> torch.Tensor:
    """Inverse of :func:`patchify` → ``[B, out_channels, pH*p, pW*p]``."""
    B = x.shape[0]
    x = x.reshape(B, pH, pW, out_channels, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, out_channels, pH * patch_size, pW * patch_size)


def composite_te_key(caption: str, control_hash: str) -> str:
    """TE cache key for an edit run — embeddings depend on the control too.

    Same caption + different control image → distinct key (prevents the silent
    bug where every edit shares one text embedding).
    """
    return f"{caption}||ctl:{control_hash}"


def control_files_hash(paths: list[str], memo: dict[str, bytes] | None = None) -> str:
    """Stable 16-hex digest of one item's control image(s).

    Hashes file bytes (so the key survives across runs); falls back to the
    path string if a file can't be read. ``memo`` caches per-path digests.
    """
    memo = memo if memo is not None else {}
    h = hashlib.sha256()
    for p in paths:
        digest = memo.get(p)
        if digest is None:
            try:
                with open(p, "rb") as f:
                    digest = hashlib.sha256(f.read()).digest()
            except OSError:
                digest = hashlib.sha256(p.encode("utf-8")).digest()
            memo[p] = digest
        h.update(digest)
    return h.hexdigest()[:16]


class QwenImageEditTrainer(QwenImageTrainer):
    """Qwen-Image-Edit trainer — clean control patches concat + composite TE cache."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ctrl_hash_memo: dict[str, bytes] = {}
        self._processor: Any | None = None
        self._processor_resolved = False
        self._warned_no_processor = False
        self._no_processor_reason: str | None = None

    def _create_sampler(self):
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler_edit import QwenImageEditSampler
            return QwenImageEditSampler(self)
        return None

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """QwenImageTransformer forward with concatenated CLEAN control patches.

        Returns the velocity prediction for the TARGET tokens only,
        unpatchified to ``[B, C, H, W]`` (control tail dropped).
        """
        control_latents = batch.get("control_latents") or []
        if not control_latents:
            # Partial/standard batch — behave exactly like the base trainer.
            return super().forward_pass(noisy_input, timesteps, text_embeddings, batch)

        if isinstance(text_embeddings, tuple):
            enc_hs, enc_mask = text_embeddings
        else:
            enc_hs, enc_mask = text_embeddings, None

        B, C, H, W = noisy_input.shape
        patch_size = getattr(self.model.config, "patch_size", 2)

        x_target, t_pH, t_pW = patchify(noisy_input, patch_size)
        target_tokens = x_target.shape[1]
        seq = [x_target]
        inner_shapes = [(1, t_pH, t_pW)]
        for ctrl in control_latents:
            ctrl = ctrl.to(device=noisy_input.device, dtype=noisy_input.dtype)
            x_c, c_pH, c_pW = patchify(ctrl, patch_size)
            seq.append(x_c)
            inner_shapes.append((1, c_pH, c_pW))

        hidden_states = torch.cat(seq, dim=1)
        # One [target, control...] shape list per batch element (matches the
        # diffusers QwenImageEditPipeline img_shapes layout).
        img_shapes = [inner_shapes] * B
        model_timesteps = timesteps / 1000.0

        # diffusers 0.39 removed txt_seq_lens from the transformer forward —
        # encoder_hidden_states_mask alone carries the valid-token lengths.
        output = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=enc_hs,
            encoder_hidden_states_mask=enc_mask,
            timestep=model_timesteps,
            img_shapes=img_shapes,
            return_dict=False,
        )
        pred = output[0] if isinstance(output, tuple) else output
        # Loss is on the target tokens only — drop the control tail.
        pred = pred[:, :target_tokens]
        out_channels = getattr(self.model.config, "out_channels", C)
        return unpatchify(pred, t_pH, t_pW, out_channels, patch_size)

    # ── Text encoding (composite cache key + optional VL image path) ─────

    def _pre_cache_text_embeddings(self) -> None:
        """Edit runs encode lazily per (caption, control) at first use.

        The base disk pre-cache keys by plain caption (text-only); those
        entries would never be hit by the composite-keyed lookups, so we skip
        it. The TE therefore stays resident for the run (it must see the
        control image), which is acceptable for the paired-edit task.
        """
        self.logger.info("qwen_edit_te_precache_skipped",
                          reason="composite (caption, control) keys encode lazily")

    def _offload_text_encoders(self) -> None:
        """Keep the VL text encoder resident — enforcement of the
        "TE stays resident" contract documented on
        :meth:`_pre_cache_text_embeddings`. The shared base offload moves the
        TE to CPU and pops it from ``self.components`` after the (no-op)
        warmup, stranding the first composite-key cache miss mid-training on
        a CPU encoder with CUDA inputs (GPU UAT 2026-07-14: "index is on
        cuda:0 ... other tensors on cpu")."""
        self.logger.info(
            "te_offload_skipped_edit_lazy_encode",
            reason="edit runs encode (caption, control) composites lazily "
                   "all run — TE must stay resident",
        )

    def _ensure_te_on_device(self) -> None:
        """Move the (CPU-loaded) VL text encoder to the trainer device.

        With pre-caching skipped, the base pre-cache — the implicit TE→GPU
        mover for every other trainer — never runs, so the first lazy
        cache-miss encode fed CUDA input ids to a CPU encoder (GPU UAT
        2026-07-14, second qwen-edit-2511 crash). Idempotent: after the
        first move the encoder stays resident (see _offload_text_encoders).
        """
        te = getattr(getattr(self, "driver", None), "text_encoder", None)
        if te is None:
            return
        param = next(te.parameters(), None)
        if param is not None and param.device != self.device:
            te.to(self.device)

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions, keyed by (caption, control image).

        Falls back to the text-only base path when no control context is
        available (e.g. a sampler preview without a control image).
        """
        ctrl_paths = (batch or {}).get("control_paths")
        if not ctrl_paths:
            return super().encode_text(captions, dtype, batch)

        n_slots = len(ctrl_paths)
        embeds: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for i, caption in enumerate(captions):
            item_controls = [ctrl_paths[s][i] for s in range(n_slots)]
            key = composite_te_key(
                caption, control_files_hash(item_controls, self._ctrl_hash_memo),
            )
            if key not in self.text_cache:
                self._ensure_te_on_device()
                emb, mask = self._encode_text_with_control(
                    caption, item_controls[0], dtype,
                )
                self.text_cache[key] = (emb.squeeze(0).cpu(), mask.squeeze(0).cpu())
            cached_emb, cached_mask = self.text_cache[key]
            embeds.append(cached_emb.to(self.device, dtype=dtype))
            masks.append(cached_mask.to(self.device))

        return self._stack_padded(embeds, masks)

    @staticmethod
    def _stack_padded(
        embeds: list[torch.Tensor], masks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad per-item (embedding, mask) to a common length, then stack."""
        max_len = max(e.size(0) for e in embeds)
        emb = torch.stack([
            torch.cat([e, e.new_zeros(max_len - e.size(0), e.size(1))]) for e in embeds
        ])
        msk = torch.stack([
            torch.cat([m, m.new_zeros(max_len - m.size(0))]) for m in masks
        ])
        return emb, msk

    def _ensure_processor(self):
        """Best-effort lazy Qwen2VL processor (for the VL image path).

        Resolves once from the model repo; returns ``None`` (→ text-only
        fallback) when the processor can't be built (e.g. weights absent in a
        test/CI environment). Failure never raises.
        """
        if self._processor_resolved:
            return self._processor
        self._processor_resolved = True
        # Prefer a processor the loader may have already wired.
        self._processor = getattr(self, "processor", None)
        if self._processor is None:
            try:
                from transformers import AutoProcessor

                repo = ""
                comps = getattr(self.definition, "components", {}) or {}
                repo = str((comps.get("repo") or {}).get("path", ""))
                repo = repo.split(":", 1)[-1] if repo else ""
                if repo:
                    self._processor = AutoProcessor.from_pretrained(
                        repo, subfolder="processor",
                    )
                else:
                    self._no_processor_reason = (
                        "no processor repo configured in the definition"
                    )
            except Exception as exc:  # noqa: BLE001 — env-gated, never fatal
                # Record the reason; the single user-visible warning fires at the
                # fallback site (_encode_text_with_control) so it is emitted once
                # per run regardless of which None-path we took here.
                self._no_processor_reason = f"processor load failed: {exc}"
                self._processor = None
        return self._processor

    def _encode_text_with_control(
        self, caption: str, control_path: str, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one (caption, control image) through Qwen2.5-VL.

        Uses the edit template + processor (so the VL encoder attends to the
        control image) when a processor is available; otherwise text-only.
        """
        processor = self._ensure_processor()
        if processor is None:
            # Silent-failure policy: the VL image path is OFF this run (the
            # control image is NOT attended by the text encoder). Warn ONCE,
            # loudly, with the reason — behavior (text-only encode) unchanged.
            if not self._warned_no_processor:
                self._warned_no_processor = True
                self.logger.warning(
                    "qwen_edit_vl_processor_fallback_text_only",
                    reason=self._no_processor_reason or "VL processor unavailable",
                    hint="control image NOT attended by the VL text encoder this "
                         "run; control still conditions the transformer via "
                         "concatenated latents",
                )
            return self._encode_text_direct([caption], dtype)

        from PIL import Image

        image = Image.open(control_path).convert("RGB")
        txt = EDIT_PROMPT_TEMPLATE.format(caption)
        model_inputs = processor(
            text=[txt], images=image, padding=True, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                pixel_values=model_inputs.get("pixel_values"),
                image_grid_thw=model_inputs.get("image_grid_thw"),
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states[-1]
        split = self._extract_masked_hidden(hidden_states, model_inputs["attention_mask"])
        split = [e[EDIT_PROMPT_DROP_IDX:] for e in split]
        mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=self.device) for e in split
        ]
        emb, msk = self._stack_padded(list(split), mask_list)
        return emb.to(dtype=dtype), msk
