"""LTX 2.3 Trainer — family hooks for the generic training pipeline.

Implements LTX-2-specific behaviour:
- Single frozen Gemma3 text encoder → ``LTX2TextConnectors`` → video/audio emb.
- Flow matching on the ``[0, 1000]`` scale (driver ``add_noise`` override).
- 5D video latents packed via ``_pack_latents`` (patch_size / patch_size_t).
- Optional joint audio stream: when ``train_audio`` is on, the audio VAE +
  vocoder are loaded and the loss adds ``audio_weight * masked_audio_fm``.

When audio is OFF the audio components are never requested and the loss is the
plain video flow-match MSE — identical to the audio-free pipeline path.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from app.engine.core.text_encoding import TextEncoderOutput

from .driver import Ltx2Driver
from .loader import Ltx2Loader
from .saver import Ltx2Saver

logger = structlog.get_logger(__name__)


class Ltx2Trainer(GenericTrainingPipeline):
    """LTX 2.3 (joint audio + video) LoRA trainer."""

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_family(self) -> None:
        """Initialize LTX-2 loader, saver, driver, and audio gating."""
        train_audio = self._resolve_train_audio()
        self.driver = Ltx2Driver(self.definition, self.device)
        self.loader = Ltx2Loader(self.device, train_audio=train_audio)
        self.saver = Ltx2Saver()
        # Surface the resolved flag onto the driver so get_lora_targets,
        # compute_loss, etc. gate the audio sub-stream consistently even before
        # components are assigned (assign_components re-confirms it from arch).
        self.driver.train_audio = train_audio

    def _resolve_train_audio(self) -> bool:
        """Decide whether to train the audio stream for this run.

        Audio training requires BOTH: the user opted in (``train_audio`` config,
        default False) AND the model declares ``has_audio`` in its definition.
        Absent either, the run is video-only.
        """
        arch = getattr(self.definition, "architecture_params", {}) or {}
        model_has_audio = bool(arch.get("has_audio", False))
        user_wants_audio = bool(self.config.get("train_audio", False))
        return model_has_audio and user_wants_audio

    def _create_sampler(self):
        """Create an Ltx2Sampler if sampling is configured."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import Ltx2Sampler

            return Ltx2Sampler(self)
        return None

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep self.transformer in sync after PEFT/quantization wrapping."""
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

    # ── Text-embedding cache (warm before the 12B Gemma3 TE is offloaded) ──

    def _pre_cache_text_embeddings(self) -> None:
        """Warm the in-memory text-embedding cache before TE offload.

        ``run_trainer`` runs ``_pre_cache_text_embeddings`` → ``_offload_text_encoders``;
        the base pre-cache is a no-op, so without this override the 12B Gemma3
        encoder is offloaded with an EMPTY cache and the first training step has
        no way to produce text embeddings (``encode_text`` → ``None`` →
        ``video_emb`` ``None`` → crash in ``_dummy_audio_inputs``).

        Each unique caption (the exact trigger/prefix/dropout composites the
        train loop builds — see :meth:`_build_caption_hints`) is encoded once and
        the FULL ``(video embeddings, audio pooled, attention mask)`` triple is
        cached on CPU; LTX-2's joint forward consumes the audio ``pooled`` too,
        so a video-only tensor cache would not suffice once audio is enabled.
        """
        if not self.config.get("cache_text_embeddings", True):
            return
        if self.driver.text_encoder is None:
            return

        dtype = self._resolve_loading_dtype()
        captions = [c for c in self._build_caption_hints() if c not in self.text_cache]
        total = len(captions)
        if not total:
            self.logger.info("ltx2_text_cache_complete", cached=len(self.text_cache))
            return

        if getattr(self, "_log_writer", None):
            self._log_writer.status("Caching Text Embeddings (0%)")

        batch_size = 4
        with torch.no_grad():
            for i in range(0, total, batch_size):
                chunk = captions[i : i + batch_size]
                out = self.driver.encode_text(chunk, dtype)
                for j, cap in enumerate(chunk):
                    self.text_cache[cap] = self._slice_te_output(out, j)
                if getattr(self, "_log_writer", None):
                    pct = round(min(i + batch_size, total) / total * 100)
                    self._log_writer.status(f"Caching Text Embeddings ({pct}%)")

        self.logger.info(
            "ltx2_text_cache_complete",
            cached=len(self.text_cache),
            newly_encoded=total,
        )

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None
    ) -> TextEncoderOutput:
        """Reassemble a batched :class:`TextEncoderOutput` from the warm cache.

        Caching off → encode directly via the driver. A cache miss while the TE
        is still resident is encoded on the fly (and cached); a miss AFTER the
        TE has been offloaded is a hard error (the pre-cache should have covered
        every caption the train loop produces).
        """
        if not self.config.get("cache_text_embeddings", True):
            return self.driver.encode_text(captions, dtype)

        embs: list[torch.Tensor] = []
        pooleds: list[torch.Tensor | None] = []
        masks: list[torch.Tensor | None] = []
        for cap in captions:
            entry = self.text_cache.get(cap)
            if entry is None:
                if self.driver.text_encoder is None:
                    raise RuntimeError(
                        "Text encoder offloaded and caption not pre-cached: "
                        f"{cap[:60]!r}"
                    )
                out = self.driver.encode_text([cap], dtype)
                entry = self._slice_te_output(out, 0)
                self.text_cache[cap] = entry
            emb_c, pooled_c, mask_c = entry
            embs.append(emb_c)
            pooleds.append(pooled_c)
            masks.append(mask_c)

        embeddings = torch.cat(
            [e.to(self.device, dtype=dtype) for e in embs], dim=0
        )
        pooled = None
        if all(p is not None for p in pooleds):
            pooled = torch.cat(
                [p.to(self.device, dtype=dtype) for p in pooleds], dim=0
            )
        mask = None
        if all(m is not None for m in masks):
            mask = torch.cat([m.to(self.device) for m in masks], dim=0)
        return TextEncoderOutput(
            embeddings=embeddings, attention_mask=mask, pooled=pooled
        )

    def _offload_text_encoders(self) -> None:
        """Offload the Gemma3 encoder AND the connectors after caching.

        The connectors are a second text-encoding stage that is intentionally
        absent from ``get_text_encoders()`` (so they are never quantized/LoRA'd
        as a text encoder), which means the base offload leaves them pinned on
        the GPU after :meth:`_run_connectors` co-located them there. Push them to
        CPU in lockstep with the Gemma3 to reclaim ~3 GB of VRAM during UNet
        training. When caching is OFF the base keeps the Gemma3 resident for
        live per-step encoding — mirror that and keep the connectors too.
        """
        super()._offload_text_encoders()
        if not self.config.get("cache_text_embeddings", True):
            return
        connectors = getattr(self.driver, "connectors", None)
        if connectors is not None and hasattr(connectors, "to"):
            connectors.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _slice_te_output(
        out: Any, j: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Extract item ``j`` of a TE batch as CPU ``(emb, pooled, mask)``."""
        emb = (out.embeddings if hasattr(out, "embeddings") else out)[j : j + 1].cpu()
        pooled = getattr(out, "pooled", None)
        mask = getattr(out, "attention_mask", None)
        return (
            emb,
            pooled[j : j + 1].cpu() if pooled is not None else None,
            mask[j : j + 1].cpu() if mask is not None else None,
        )

    # ── Joint audio + video loss ─────────────────────────────────────────

    def _compute_step_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        batch: dict[str, Any],
        grad_accum: int,
    ) -> torch.Tensor:
        """Route loss through the driver's joint audio+video recipe.

        For video-only runs this is the plain video flow-match MSE (identical to
        the base implementation).  When ``train_audio`` is on, the audio
        prediction/target/mask are read from ``batch`` (populated by the audio
        forward path) and the driver adds ``audio_weight * masked_audio_fm``,
        sharing the SAME timestep.
        """
        audio_pred = batch.get("audio_pred")
        audio_target = batch.get("audio_target")
        audio_mask = batch.get("audio_mask")

        loss = self.driver.compute_loss(
            pred,
            target,
            batch,
            audio_pred=audio_pred,
            audio_target=audio_target,
            audio_mask=audio_mask,
        )
        return loss / grad_accum
