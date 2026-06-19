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

import os
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

        The expanded SAMPLE prompts are warmed here too: the sampler runs after
        this TE offload, so it serves prompts from ``self.text_cache`` via
        :meth:`encode_text` — without this, sampling would hit the offloaded
        (``None``) encoder and crash with "'NoneType' object is not callable".
        """
        if not self.config.get("cache_text_embeddings", True):
            return
        if self.driver.text_encoder is None:
            return

        dtype = self._resolve_loading_dtype()
        captions = [c for c in self._build_caption_hints() if c not in self.text_cache]
        for sp in self._sample_prompt_texts():
            if sp not in self.text_cache and sp not in captions:
                captions.append(sp)
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

    def _sample_prompt_texts(self) -> list[str]:
        """Expanded sample-prompt strings to pre-cache.

        Mirrors the sampler's wildcard expansion (shared module helper) so the
        cache key matches the exact string the sampler later requests via
        :meth:`encode_text`.
        """
        from app.engine.core.sampling import expand_prompt_wildcards

        texts: list[str] = []
        for sp in self.config.get("sample_prompts", []) or []:
            raw = (
                sp.get("prompt", "")
                if isinstance(sp, dict)
                else getattr(sp, "prompt", "")
            )
            if raw:
                expanded = expand_prompt_wildcards(raw, self.config)
                if expanded not in texts:
                    texts.append(expanded)
        return texts

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

    # ── i2v first-frame conditioning gate ───────────────────────────────

    def _attach_conditioning(self, batch: dict, latents: object) -> None:
        """Per-step i2v gate. Sets driver._i2v_active for this step.

        i2v is active when video_mode=='i2v' AND a Bernoulli draw with
        first_frame_conditioning_probability succeeds (the LTX recipe trains a
        mix of conditioned + unconditioned steps).  Video-only (no audio) for now.
        """
        import random as _random

        active = False
        if str(self.config.get("video_mode", "t2v")).lower() == "i2v":
            p = float(self.config.get("first_frame_conditioning_probability", 0.5))
            active = _random.random() < p
        # Audio i2v not handled yet — only condition video-only steps.
        if batch.get("audio_clean") is not None:
            active = False
        self.driver._i2v_active = active

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

        i2v first-frame mask: when i2v is active and the batch carries no audio
        (video-only step), the first ``tokens_per_frame`` tokens in ``pred`` /
        ``target`` are the conditioning frame and must be excluded from the loss
        (their velocity target is trivially zero since t=0 → they were not noised).
        The audio path and non-i2v path are unchanged.
        """
        audio_pred = batch.get("audio_pred")
        audio_target = batch.get("audio_target")
        audio_mask = batch.get("audio_mask")

        # i2v loss mask: drop the conditioning frame tokens (first tpf).
        # Conditions:
        #   1. driver reports i2v active for this step
        #   2. no audio in this batch (audio_clean absent → video-only step)
        if (
            self.driver._i2v_conditioning_engaged()
            and batch.get("audio_clean") is None
        ):
            _, h, w = self.driver._latent_grid()
            tpf = h * w
            pred = pred[:, tpf:]
            target = target[:, tpf:]

        loss = self.driver.compute_loss(
            pred,
            target,
            batch,
            audio_pred=audio_pred,
            audio_target=audio_target,
            audio_mask=audio_mask,
        )
        return loss / grad_accum

    # ── Audio latent cache (data path) ────────────────────────────────────

    @staticmethod
    def _audio_cache_dir(video_cache_dir: str) -> str:
        """Audio latents live in an ``audio/`` sibling of the video latent dir.

        Keeping them in a SEPARATE file (same content-addressed, trim-aware
        filename) means enabling audio never disturbs an existing video-only
        cache, and audio coverage is checked independently of video coverage.
        """
        return os.path.join(video_cache_dir, "audio")

    def _pre_cache_aux(self) -> None:
        """Pre-cache LTX-2 audio latents alongside the video latents.

        Runs (via ``run_trainer``) right after ``_pre_cache_latents`` while the
        audio VAE is still resident — the next step offloads VAEs. For each
        video clip it decodes the audio of the SAME trim window, builds the
        clean (packed + normalized) audio latent, and caches it keyed identically
        to the video latent. Clips with no audio stream are skipped (their loss
        is masked to zero at train time). No-op unless this run trains audio.

        This is what makes ``batch["audio_clean"]`` available — without it the
        driver's joint forward always falls back to the video-only branch and
        the audio LoRA modules receive zero gradient.
        """
        if not getattr(self.driver, "train_audio", False):
            return
        if getattr(self.driver, "audio_vae", None) is None:
            return
        if not self.config.get("cache_latents", True):
            return

        from safetensors.torch import save_file

        from app.engine.core.pipeline.pipeline_data import video_trim_extra_key

        from .audio_io import load_audio_waveform

        sr = int(self.driver.audio_sampling_rate)
        default_fps = float(getattr(self.driver, "frame_rate", 24.0) or 24.0)
        encoded = skipped = absent = 0

        for item in self.inventory:
            if not item.get("is_video"):
                continue  # stills carry no temporal audio → masked at train time
            extra_key = video_trim_extra_key(item)
            adir = self._audio_cache_dir(item["cache_dir"])
            fname = self.latent_manager.latent_filename(
                item["id"], item["path"], extra_key
            )
            path = os.path.join(adir, fname)
            if os.path.exists(path):
                skipped += 1
                continue

            frames = int(item.get("target_frames", 1) or 1)
            fps = float(item.get("target_fps") or default_fps)
            duration = frames / fps if fps > 0 else 0.0
            wav = load_audio_waveform(
                item["path"],
                trim_start_s=float(item.get("trim_start_s") or 0.0),
                duration_s=duration,
                target_sr=sr,
            )
            if wav is None:
                absent += 1
                continue

            waveform, wav_sr = wav
            with torch.no_grad():
                # encode_audio_clean wants [B, C, N]; waveform is [C=1, N].
                latent = self.driver.encode_audio_clean(
                    waveform.unsqueeze(0).to(self.device), wav_sr,
                )  # [1, L, 128]
            os.makedirs(adir, exist_ok=True)
            save_file({"audio_latents": latent[0].detach().cpu()}, path)
            encoded += 1

        self.logger.info(
            "ltx2_audio_precache_done",
            encoded=encoded,
            skipped=skipped,
            absent=absent,
        )

        # The audio VAE is unused after this point at train time — offload it so
        # it doesn't sit in VRAM during UNet training (the base _offload_vae only
        # handles the video VAE).
        audio_vae = getattr(self.driver, "audio_vae", None)
        if audio_vae is not None and hasattr(audio_vae, "to"):
            audio_vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def build_batch_extra(self, items: list[dict]) -> dict[str, Any]:
        """Attach cached audio latents + presence mask to the training batch.

        Returns ``{"audio_clean": [B, L, 128], "audio_mask": [B]}`` so the
        driver's joint forward noises the audio stream and the masked audio loss
        flows. Items with no cached audio (no audio stream, or a still image) get
        a zero latent shaped like a present sibling and ``mask = 0`` — so they
        neither contribute to nor crash the collated batch. When NOTHING in the
        batch carries audio, returns ``{}`` so the forward stays video-only.

        Defensive on shape: a present latent whose ``L`` differs from the batch
        reference (should not happen within a temporal bucket) is treated as
        absent rather than crashing ``torch.stack``.
        """
        if not getattr(self.driver, "train_audio", False):
            return {}
        if not self.config.get("cache_latents", True):
            return {}

        from safetensors.torch import load_file

        from app.engine.core.pipeline.pipeline_data import video_trim_extra_key

        latents: list[torch.Tensor | None] = []
        for item in items:
            lat: torch.Tensor | None = None
            if item.get("is_video"):
                extra_key = video_trim_extra_key(item)
                adir = self._audio_cache_dir(item["cache_dir"])
                fname = self.latent_manager.latent_filename(
                    item["id"], item["path"], extra_key
                )
                path = os.path.join(adir, fname)
                if os.path.exists(path):
                    try:
                        lat = load_file(path)["audio_latents"]
                    except (OSError, KeyError) as e:
                        self.logger.warning(
                            "ltx2_audio_cache_load_failed", path=path, error=str(e)
                        )
            latents.append(lat)

        ref = next((latent for latent in latents if latent is not None), None)
        if ref is None:
            return {}  # no audio anywhere in this batch → video-only forward

        filled: list[torch.Tensor] = []
        present: list[float] = []
        for latent in latents:
            if latent is not None and latent.shape == ref.shape:
                filled.append(latent)
                present.append(1.0)
            else:
                if latent is not None:
                    self.logger.warning(
                        "ltx2_audio_latent_shape_mismatch",
                        expected=list(ref.shape),
                        actual=list(latent.shape),
                    )
                filled.append(torch.zeros_like(ref))
                present.append(0.0)

        return {
            "audio_clean": torch.stack(filled).to(self.device),
            "audio_mask": torch.tensor(present, device=self.device),
        }
