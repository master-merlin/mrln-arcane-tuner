"""LTX 2.3 Trainer — family hooks for the generic training pipeline.

Implements LTX-2-specific behaviour:
- Single frozen Gemma3 text encoder → ``LTX2TextConnectors`` → video/audio emb.
- Flow matching on the ``[0, 1000]`` scale, wired to the driver's
  ``add_noise`` (see the ``add_noise`` override below — the i2v frame-0-token
  pin is dead code on the real training path unless the trainer delegates).
- 5D video latents packed via ``_pack_latents`` (patch_size / patch_size_t).
- Optional joint audio stream: when ``train_audio`` is on, the audio VAE +
  vocoder are loaded and the loss adds ``audio_weight * masked_audio_fm``.

When audio is OFF the audio components are never requested and the loss is the
plain video flow-match MSE — identical to the audio-free pipeline path.
"""

from __future__ import annotations

import hashlib
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


def _audio_vae_fingerprint(audio_vae: Any) -> str:
    """Stable short id for an audio VAE's encode-relevant identity.

    Prefers the per-feature normalization stats (``latents_mean``/``latents_std``)
    the packed latents are normalized by — these change whenever the VAE does and
    are cheap to hash. Falls back to the class name when stats are absent.
    """
    if audio_vae is None:
        return "none"
    tag = type(audio_vae).__name__
    stats = []
    for attr in ("latents_mean", "latents_std"):
        val = getattr(audio_vae, attr, None)
        if val is not None:
            try:
                stats.append(
                    hashlib.sha1(
                        val.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
                    ).hexdigest()[:8]
                )
            except Exception:  # noqa: BLE001 — identity is best-effort
                stats.append(str(val))
    return tag + ("-" + "-".join(stats) if stats else "")


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
        """Warm the text-embedding cache (disk + memory) before TE offload.

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

        Disk-backed cache (P1c): mirroring the image families (``qwen_image`` /
        ``krea2``), the triple is persisted via the shared
        :class:`TextEmbeddingCache` under
        ``{ds}/.cache/{model}/{ver}/embeddings/{te_quant}/{te1,te2,te3}/`` —
        te1=video emb, te2=audio pooled, te3=attention mask — keyed on the
        caption hash. A warm run loads the whole set from disk and NEVER
        re-encodes through the 12B Gemma3 + connectors, closing the
        "re-encode every run" gap.

        Cache-key soundness (audio/i2v): the text embedding is a PURE function of
        the caption text and the (Gemma3 + connectors) identity — the connectors
        always produce BOTH the video and audio text embeddings from the caption
        alone. audio_timestep / i2v first-frame conditioning enters only LATER in
        the DiT forward (via timesteps + latents), never text encoding, so the
        caption hash (plus the ``te_quant`` path segment) fully keys the cache and
        no conditioning variant can cross-contaminate a hit.

        The expanded SAMPLE prompts + the CFG negative prompt round-trip through
        the SAME disk cache as the training captions: the sampler runs after this
        TE offload and serves prompts from ``self.text_cache`` via
        :meth:`encode_text` — without warming, sampling would hit the offloaded
        (``None``) encoder and crash with "'NoneType' object is not callable".
        """
        if not self.config.get("cache_text_embeddings", True):
            return
        if self.driver.text_encoder is None:
            return

        from app.engine.components.text_embeddings import TextEmbeddingCache

        # Disk cache dirs (te_quant path segment keeps FP8/bf16 embeddings apart).
        te_cache_dirs = self._resolve_te_cache_dirs()
        te_quant = self.config.get("te_quantization", "none")

        def _slot_dir(slot: str) -> str:
            return (
                os.path.join(te_cache_dirs[0], "embeddings", te_quant, slot)
                if te_cache_dirs
                else ""
            )

        te1_dir, te2_dir, te3_dir = _slot_dir("te1"), _slot_dir("te2"), _slot_dir("te3")

        dtype = self._resolve_loading_dtype()

        # ── Full ordered work set: training captions, expanded sample prompts,
        # then the CFG negative (default "" = standard unconditional). ──
        work: list[tuple[str, str]] = []
        seen: set[str] = set()

        def _add(cap: str, hint: str) -> None:
            if cap in self.text_cache or cap in seen:
                return
            seen.add(cap)
            work.append((cap, hint))

        # NOTE: _build_caption_hints() already expands DETERMINISTIC sample
        # prompts (no wildcards) into its captions; the explicit
        # _sample_prompt_texts() loop below exists to additionally catch
        # RANDOM-wildcard expansions, which are re-rolled independently and
        # can differ from what _build_caption_hints saw. Any overlap between
        # the two is a no-op: _add's `seen`/text_cache guard dedupes it.
        for cap, hint in self._build_caption_hints().items():
            _add(cap, hint)
        sample_texts = self._sample_prompt_texts()
        for sp in sample_texts:
            _add(sp, "")
        if sample_texts:
            _add(str(self.config.get("sample_negative_prompt", "") or ""), "")

        # ── Phase 1: load the triple from disk (te1 presence gates the hit) ──
        #
        # te1 is written LAST during the save (see Phase 2 below) so its
        # presence acts as a commit marker for the whole triple. If te1 hits
        # but te2/te3 (when their dirs are configured) are missing, an earlier
        # run crashed mid-write and left a PARTIAL triple on disk (a pre-fix
        # build wrote te1 first, so this also self-heals caches poisoned
        # before this fix). Treat that as a MISS — re-encode + re-save all
        # three — instead of silently caching (emb, None, None) forever.
        disk_loaded = 0
        need_encode: list[tuple[str, str]] = []
        for cap, hint in work:
            if te1_dir:
                emb = TextEmbeddingCache.load(cap, te1_dir, hint)
                if emb is not None:
                    pooled = (
                        TextEmbeddingCache.load(cap, te2_dir, hint) if te2_dir else None
                    )
                    mask = (
                        TextEmbeddingCache.load(cap, te3_dir, hint) if te3_dir else None
                    )
                    partial = (te2_dir and pooled is None) or (
                        te3_dir and mask is None
                    )
                    if not partial:
                        self.text_cache[cap] = (emb, pooled, mask)
                        disk_loaded += 1
                        continue
                    self.logger.warning(
                        "ltx2_partial_triple_treated_as_miss",
                        caption_hash=hashlib.sha256(cap.encode("utf-8")).hexdigest()[:16],
                        hint=hint,
                    )
            need_encode.append((cap, hint))

        if not need_encode:
            if getattr(self, "_log_writer", None):
                self._log_writer.status("TE Cache Loaded from Disk")
            self.logger.info(
                "ltx2_text_cache_complete",
                cached=len(self.text_cache),
                from_disk=disk_loaded,
                source="disk",
            )
            return

        # ── Phase 2: encode the misses (batched) + persist the triple ──
        if getattr(self, "_log_writer", None):
            self._log_writer.status("Caching Text Embeddings (0%)")

        total = len(need_encode)
        batch_size = 4
        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch_items = need_encode[i : i + batch_size]
                chunk = [cap for cap, _ in batch_items]
                out = self.driver.encode_text(chunk, dtype)
                for j, (cap, hint) in enumerate(batch_items):
                    emb, pooled, mask = self._slice_te_output(out, j)
                    self.text_cache[cap] = (emb, pooled, mask)
                    # Save order matters: te3/te2 first, te1 LAST. te1's
                    # presence is the Phase-1 disk-hit gate, so writing it
                    # last makes it the commit marker for the triple — a
                    # crash mid-write leaves te1 absent (a clean miss on the
                    # next run) instead of poisoning the cache with a
                    # partial (emb, None, None) hit.
                    if te3_dir and mask is not None:
                        TextEmbeddingCache.save(cap, mask, te3_dir, hint)
                    if te2_dir and pooled is not None:
                        TextEmbeddingCache.save(cap, pooled, te2_dir, hint)
                    if te1_dir:
                        TextEmbeddingCache.save(cap, emb, te1_dir, hint)
                if getattr(self, "_log_writer", None):
                    pct = round(min(i + batch_size, total) / total * 100)
                    self._log_writer.status(f"Caching Text Embeddings ({pct}%)")

        self.logger.info(
            "ltx2_text_cache_complete",
            cached=len(self.text_cache),
            from_disk=disk_loaded,
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

    # ── Convention delegation (I2V frame-0 pin — real-path wiring) ───────

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate to the driver's flow-match lerp + i2v frame-0-token pin.

        The base ``PipelineBaseMixin.add_noise`` hardcodes
        ``self.noise_interpolation.add_noise`` (:class:`NoiseInterpolation`,
        mode ``"linear"``) — a component with NO knowledge of LTX-2's i2v
        conditioning-frame pin. Left un-overridden, the REAL training loop
        (``pipeline_train.py``'s ``self.add_noise(...)`` family-hook call for
        the VIDEO stream) resolves to that generic component and noises the
        conditioning frame's tokens even when i2v is engaged — directly
        contradicting ``_compute_step_loss``'s frame-0-token exclusion below,
        which assumes those tokens stay clean (kandinsky5/boogu_image
        convention-delegation precedent).

        Note ``Ltx2Driver.add_noise`` is NOT dead code in general — the
        driver's OWN ``forward_pass`` calls ``self.add_noise(...)`` directly
        for the AUDIO stream (a driver-internal call, unaffected by this
        trainer-level MRO gap). Only the VIDEO stream's real-path dispatch
        was broken.

        SAFE for T2V / i2v-inactive steps: ``Ltx2Driver.add_noise``'s
        un-engaged math (``frac = timesteps / _FLOWMATCH_SCALE; frac*noise +
        (1-frac)*latents``) is algebraically identical to
        ``NoiseInterpolation._linear``'s ``(1-t)*latents + t*noise`` — same
        terms, commutative sum, same ``/1000`` scale — so this delegation
        changes ZERO non-i2v training behavior (pinned by
        ``test_ltx2_addnoise_wiring.py``'s bit-identity test).
        """
        return self.driver.add_noise(latents, noise, timesteps)

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

    def _audio_cache_dir(self, video_cache_dir: str) -> str:
        """Audio latents live in an ``audio/<version>/`` sibling of the video dir.

        Keeping them in a SEPARATE file (same content-addressed, trim-aware
        filename) means enabling audio never disturbs an existing video-only
        cache, and audio coverage is checked independently of video coverage.

        The ``<version>`` segment fingerprints the params the encode depends on
        (mel transform + audio-VAE identity) so a param change forces a
        re-encode instead of silently serving stale audio latents.
        """
        return os.path.join(video_cache_dir, "audio", self._audio_cache_version())

    def _audio_cache_version(self) -> str:
        """Short fingerprint of everything the audio-latent encode depends on.

        Covers the log-mel transform (sample rate, n_fft, hop, mel bins) and
        the audio VAE's identity (its per-feature normalization stats, which
        change whenever the VAE does). A mismatch here previously exposed stale
        cached latents when mel/audio-VAE params changed.
        """
        from .audio_mel import DEFAULT_MEL_BINS, DEFAULT_MEL_HOP, DEFAULT_N_FFT

        driver = getattr(self, "driver", None)
        parts = [
            str(int(getattr(driver, "audio_sampling_rate", 16000))),
            str(DEFAULT_N_FFT),
            str(DEFAULT_MEL_HOP),
            str(DEFAULT_MEL_BINS),
            _audio_vae_fingerprint(getattr(driver, "audio_vae", None)),
        ]
        return "v" + hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]

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
        encoded = skipped = absent = failed = 0

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
            try:
                with torch.no_grad():
                    # encode_audio_clean wants [B, C, N]; waveform is [C=1, N].
                    latent = self.driver.encode_audio_clean(
                        waveform.unsqueeze(0).to(self.device), wav_sr,
                    )  # [1, L, 128]
                os.makedirs(adir, exist_ok=True)
                save_file({"audio_latents": latent[0].detach().cpu()}, path)
                encoded += 1
            except Exception as e:  # noqa: BLE001 — a bad clip must not kill the run
                # Graceful path: leave this clip's audio uncached — build_batch_extra
                # treats it as absent (audio_mask=0). That degradation (the clip's
                # audio stream silently drops out of training) must be VISIBLE.
                failed += 1
                self.logger.warning(
                    "ltx2_audio_encode_failed",
                    path=item.get("path"),
                    error=str(e),
                )

        if failed:
            self.logger.warning(
                "ltx2_audio_precache_incomplete",
                failed=failed,
                encoded=encoded,
                skipped=skipped,
                absent=absent,
                hint="clips whose audio failed to encode train with NO audio "
                     "(audio_mask=0 → zero audio-loss contribution)",
            )
        self.logger.info(
            "ltx2_audio_precache_done",
            encoded=encoded,
            skipped=skipped,
            absent=absent,
            failed=failed,
        )

        if failed and not encoded and not skipped:
            # TOTAL failure: every clip whose audio we attempted to encode
            # failed, NONE succeeded, and nothing was already cached. Audio
            # training is ON, yet the run would carry ZERO audio latents (every
            # clip audio_mask=0 → audio LoRA gets no gradient) — a misconfigured
            # run. Escalate loudly instead of silently training video-only.
            # With skipped>0 (resume with cached clips) the run still carries
            # real latents for the cached majority → partial-degrade path above.
            raise RuntimeError(
                f"ltx2 audio precache produced ZERO audio latents: all "
                f"{failed} clip(s) failed to encode "
                f"(ltx2_audio_precache_incomplete). Audio-on training with no "
                f"audio latents is misconfigured — refusing to proceed."
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
