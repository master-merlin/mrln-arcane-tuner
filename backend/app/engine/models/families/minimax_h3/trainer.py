"""MiniMax-H3 Trainer — the family's training path (plan rows in brackets).

Owns: the inverted flow-match delegations (row 1.2), the text-embedding
lifecycle — pre-cache, unconditional encoder release, cache-serving
``encode_text``, deferred DiT materialisation (row 2.1) — the latent-cache
fingerprint seam (2.3), the batch-extra delegation (2.4), the real
setup + audio-latent lifecycle (2.5), and the joint forward + step loss
(2.6: the audio stream noised on its own clock, ``driver.compute_loss``
scaled by ``1/grad_accum``, the three numbers on the ``h3_step_loss`` line).

``_setup_family`` is the first family hook the pipeline calls (via
``setup()``); ``self.driver = MiniMaxH3Driver(...)`` is written literally
here because ``tests/engine/test_hook_wiring_meta.py::_driver_for_trainer``
resolves a family's driver by regex-searching the trainer's MRO source for
exactly this assignment shape.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from typing import Any

import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from app.engine.core.text_encoding import TextEncoderOutput

from .driver import MiniMaxH3Driver
from .settings import H3EffectiveSettings


def _host_peak_bytes() -> int | None:
    """Peak host RSS of this process (the TE loading peak the plan records);
    ``psutil`` is a pinned dependency, but a missing attribute (non-Windows
    lacks ``peak_wset``) degrades to the current RSS, never to a raise."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", None) or info.rss)
    except Exception:  # noqa: BLE001 — a metric, never a failure
        return None


class MiniMaxH3Trainer(GenericTrainingPipeline):
    """MiniMax-H3 LoRA trainer (PR1)."""

    settings: H3EffectiveSettings | None = None

    # H3 has ONE clock and the definition states it (`video.frame_rate`): the
    # rotary positions (`packing.py`) and the audio fit (`driver.build_batch_extra`)
    # are laid out on it, so ingestion resamples every clip to it
    # (`PipelineDataMixin._ingest_video_at_native_fps`) and `_setup_family`
    # refuses a setting that asks for another rate.
    _ingest_video_at_native_fps = True

    # ── Explicit delegations of the driver's CLOBBER hooks (plan row 1.2,
    # ordering rule 1). The base pipeline WOULD auto-delegate these, but an
    # explicit method keeps the family out of the reviewed auto-delegation
    # allowlist (`test_autodelegated_family_hook_set_is_exactly_expected`)
    # and makes the seam greppable.

    def add_noise(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        return self.driver.add_noise(latents, noise, timesteps)

    def compute_target(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        return self.driver.compute_target(latents, noise, timesteps)

    def sample_timesteps(
        self, batch_size: int, latents: torch.Tensor | None = None
    ) -> torch.Tensor:
        max_steps = getattr(self, "max_train_steps", 1)
        progress = getattr(self, "global_step", 0) / max(max_steps, 1)
        return self.driver.sample_timesteps(
            batch_size, self.device, self.config, latents=latents, progress=progress
        )

    def build_batch_extra(self, items: list[dict]) -> dict[str, Any]:
        # CLOBBER hook (plan row 2.4, ordering rule 1): the driver stacks the
        # items' clean audio latents; this override (row 2.5) loads them from
        # the audio cache into a per-step COPY of each item (the inventory
        # never retains tensors) before the explicit delegation. `train_audio`
        # does not gate it (row 3.2): the rows stay packed, audio off is a
        # zero loss weight. The cache is the ONLY source (`_setup_family`
        # refuses `cache_latents=False`).
        items = [
            {**item, "audio_latents": lat} if (lat := self._load_cached_audio(item)) is not None else item
            for item in items
        ]
        return self.driver.build_batch_extra(items)

    # ── The joint forward + step loss (plan row 2.6, ASTRA MAJOR-6) ────────
    #
    # `forward_pass` is NOT a CLOBBER hook (the base calls the driver directly,
    # `pipeline_base.py:189`), so this override is behaviour, not a delegation:
    # the audio stream is noised HERE, on its own clock (`t_a` derived from the
    # batch's `t_v` through the dual-shift schedule — one `u`, two clocks), and
    # its noise/target are stashed on the batch for the step loss — the
    # training loop (`pipeline_train.py:547-584`) only knows the video tensors.

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        audio_clean = batch.get("audio_clean")
        # Whenever audio rows are packed they are noised on the audio clock —
        # with `train_audio` off too (row 3.2): the DiT always sees x_t rows,
        # the loss just weights them 0.
        if audio_clean is not None:
            audio_clean = audio_clean.to(device=noisy_input.device)
            audio_noise = torch.randn_like(audio_clean)
            t_a = self.driver.audio_timestep(timesteps.to(device=noisy_input.device, dtype=torch.float32))
            batch["audio_noise"] = audio_noise
            batch["audio_noisy"] = self.driver.add_noise(audio_clean, audio_noise, t_a)
            batch["audio_target"] = self.driver.compute_target(audio_clean, audio_noise, t_a)
        # Row 3.1's uncond arm: the empty-prompt row comes from the text cache
        # (the caption-dropout entry "" is always pre-cached; the encoder is
        # released before the DiT loads, so a miss raises by name).
        if self.settings is not None and float(self.settings.cfg_augment_scale) > 1.0:
            batch["text_embeddings_uncond"] = self.encode_text([""], noisy_input.dtype)
        video_pred, audio_pred = self.driver.forward_pass(noisy_input, timesteps, text_embeddings, batch)
        batch["audio_pred"] = audio_pred
        return video_pred, audio_pred

    def _compute_step_loss(
        self,
        pred: Any,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        batch: dict[str, Any],
        grad_accum: int,
    ) -> torch.Tensor:
        """``driver.compute_loss`` on the ``(video, audio)`` pair → the SCALAR
        ``loss / grad_accum`` the loop backpropagates (`pipeline_train.py:578,603`);
        the three UNSCALED numbers go to the per-step ``h3_step_loss`` line."""
        video_pred, audio_pred = pred if isinstance(pred, tuple) else (pred, batch.get("audio_pred"))
        out = self.driver.compute_loss(
            video_pred,
            target,
            batch,
            audio_pred=audio_pred,
            audio_target=batch.get("audio_target"),
            audio_mask=batch.get("audio_mask"),
        )
        self.last_step_losses = out
        self.logger.info(
            "h3_step_loss",
            loss=float(out.loss.detach()),
            loss_video=float(out.loss_video.detach()),
            loss_audio=float(out.loss_audio.detach()),
        )
        return out.loss / grad_accum

    # ── Sampling (plan row 2.10; round 13 MAJOR 13.01, DECISION-67 (a)) ──
    #
    # Production caller: `pipeline_optimization.py` ends its setup with
    # `self.sampler = self._create_sampler()`; the base returns None, which
    # makes `pipeline_train.py` never sample. 28 families override this —
    # without it an H3 run trains blind.

    def _create_sampler(self):
        if int(self.config.get("sample_every_n_steps", 0) or 0) > 0:
            from .sampler import MiniMaxH3Sampler

            return MiniMaxH3Sampler(self)
        return None

    # ── Text-embedding lifecycle (plan row 2.1; DECISION-68 (a)) ──────────
    #
    # Production caller: `run_trainer.py:159` runs `_pre_cache_text_embeddings`
    # (base: NO-OP) → `_offload_text_encoders`; the base `encode_text`
    # (`pipeline_base.py`) returns None once the driver reports no encoders.
    # H3's 63 GB Qwen3-VL and 62 GB DiT never coexist, so the encoder is
    # ALWAYS released after warming and every caption the run will ever ask
    # for — training composites, the dropout/CFG empty prompt, the expanded
    # sample prompts — is served from this cache. A miss after the release is
    # a hard error, never a silent reload.

    def _te_cache_dirs(self) -> tuple[str, str]:
        """``(te1, te2)`` disk dirs — te1 = embeddings, te2 = attention mask.
        The path carries the te_quant segment AND the driver's cache scope
        (definition, tap index, tokenizer fingerprint) so a different tap or
        vocabulary can never hit a stale file."""
        dirs = self._resolve_te_cache_dirs()
        if not dirs:
            return "", ""
        te_quant = self.config.get("te_quantization", "none")
        base = os.path.join(dirs[0], "embeddings", te_quant, self.driver.te_cache_scope())
        return os.path.join(base, "te1"), os.path.join(base, "te2")

    def _sample_prompt_texts(self) -> list[str]:
        """The EXACT expanded strings the sampler will request (ltx2 `:323`)."""
        from app.engine.core.sampling import expand_prompt_wildcards

        texts: list[str] = []
        for sp in self.config.get("sample_prompts", []) or []:
            raw = sp.get("prompt", "") if isinstance(sp, dict) else getattr(sp, "prompt", "")
            if raw:
                expanded = expand_prompt_wildcards(raw, self.config)
                if expanded not in texts:
                    texts.append(expanded)
        return texts

    def _pre_cache_text_embeddings(self) -> None:
        """Warm the disk + memory cache for every caption while the encoder
        is resident; entries are ``(emb [L, D] cpu, mask [L] long)`` trimmed
        to the caption's true length."""
        if not self.config.get("cache_text_embeddings", True):
            raise ValueError(
                "minimax_h3 requires cache_text_embeddings=True: the 63 GB "
                "Qwen3-VL encoder cannot stay resident beside the 62 GB DiT "
                "for live per-step encoding"
            )
        if self.driver.text_encoder is None:
            raise RuntimeError(
                "minimax_h3 _pre_cache_text_embeddings: the text encoder is "
                "not loaded — nothing can be cached"
            )

        from app.engine.components.text_embeddings import TextEmbeddingCache

        te1_dir, te2_dir = self._te_cache_dirs()
        dtype = self.driver.resolve_loading_dtype()

        work: list[tuple[str, str]] = []
        seen: set[str] = set()

        def _add(cap: str, hint: str) -> None:
            if cap in self.text_cache or cap in seen:
                return
            seen.add(cap)
            work.append((cap, hint))

        for cap, hint in self._build_caption_hints().items():
            _add(cap, hint)
        sample_texts = self._sample_prompt_texts()
        for sp in sample_texts:
            _add(sp, "sample")
        if sample_texts:
            _add(str(self.config.get("sample_negative_prompt", "") or ""), "negative")

        # Phase 1: disk (te1 presence gates the hit; a partial pair is a miss).
        disk_loaded = 0
        need_encode: list[tuple[str, str]] = []
        for cap, hint in work:
            if te1_dir:
                key = self.driver.te_cache_key(cap)
                emb = TextEmbeddingCache.load(key, te1_dir, hint)
                mask = TextEmbeddingCache.load(key, te2_dir, hint) if emb is not None else None
                if emb is not None and mask is not None:
                    self.text_cache[cap] = (emb, mask)
                    disk_loaded += 1
                    continue
            need_encode.append((cap, hint))

        # Phase 2: encode the misses one caption at a time (the driver encodes
        # per caption anyway — no cross-batch padding to trim) and persist,
        # mask FIRST so te1's presence is the commit marker.
        with torch.no_grad():
            for i, (cap, hint) in enumerate(need_encode):
                out = self.driver.encode_text([cap], dtype)
                n = int(out.attention_mask[0].sum())
                emb_cpu = out.embeddings[0, :n].detach().to("cpu")
                mask_cpu = out.attention_mask[0, :n].detach().to("cpu")
                self.text_cache[cap] = (emb_cpu, mask_cpu)
                if te1_dir:
                    key = self.driver.te_cache_key(cap)
                    TextEmbeddingCache.save(key, mask_cpu, te2_dir, hint)
                    TextEmbeddingCache.save(key, emb_cpu, te1_dir, hint)
                if getattr(self, "_log_writer", None):
                    pct = round((i + 1) / len(need_encode) * 100)
                    self._log_writer.status(f"Caching Text Embeddings ({pct}%)")

        self.logger.info(
            "minimax_h3_text_cache_complete",
            cached=len(self.text_cache),
            from_disk=disk_loaded,
            newly_encoded=len(need_encode),
        )

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None
    ) -> TextEncoderOutput:
        """Serve from the warm cache; a miss while the encoder is resident is
        encoded and cached, a miss after the release raises naming the caption."""
        entries = []
        for cap in captions:
            entry = self.text_cache.get(cap)
            if entry is None:
                if not self._get_text_encoders():
                    raise RuntimeError(
                        "minimax_h3 encode_text: text encoder released and caption "
                        f"not pre-cached: {cap[:80]!r}"
                    )
                out = self.driver.encode_text([cap], dtype)
                n = int(out.attention_mask[0].sum())
                entry = (out.embeddings[0, :n].detach().to("cpu"), out.attention_mask[0, :n].detach().to("cpu"))
                self.text_cache[cap] = entry
            entries.append(entry)

        max_len = max(int(m.shape[0]) for _, m in entries)
        dim = entries[0][0].shape[-1]
        emb = torch.zeros((len(entries), max_len, dim), dtype=dtype, device=self.device)
        mask = torch.zeros((len(entries), max_len), dtype=torch.long, device=self.device)
        for i, (e, m) in enumerate(entries):
            n = int(m.shape[0])
            emb[i, :n] = e.to(self.device, dtype=dtype)
            mask[i, :n] = m.to(self.device)
        return TextEncoderOutput(embeddings=emb, attention_mask=mask)

    def _offload_text_encoders(self) -> None:
        """Release the encoder UNCONDITIONALLY (the base releases the driver's
        reference only under ``unload_text_encoder``; a CPU-offloaded 63 GB
        encoder is still 63 GB of host RAM beside the DiT) and record what the
        plan asks for: weight bytes + the loading peaks.

        The base is NOT called and the encoder is NEVER moved to the host
        first: the module is dropped right here, so a device→host copy of the
        whole encoder buys nothing, and that copy crashed the process natively
        (access violation inside ``Module.to("cpu")``, no Python exception).
        VRAM is freed by dropping EVERY reference — trainer alias, component
        dict, driver attribute + dict — before ``empty_cache()``; pinned by
        ``test_release_drops_every_reference_to_the_encoder``."""
        if self._te_unloaded:
            return
        weight_bytes = self.driver.text_encoder_weight_bytes()
        self.components.pop("text_encoder", None)
        self.driver.release_text_encoders()
        if not isinstance(getattr(type(self), "text_encoder", None), property):
            self.text_encoder = None
        self._te_unloaded = True
        gc.collect()
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.empty_cache()
        # Logged AFTER the cleanup: `cuda_allocated_after_bytes` is the number
        # that says whether dropping the references actually freed the device.
        self.logger.info(
            "text_encoder released",
            weight_bytes=weight_bytes,
            host_peak_bytes=_host_peak_bytes(),
            cuda_peak_bytes=torch.cuda.max_memory_allocated() if cuda else None,
            cuda_allocated_after_bytes=torch.cuda.memory_allocated() if cuda else None,
        )

    # ── Latent-cache fingerprint — the production seam (plan row 2.3) ─────
    #
    # Production caller: `run_trainer.py` runs `prepare_data` → (TE phase) →
    # `_validate_latent_cache` → `_pre_cache_latents` → training; coverage,
    # pre-cache and `_get_batch` all read `item["cache_dir"]` from the
    # inventory the base `prepare_data` wrote with the literal variant
    # `"original"`. The fingerprint is real only because the rewrite happens
    # HERE, before every consumer; a driver helper nobody applies is dead code.

    async def prepare_data(self):
        await super().prepare_data()
        fingerprint = self.driver.latent_cache_fingerprint()
        for item in self.inventory:
            for key in ("cache_dir", "masked_cache_dir"):
                path = item.get(key)
                if path:
                    item[key] = self._fingerprinted_cache_dir(path, fingerprint)
        self.logger.info("minimax_h3_latent_cache_fingerprint", fingerprint=fingerprint)

    @staticmethod
    def _fingerprinted_cache_dir(cache_dir: str, fingerprint: str) -> str:
        """`.../latents/<res>/<variant>` → `.../latents/<res>/<variant>-h3<fp>`:
        INSIDE the last segment, same depth — `_resolve_te_cache_dirs` walks a
        fixed number of levels up from `cache_dir`, and an extra level would
        land its dataset-root lookup one directory too deep."""
        head, tail = os.path.split(cache_dir.rstrip("/\\"))
        return os.path.join(head, f"{tail}-h3{fingerprint}")

    def _materialise_transformer(self) -> None:
        """Load the deferred DiT — only once the encoder is gone."""
        self.driver.assert_text_encoder_released()
        model = self.loader.load_transformer(
            self.definition, self.driver.resolve_loading_dtype(), initial_device="cpu"
        )
        self.components["transformer"] = model
        self.components["unet"] = model  # the base pipeline's primary key
        self.driver.assign_components(self.components)
        self.transformer = model
        self.logger.info("transformer loaded", subfolder=self.loader._transformer_spec(self.definition).subfolder)

    async def prepare_for_training(self):
        # Phase B starts by freezing/wrapping the primary model, so the
        # deferred DiT must exist here; every caching phase before this ran
        # without it in host RAM.
        if self.components.get("transformer") is None:
            self._materialise_transformer()
        await super().prepare_for_training()

    # ── Setup + lifecycle (plan row 2.5) ───────────────────────────────────

    def _setup_family(self) -> None:
        """Refuse what this release cannot train FIRST, then resolve the ONE
        settings object, build the driver (the assignment shape
        `test_hook_wiring_meta._driver_for_trainer` greps for) and the loader
        with the DiT deferred out of Phase A (row 2.1)."""
        coverage = str(self.config.get("temporal_coverage", "first") or "first")
        if coverage == "sliding":
            # PR1 bound (§ Scope): the sliding window assumes (F-1)/t+1 latent
            # frames per window; H3 chunks 17n+5 pixel frames -> 5n+2 latents.
            raise ValueError(
                "minimax_h3: temporal_coverage='sliding' is not supported in this "
                "release (its latent window assumes (F-1)/t+1 frames; H3 chunks "
                "17n+5 -> 5n+2) — use 'first' or 'tiled'"
            )
        self._refuse_a_second_clock()
        self._refuse_uncached_latents()
        from .loader import MiniMaxH3Loader
        from .settings import resolve_h3_settings

        self.settings = resolve_h3_settings(self.definition, self.config)
        self.driver = MiniMaxH3Driver(self.definition, self.device)
        self.driver.apply_settings(self.settings)
        self.loader = MiniMaxH3Loader(self.device, defer_transformer=True)
        # The step-0 banner (row 3.2): one greppable line, every setting with
        # its source; the seed is the run seed `_apply_run_seed` reads (an
        # uninterpretable one is "unseeded" there too).
        self.logger.info(self.driver.step0_banner(u_seed=self._banner_seed()))

    def _refuse_a_second_clock(self) -> None:
        """An explicit `target_fps`, or a `frame_stride` whose effective rate
        differs from the definition's fps, would train the video frames on one
        timeline and the soundtrack on another. Refused by name with both
        numbers — never overridden silently, never trained. Unset (0) is
        resolved to the definition's fps at ingestion and reported there."""
        from app.engine.core.pipeline.pipeline_data import _coerce_fps
        from app.engine.core.video_contract import resolve_video_profile

        clock = resolve_video_profile(self.definition).native_fps
        if not clock:
            raise ValueError(
                f"minimax_h3 {self.definition.id}: the definition states no fps "
                "(architecture_params['video.frame_rate']) — the model's clock is unknown"
            )
        raw = self.config.get("target_fps")
        target = _coerce_fps(raw)
        if target > 0.0 and abs(target - clock) > 1e-6:
            raise ValueError(
                f"minimax_h3: target_fps={raw} contradicts the model's clock of {clock:g} fps "
                "(the definition's video.frame_rate) — the soundtrack and the frames would "
                f"train on different timelines; leave target_fps at 0 or set it to {clock:g}"
            )
        stride = int(self.config.get("frame_stride", 1) or 1)
        if stride > 1:
            raise ValueError(
                f"minimax_h3: frame_stride={stride} gives an effective {clock / stride:g} fps, "
                f"the model's clock is {clock:g} fps (the definition's video.frame_rate) — "
                "the soundtrack and the frames would train on different timelines; "
                "leave frame_stride at 1"
            )

    def _refuse_uncached_latents(self) -> None:
        """The soundtrack is encoded ONCE, up front (`_pre_cache_aux`), and the
        step reads it from that cache (`build_batch_extra`); there is no
        per-step audio encode. Without the latent cache every clip would train
        as "no soundtrack" with a zero audio loss and no line saying so."""
        if not self.config.get("cache_latents", True):
            raise ValueError(
                "minimax_h3: cache_latents=False is not supported — this family encodes "
                "each clip's frames and soundtrack once, before training, and the step "
                "reads them from that cache; without it the soundtrack would silently "
                "not be trained. Set cache_latents to true"
            )

    def _banner_seed(self) -> int | None:
        seed = self.config.get("seed")
        if seed in (None, ""):
            return None
        try:
            return int(seed)
        except (TypeError, ValueError):
            return None

    def _resolve_train_audio(self) -> bool:
        """`H3EffectiveSettings.train_audio` — the one resolver (row 1.0)."""
        return bool(self.driver._require_settings("_resolve_train_audio").train_audio)

    def _resolve_loading_dtype(self) -> torch.dtype:
        """bf16 from the definition's `detected_precision` (every H3 component
        ships bf16); the driver's answer is the same and is the fallback."""
        prec = (getattr(self.definition, "detected_precision", None) or {}).get("unet")
        if isinstance(prec, str) and prec.startswith("torch."):
            dtype = getattr(torch, prec.split(".", 1)[1], None)
            if isinstance(dtype, torch.dtype):
                return dtype
        return self.driver.resolve_loading_dtype()

    def _get_primary_model(self) -> torch.nn.Module:
        # The loader keys the DiT `transformer`; the base pipeline's primary
        # key is `unet` (`_move_component_to_gpu("unet")`, the PEFT wrap, the
        # saver) — `_materialise_transformer` / `_update_primary_model` keep
        # both entries pointing at the one model the driver holds.
        return self.driver.get_primary_model()

    def _get_text_encoders(self) -> dict[str, torch.nn.Module]:
        # `{"text_encoder": Qwen3-VL}` while resident, `{}` after row 2.1's
        # release — which is what makes the cache-serving `encode_text` load-bearing.
        return self.driver.get_text_encoders()

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep every alias in sync after the PEFT wrap: the trainer's, both
        component keys, and the driver's (its forward runs THIS model)."""
        self.transformer = new_model
        self.components["unet"] = new_model
        self.components["transformer"] = new_model
        self.driver.transformer = new_model

    # ── Audio latents: the second cached modality (rows 2.2 + 2.5) ────────

    def _audio_cache_dir(self, video_cache_dir: str) -> str:
        """`audio/<version>/` sibling of the video latent dir — enabling audio
        never disturbs a video-only cache, coverage is checked independently."""
        return os.path.join(video_cache_dir, "audio", self._audio_cache_version())

    def _audio_cache_version(self) -> str:
        """Fingerprint of everything the audio encode depends on: the family
        module's version, the sample/latent rates and the audio VAE's identity
        (class + its per-channel normalisation stats)."""
        from .audio_latents import AUDIO_LATENT_VERSION

        arch = self.definition.architecture_params or {}
        audio_vae = getattr(self.driver, "audio_vae", None)
        cfg = getattr(audio_vae, "config", None)
        stats = json.dumps(
            {
                "mean": list(getattr(cfg, "latents_mean", None) or []),
                "std": list(getattr(cfg, "latents_std", None) or []),
            },
            default=str,
        )
        parts = [
            f"al{AUDIO_LATENT_VERSION}",
            str(arch.get("audio.sampling_rate", "")),
            str(arch.get("audio.latent_rate", "")),
            type(audio_vae).__name__ if audio_vae is not None else "none",
            hashlib.sha256(stats.encode("utf-8")).hexdigest()[:12],
        ]
        return "v" + hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]

    def _pre_cache_aux(self) -> None:
        """Encode every inventory clip's soundtrack through `audio_latents`
        while the audio VAE is resident (run_trainer calls this right after
        the video pre-cache, before the VAEs are offloaded). Stills carry no
        soundtrack (masked at train time); a clip without an audio stream is
        skipped the same way. Runs with `train_audio` off too (row 3.2): the
        rows are packed either way, only their loss weight is 0."""
        audio_vae = getattr(self.driver, "audio_vae", None)
        if audio_vae is None:
            self._log_audio_coverage()
            return

        from app.engine.components.audio_io import load_audio_waveform
        from app.engine.core.pipeline.pipeline_data import video_trim_extra_key
        from app.engine.utils.safe_save import safe_save_file

        from .audio_latents import encode_stereo

        arch = self.definition.architecture_params or {}
        sr = int(arch["audio.sampling_rate"])
        # The orchestrator's VAE phase moves ONLY `vae` to the card
        # (`run_trainer.py` `_move_component_to_gpu("vae")`); the audio VAE is
        # still where Phase A left it (CPU) while the waveform below goes to
        # `self.device` — GATE-1 (plan row 2.12) failed every clip on exactly
        # that. Bring it over here; it goes back to the CPU at the end.
        if hasattr(audio_vae, "to"):
            audio_vae.to(self.device)
        encoded = skipped = absent = failed = 0
        for item in self.inventory:
            if not item.get("is_video"):
                continue
            adir = self._audio_cache_dir(item["cache_dir"])
            fname = self.latent_manager.latent_filename(
                item["id"], item["path"], video_trim_extra_key(item)
            )
            path = os.path.join(adir, fname)
            if os.path.exists(path):
                skipped += 1
                continue
            frames = int(item.get("target_frames", 1) or 1)
            fps = float(item.get("target_fps") or 0.0)
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
            waveform, _sr = wav
            try:
                with torch.no_grad():
                    latent = encode_stereo(audio_vae, waveform.unsqueeze(0).to(self.device))
                os.makedirs(adir, exist_ok=True)
                safe_save_file({"audio_latents": latent[0].detach().cpu()}, path)
                encoded += 1
            except Exception as e:  # noqa: BLE001 — one bad clip must not kill the run
                failed += 1
                self.logger.warning(
                    "minimax_h3_audio_encode_failed", path=item.get("path"), error=str(e)
                )
        self.logger.info(
            "minimax_h3_audio_precache_done",
            encoded=encoded, skipped=skipped, absent=absent, failed=failed,
        )
        self._log_audio_coverage()
        if failed and not encoded and not skipped:
            raise RuntimeError(
                f"minimax_h3 audio precache produced ZERO audio latents: all {failed} "
                "clip(s) failed to encode — audio-on training with no audio latents "
                "is misconfigured, refusing to proceed"
            )
        if hasattr(audio_vae, "to"):
            audio_vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _log_audio_coverage(self) -> None:
        """ONE line per job: how many inventory items train with explicitly
        absent audio (zero rows, mask 0 — `driver.build_batch_extra`). Counted
        on what the step will find, the audio cache on disk: a clip without a
        soundtrack and a failed encode read the same way there."""
        without = sum(1 for item in self.inventory if not self._audio_cache_path(item))
        self.logger.info(
            "minimax_h3_data_summary", total_items=len(self.inventory), clips_without_audio=without
        )

    def _audio_cache_path(self, item: dict) -> str | None:
        """The item's cached audio latent file, or None when there is none."""
        if not item.get("is_video"):
            return None
        from app.engine.core.pipeline.pipeline_data import video_trim_extra_key

        path = os.path.join(
            self._audio_cache_dir(item["cache_dir"]),
            self.latent_manager.latent_filename(item["id"], item["path"], video_trim_extra_key(item)),
        )
        return path if os.path.exists(path) else None

    def _load_cached_audio(self, item: dict) -> torch.Tensor | None:
        path = self._audio_cache_path(item)
        if path is None:
            return None
        from safetensors.torch import load_file

        try:
            return load_file(path)["audio_latents"]
        except Exception as e:  # noqa: BLE001 — a corrupt file degrades to absent (mask 0)
            self.logger.warning("minimax_h3_audio_cache_load_failed", path=path, error=str(e))
            return None
