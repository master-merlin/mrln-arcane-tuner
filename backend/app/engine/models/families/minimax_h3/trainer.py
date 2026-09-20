"""MiniMax-H3 Trainer — PR1 in progress.

Landed: the inverted flow-match delegations (row 1.2) and the text-embedding
lifecycle — pre-cache, unconditional encoder release, cache-serving
``encode_text``, deferred DiT materialisation (row 2.1). Still to land: the
real ``_setup_family`` (loader/saver, row 2.5), the packed-sequence forward
and the joint audio+video loss (2.4/2.6). Until 2.5, ``_setup_family`` raises
so ``MiniMaxH3Family.get_trainer_class()`` returns a real class
instead of raising — the registry-wide guard
(``tests/engine/test_hook_wiring_meta.py::test_every_family_resolves_a_trainer_and_driver``)
requires every registered family to resolve BOTH a trainer and a driver, and
a family whose trainer resolution raises is indistinguishable from a broken
one.

``_setup_family`` is the only method the base ``GenericTrainingPipeline``
requires a subclass to implement, and it is the first family hook the real
pipeline calls (via ``setup()``). Wiring ``self.driver = MiniMaxH3Driver(...)``
here — even though the next line always raises — keeps the trainer→driver
seam real and source-greppable (see
``tests/engine/test_hook_wiring_meta.py::_driver_for_trainer``, which
resolves a family's driver by regex-searching the trainer's MRO source for
exactly this assignment shape) instead of a fabricated shortcut. A job that
somehow reaches this trainer must fail loudly here, not limp through with a
missing loader/data path.
"""

from __future__ import annotations

import gc
import os
from typing import Any

import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from app.engine.core.text_encoding import TextEncoderOutput

from .driver import MiniMaxH3Driver


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
    """MiniMax-H3 LoRA trainer — PR0 STUB. Real training lands in PR1."""

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
        # items' clean audio latents; row 2.5 loads them from the audio cache
        # into `item["audio_latents"]` before this delegation.
        return self.driver.build_batch_extra(items)

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
        plan asks for: weight bytes + the loading peaks."""
        weight_bytes = self.driver.text_encoder_weight_bytes()
        super()._offload_text_encoders()
        te = self.driver.text_encoder
        if te is not None:
            if hasattr(te, "to"):
                te.to("cpu")
            self.components.pop("text_encoder", None)
            self.driver.release_text_encoders()
            self._te_unloaded = True
        if not isinstance(getattr(type(self), "text_encoder", None), property):
            self.text_encoder = None
        self.logger.info(
            "text_encoder released",
            weight_bytes=weight_bytes,
            host_peak_bytes=_host_peak_bytes(),
            cuda_peak_bytes=(
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            ),
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    def _setup_family(self) -> None:
        # Real assignment, so the trainer→driver seam is genuine (see the
        # module docstring): the resolution guard's regex finds this line,
        # not a decoy. No loader exists yet (PR1 scope), so setup stops here.
        self.driver = MiniMaxH3Driver(self.definition, self.device)
        raise NotImplementedError(
            "minimax_h3 training lands in PR1; PR0 ships the scaffold only."
        )
