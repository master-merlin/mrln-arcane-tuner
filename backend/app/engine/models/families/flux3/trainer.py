"""FLUX.3 Image Klein trainer -- family hooks for the generic training pipeline (LANE-144 spec D3).

- ONE job-wide text window: ``_pre_cache_text_embeddings`` computes ``self.text_window`` once over the
  complete caption set the shared ``_build_caption_hints()`` returns (composites, dropout variants, the
  empty string, the trigger-only string, the sample prompts) and encodes every hint at it before the
  text encoder is offloaded. The shared loop batches by resolution bucket and substitutes a dropped
  caption by its dropout-variant string, so every cached row has length ``W`` and any batch stacks.
  The TE disk cache directory carries ``w<W>``: a dataset edit that grows the window never reads a
  stale file.
- Timesteps: a base sample from the configured mode through the FIXED shift
  ``t' = a*t / (1 + (a-1)*t)``, ``a = 6.93`` (the checkpoint's schedule; the ``flux2`` resolution-
  dependent ``flux_shift`` is not inherited), returned in the pipeline's ``[0, 1000]`` scale.
- Target: the pipeline default ``noise - x0`` (flow matching, 1 = noise).
"""

from __future__ import annotations

import os
from typing import Any

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline

logger = structlog.get_logger(__name__)

# Spec D3: the fixed flow-match shift (reference FLOW_SHIFT; YAML ``scheduler.shift``).
DEFAULT_FLOW_SHIFT = 6.93
# Base modes that already carry their own (resolution-dependent) shift: stacking 6.93 on them would
# shift twice, so they fall back to a uniform base sample (reference default weighting "none").
_SELF_SHIFTED_MODES = ("flux_shift", "model_shift")


def fixed_shift(t: torch.Tensor, shift: float = DEFAULT_FLOW_SHIFT) -> torch.Tensor:
    """``a*t / (1 + (a-1)*t)`` on ``t`` in ``[0, 1]``."""
    return shift * t / (1.0 + (shift - 1.0) * t)


class Flux3Trainer(GenericTrainingPipeline):
    """FLUX.3 Image Klein (Base) LoRA trainer."""

    text_window: int | None = None

    # -- Setup --

    def _setup_family(self) -> None:
        from .driver import Flux3Driver  # noqa: PLC0415
        from .loader import Flux3Loader  # noqa: PLC0415

        self.driver = Flux3Driver(self.definition, self.device)
        self.loader = Flux3Loader(self.device)

    def _create_sampler(self):
        """Create a Flux3Sampler if sampling is configured."""
        if int(self.config.get("sample_every_n_steps", 0)) > 0:
            from .sampler import Flux3Sampler  # noqa: PLC0415

            return Flux3Sampler(self)
        return None

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep self.transformer, the components dict and the driver in sync after PEFT wrapping."""
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

    # -- Text-embedding pre-cache at ONE job window --

    def _te_cache_dir(self, window: int) -> str:
        dirs = self._resolve_te_cache_dirs()
        if not dirs:
            return ""
        te_quant = self.config.get("te_quantization", "none")
        return os.path.join(dirs[0], "embeddings", te_quant, "te1", f"w{window}")

    def _pre_cache_text_embeddings(self) -> None:
        """Fix ``self.text_window`` over the shared caption hints, then warm the cache at it.

        The window is fixed even when caching is off, so the direct path encodes at it as well.
        """
        caption_hints = self._build_caption_hints()
        if self.driver is None or getattr(self.driver, "tokenizer", None) is None:
            return
        window = self.driver.window_for(list(caption_hints))
        self.text_window = window
        self.logger.info("flux3_text_window", window=window, hints=len(caption_hints),
                         cap=self.driver.max_length)
        if getattr(self, "_log_writer", None):
            self._log_writer.log(f"FLUX.3 text window: {window} tokens over {len(caption_hints)} prompts")

        if not self.config.get("cache_text_embeddings", True):
            return
        if self.text_encoder is None:
            return

        from app.engine.components.text_embeddings import TextEmbeddingCache

        te_cache_dir = self._te_cache_dir(window)
        disk_loaded = 0
        need_encode: list[tuple[str, str]] = []
        for caption, hint in caption_hints.items():
            cached = self.text_cache.get(caption)
            if cached is not None and cached.shape[-2] == window:
                continue  # already in memory at this window (e.g. restored from a checkpoint)
            if te_cache_dir:
                tensor = TextEmbeddingCache.load(caption, te_cache_dir, hint)
                if tensor is not None and tensor.shape[-2] == window:
                    self.text_cache[caption] = tensor
                    disk_loaded += 1
                    continue
            need_encode.append((caption, hint))

        total = len(caption_hints)
        self.logger.info(
            "te_disk_cache_status", total=total, window=window,
            from_memory=total - disk_loaded - len(need_encode),
            from_disk=disk_loaded, need_encode=len(need_encode),
        )
        if not need_encode:
            if getattr(self, "_log_writer", None):
                self._log_writer.status("TE Cache Loaded from Disk")
            self.logger.info("text_embedding_cache_complete", cached_embeddings=len(self.text_cache),
                             source="disk")
            return

        if getattr(self, "_log_writer", None):
            self._log_writer.status("Caching Text Embeddings (0%)")
        encode_total = len(need_encode)
        batch_size = 4
        dtype = self._resolve_loading_dtype()
        with torch.no_grad():
            for i in range(0, encode_total, batch_size):
                batch_items = need_encode[i: i + batch_size]
                output = self.driver.encode_text([cap for cap, _ in batch_items], dtype, window=window)
                emb_batch = output.embeddings if hasattr(output, "embeddings") else output
                for j, (cap, hint) in enumerate(batch_items):
                    emb_cpu = emb_batch[j: j + 1].cpu()
                    self.text_cache[cap] = emb_cpu
                    if te_cache_dir:
                        TextEmbeddingCache.save(cap, emb_cpu, te_cache_dir, hint)
                pct = round(min(i + batch_size, encode_total) / encode_total * 100)
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Caching Text Embeddings ({pct}%)")
        self.logger.info("text_embedding_cache_complete", cached_embeddings=len(self.text_cache),
                         newly_encoded=encode_total, window=window)

    def _offload_text_encoders(self) -> None:
        """Base TE offload/unload; when unloading, drop the processor too (the flux2 behaviour)."""
        unloading = self.config.get("unload_text_encoder", False)
        super()._offload_text_encoders()
        if unloading and not hasattr(self, "_tok_cleaned"):
            self.tokenizer = None
            self.components.pop("tokenizer", None)
            self._tok_cleaned = True

    # -- Text encoding --

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None,
    ) -> Any:
        """Raw ``(B, W, 20480)`` tensor; ``batch`` is accepted for hook compatibility and ignored."""
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(captions, dtype)
        output = self.driver.encode_text(list(captions), dtype, window=self.text_window)
        return output.embeddings if hasattr(output, "embeddings") else output

    def _get_cached_text_embeddings(self, captions: list[str], dtype: torch.dtype) -> torch.Tensor:
        """Stack cached rows; a miss is encoded at the job window (flux2's offload-miss warning path)."""
        uncached = [cap for cap in dict.fromkeys(captions) if cap not in self.text_cache]
        if uncached and self.text_encoder is not None:
            te_device = next(self.text_encoder.parameters()).device
            te_was_offloaded = te_device != self.device
            if te_was_offloaded:
                self.logger.warning(
                    "te_cache_miss_after_offload", count=len(uncached), window=self.text_window,
                    hint="pre-caching should have covered all captions",
                )
                self.text_encoder.to(self.device)
            try:
                for cap in uncached:
                    out = self.driver.encode_text([cap], dtype, window=self.text_window)
                    emb = out.embeddings if hasattr(out, "embeddings") else out
                    self.text_cache[cap] = emb.cpu()
            finally:
                if te_was_offloaded:
                    self.text_encoder.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        elif uncached:
            raise RuntimeError(
                "Text encoder was unloaded but encountered uncached caption(s): "
                + ", ".join(cap[:50] for cap in uncached)
            )
        return torch.cat([self.text_cache[cap].to(self.device, dtype=dtype) for cap in captions], dim=0)

    # -- Timesteps --

    def _flow_shift(self) -> float:
        arch = getattr(getattr(self, "definition", None), "architecture_params", None) or {}
        return float(arch.get("scheduler.shift", DEFAULT_FLOW_SHIFT))

    def sample_timesteps(self, batch_size: int, latents: torch.Tensor | None = None) -> torch.Tensor:
        """Base sample of the configured mode -> fixed 6.93 shift -> ``[0, 1000]`` (resolution-free)."""
        from app.engine.strategies.timestep_sampling import TimestepSampler

        mode = self.config.get("timestep_sampling", "uniform")
        if mode in _SELF_SHIFTED_MODES:
            if not getattr(self, "_flux3_mode_warned", False):
                self.logger.warning("flux3_timestep_mode_replaced", configured=mode, used="uniform",
                                    reason="flux3 applies its fixed 6.93 shift; a second shift is refused")
                self._flux3_mode_warned = True
            mode = "uniform"
        max_steps = getattr(self, "max_train_steps", 1)
        progress = getattr(self, "global_step", 0) / max(max_steps, 1)
        base = TimestepSampler.sample(mode, batch_size, self.device, self.config, progress=progress)
        return (fixed_shift(base, self._flow_shift()) * 1000.0).clamp(0.0, 1000.0)
