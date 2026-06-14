"""Generic sampling pipeline for generating images during training.

This module provides the abstract base class ``GenericSamplingPipeline`` that
model families override to produce sample images at periodic intervals during
training.  It handles:

- Step-based scheduling (``sample_every_n_steps``, ``sample_skip_first_n_steps``)
- Wildcard expansion (``[triggerword]``, ``[captionprefix]``)
- EMA weight swap (if enabled)
- Output persistence and WebSocket broadcast
- ``@torch.no_grad()`` context for zero-gradient overhead

Family-specific subclasses implement ``encode_prompt``, ``denoise``,
``decode_latents``, and ``_create_initial_noise`` hooks.

A family may return either a :class:`PIL.Image.Image` (still) **or** a
:class:`SampleArtifact` (short video clip, optionally with audio) from
``decode_latents``.  The base persists the right artifact (``.png`` vs
``.mp4``) and tags the broadcast event with ``media_type`` so the frontend
can pick ``<img>`` vs ``<video>``.
"""

from __future__ import annotations

import gc
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import torch
from PIL import Image
from torch import Tensor

if TYPE_CHECKING:
    from app.engine.core.pipeline import GenericTrainingPipeline

logger = structlog.get_logger(__name__)


@dataclass
class SampleArtifact:
    """A decoded video sample to be persisted as an mp4.

    Returned from a video-capable family's ``decode_latents`` in place of a
    :class:`PIL.Image.Image`.  The base writes it through
    ``VideoFrameLoader.encode_video`` (H.264 + optional AAC audio).

    Canonical frame layout is **``[C, F, H, W]`` float in ``[-1, 1]``** — the
    exact tensor ``VideoFrameLoader.load_clip`` returns and ``encode_video``
    consumes natively (a ``[F, H, W, C]`` uint8 tensor is also accepted by
    ``encode_video`` and therefore tolerated here, but ``[C, F, H, W]`` float
    is the documented contract).

    Attributes:
        frames: ``[C, F, H, W]`` float tensor in ``[-1, 1]`` (canonical), or a
            ``[F, H, W, C]`` uint8 tensor.
        audio: Optional waveform in ``[-1, 1]`` (1-D mono / 2-D channels-first),
            or a ``(waveform, sample_rate)`` tuple, muxed as an AAC stream.
        fps: Output framerate for the encoded clip.
    """

    frames: Tensor
    audio: Tensor | None = None
    fps: float = 16.0


class GenericSamplingPipeline(ABC):
    """Base sampling pipeline — runs inference using the model being trained.

    Lifecycle (called every training step):
        1. ``should_sample(step)`` → check interval/skip
        2. ``generate_samples(step)`` →
           a. eval mode + EMA swap
           b. ``encode_prompt()`` per prompt (family hook)
           c. ``_create_initial_noise()`` (family hook)
           d. ``denoise()`` (family hook)
           e. ``decode_latents()`` (family hook)
           f. save PNG + broadcast
           g. restore training state
    """

    def __init__(self, pipeline: GenericTrainingPipeline) -> None:
        self.pipeline = pipeline
        self.config: dict[str, Any] = pipeline.config
        self.device: torch.device = pipeline.device
        self.logger = structlog.get_logger(self.__class__.__name__)

    # ── Scheduling ───────────────────────────────────────────────────────

    def should_sample(self, step: int) -> bool:
        """Check whether sampling should occur at *step*.

        Uses 1-based step numbering (``step + 1``) to align with the
        UI display.  When the user configures ``sample_every_n_steps: 50``,
        sampling fires when the displayed step is 50, 100, 150, …

        Independent from the checkpoint save interval — user controls both
        ``sample_every_n_steps`` and ``save_every_n_steps`` separately.

        Also checks for:
        - A ``sampling_cadence`` override file — if present, its value
          replaces the config interval **and** is persisted back into
          ``self.config`` so checkpoint saves capture the updated cadence.
        - A ``sampling_paused`` flag file — if present, sampling is
          skipped entirely.
        """
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval <= 0:
            return False

        # Resolve output base once (used for both cadence + pause checks)
        output_base = self._resolve_output_dir().parent

        # Runtime cadence override (file-based signal from UI)
        cadence_path = os.path.join(output_base, "sampling_cadence")
        if os.path.exists(cadence_path):
            try:
                with open(cadence_path) as f:
                    override = int(f.read().strip())
                if override > 0:
                    if override != interval:
                        self.logger.info(
                            "sampling_cadence_override",
                            old_interval=interval,
                            new_interval=override,
                        )
                        # Persist into config so checkpoint saves capture it
                        self.config["sample_every_n_steps"] = override
                    interval = override
            except (ValueError, OSError):
                pass

        displayed_step = step + 1  # UI shows 1-based steps
        skip_first = int(self.config.get("sample_skip_first_n_steps", 0))
        if displayed_step <= skip_first:
            return False
        if displayed_step % interval != 0:
            return False
        # Check for sampling_paused flag file
        flag_path = os.path.join(output_base, "sampling_paused")
        if os.path.exists(flag_path):
            self.logger.info("sampling_skipped_paused", step=displayed_step)
            return False
        return True

    # ── Wildcard Expansion ───────────────────────────────────────────────

    def _expand_wildcards(self, prompt: str) -> str:
        """Expand ``[triggerword]`` and ``[captionprefix]`` in *prompt*.

        - ``[triggerword]`` → ``global_triggerword`` from config
        - ``[captionprefix]`` → first dataset's ``caption_prefix``
        """
        triggerword = self.config.get("global_triggerword", "")
        prompt = prompt.replace("[triggerword]", triggerword)

        datasets = self.config.get("datasets", [])
        if datasets:
            first = datasets[0]
            prefix = (
                first.get("caption_prefix", "")
                if isinstance(first, dict)
                else getattr(first, "caption_prefix", "")
            )
        else:
            prefix = ""
        prompt = prompt.replace("[captionprefix]", prefix)
        return prompt

    # ── Main Entry Point ─────────────────────────────────────────────────

    @torch.no_grad()
    def generate_samples(self, step: int, final: bool = False) -> list[Path]:
        """Generate samples, save to disk and broadcast via WebSocket.

        Each sample is a PNG (image families) or an mp4 (video families,
        when ``decode_latents`` returns a :class:`SampleArtifact`).

        Args:
            step: Current training step number (0-based internal).
            final: If True, use ``_final`` suffix instead of step number.

        Returns:
            List of saved sample file paths (``.png`` or ``.mp4``).
        """
        prompts = self._get_sample_prompts()
        if not prompts:
            return []

        # Guard: NVFP4 quantized models don't support all aten ops needed
        # for sampling (e.g. aten.expand).  Skip with a clear warning.
        quant_scheme = self.config.get("quantization", "none")
        if quant_scheme == "nvfp4":
            self.logger.warning(
                "sampling_skipped_nvfp4",
                scheme=quant_scheme,
                reason="NVFP4 prototype tensor does not support aten.expand; sampling unavailable",
            )
            if getattr(self, "_log_writer", None):
                self._log_writer.warning(
                    f"Sampling skipped — {quant_scheme} quantization does not support inference ops"
                )
            return []

        displayed_step = step + 1
        self.logger.info(
            "sampling_start",
            step=displayed_step,
            num_prompts=len(prompts),
            final=final,
        )
        t0 = time.perf_counter()

        # 1. Switch to eval mode & disable gradient checkpointing
        # Gradient checkpointing MUST be disabled during sampling:
        # torch.utils.checkpoint.checkpoint re-runs the forward under
        # torch.enable_grad() internally, which interacts poorly with the
        # outer torch.no_grad() context causing activation mismatches.
        model = self.pipeline._get_primary_model()
        was_training = model.training
        had_grad_ckpt = getattr(model, "is_gradient_checkpointing", False)
        model.eval()
        if had_grad_ckpt and hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
            self.logger.debug("sampling_grad_ckpt_disabled")

        # Schedule-Free optimizers need eval/train mode switch too
        optimizer = getattr(self.pipeline, "optimizer", None)
        if optimizer and hasattr(optimizer, "eval"):
            optimizer.eval()

        # 2. EMA swap (if enabled)
        ema = getattr(self.pipeline, "ema_handler", None)
        if ema:
            ema.store_and_swap()

        try:
            output_dir = self._resolve_output_dir()
            saved_paths: list[Path] = []

            # NOTE: No outer autocast here.  Each family's sampler manages
            # its own autocast context around the transformer forward pass.
            # Wrapping _sample_single in autocast caused bfloat16 precision
            # issues in VAE decode and Euler accumulation.

            for i, prompt_cfg in enumerate(prompts):
                # Materialise the prompt config as a plain dict
                cfg = (
                    prompt_cfg.model_dump()
                    if hasattr(prompt_cfg, "model_dump")
                    else dict(prompt_cfg)
                )

                # Expand wildcards
                raw_prompt = cfg.get("prompt", "")
                cfg["prompt"] = self._expand_wildcards(raw_prompt)

                artifact = self._sample_single(cfg, step)

                path = self._persist_artifact(
                    artifact, output_dir, i, displayed_step, final
                )
                saved_paths.append(path)

                media_type = (
                    "video" if isinstance(artifact, SampleArtifact) else "image"
                )
                self._broadcast_sample(
                    path, displayed_step, i, cfg, raw_prompt, media_type=media_type
                )

            elapsed = time.perf_counter() - t0
            self.logger.info(
                "sampling_complete",
                step=displayed_step,
                num_images=len(saved_paths),
                elapsed_s=round(elapsed, 2),
            )
            return saved_paths

        finally:
            # 3. Restore: gradient checkpointing, EMA, training mode
            if had_grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            if ema:
                ema.restore()
            if was_training:
                model.train()
            if optimizer and hasattr(optimizer, "train"):
                optimizer.train()
            # Sync first so all denoise kernels complete before we let
            # the allocator reclaim. gc.collect breaks any circular refs
            # holding intermediates (PyTorch autograd graphs, scheduler
            # state). empty_cache returns reserved blocks to the driver,
            # which on Windows/WDDM is required to release pages from
            # the shared-memory rail that a transient sampling spike
            # may have spilled into.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Abstract Hooks (family-specific) ─────────────────────────────────

    @abstractmethod
    def encode_prompt(self, prompt: str) -> Any:
        """Encode a text prompt into embeddings.

        May re-use the trainer's cached text embeddings if available.

        Returns:
            Prompt embedding(s) in the format expected by ``denoise()``.
        """

    @abstractmethod
    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Any:
        """Run the full denoising loop.

        Returns:
            Latent tensor(s) ready for ``decode_latents()``.
        """

    @abstractmethod
    def decode_latents(self, latents: Any) -> Image.Image | SampleArtifact:
        """Decode latent tensor(s) to a sample.

        Returns:
            A :class:`PIL.Image.Image` for still-image families, or a
            :class:`SampleArtifact` for video families (a ``[C, F, H, W]``
            float clip plus optional audio + fps).  The base persists the
            correct file type (``.png`` vs ``.mp4``) accordingly.
        """

    @abstractmethod
    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator
    ) -> Tensor:
        """Create an initial noise tensor of the correct shape for this family."""

    # ── Concrete Helpers ─────────────────────────────────────────────────

    def _get_sample_prompts(self) -> list[dict[str, Any]]:
        """Read sample prompts from config."""
        raw = self.config.get("sample_prompts", [])
        result: list[dict[str, Any]] = []
        for entry in raw:
            if hasattr(entry, "model_dump"):
                result.append(entry.model_dump())
            elif isinstance(entry, dict):
                result.append(entry)
        return result

    def _resolve_output_dir(self) -> Path:
        """Create output directory: ``{checkpoint_output_dir}/samples/``.

        Uses the same base output directory as the checkpoint manager
        so samples sit alongside checkpoint folders.
        """
        # Prefer the resolved checkpoint path (includes model-part suffix)
        ckpt_mgr = getattr(self.pipeline, "checkpoint_manager", None)
        if ckpt_mgr and hasattr(ckpt_mgr, "output_dir"):
            base = Path(ckpt_mgr.output_dir)
        else:
            # Fallback: replicate the path logic from _configure_managers
            output_root = self.config.get("output_dir", "outputs")
            lora_name = self.config.get("lora_name", "lora")
            definition = getattr(self.pipeline, "definition", None)
            if definition:
                from app.core.naming import model_part_from_definition_id

                model_part = model_part_from_definition_id(definition.id)
                run_name = f"{lora_name}_{model_part}"
            else:
                run_name = lora_name
            base = Path(output_root) / run_name

        sample_dir = base / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        return sample_dir

    # ── Device helpers for phased sampling ─────────────────────────────

    def _resolve_component(self, name: str):
        """Resolve a named component from pipeline attrs, components dict, or driver.

        After ``_offload_text_encoders()`` clears pipeline-level refs,
        the driver still holds component references.  This method provides
        a unified fallback chain.
        """
        comp = getattr(self.pipeline, name, None) or self.pipeline.components.get(name)
        if comp is None:
            driver = getattr(self.pipeline, "driver", None)
            if driver is not None:
                comp = getattr(driver, name, None)
        return comp

    def _ensure_on_gpu(self, names: list[str]) -> list[str]:
        """Move named components to GPU if they are on CPU.

        Returns list of component names that were actually moved
        (so only those need to be offloaded afterwards).
        """
        gpu = self.device
        moved: list[str] = []
        for name in names:
            comp = self._resolve_component(name)
            if comp is None or not isinstance(comp, torch.nn.Module):
                continue
            try:
                first_param = next(comp.parameters())
            except StopIteration:
                continue
            if first_param.device.type != gpu.type:
                comp.to(gpu)
                moved.append(name)
        return moved

    def _offload_to_cpu(self, names: list[str]) -> None:
        """Move named components back to CPU and free VRAM."""
        for name in names:
            comp = self._resolve_component(name)
            if comp is not None and isinstance(comp, torch.nn.Module):
                comp.to("cpu")
        if names:
            torch.cuda.empty_cache()

    def _ensure_transformer_on_device(self, transformer: torch.nn.Module) -> None:
        """Place the transformer on the sampling device for denoise.

        Skipped when block swapping is active: the swap manager's
        pre-forward hooks pull blocks to GPU on demand, and a bulk
        ``module.to(device)`` would put every block on GPU at once
        (~40 GB on Qwen-class models), defeating the swap and bloating
        the sampling peak by the full base-model size.
        """
        if getattr(self.pipeline, "_block_swap_managers", None):
            return
        transformer.to(self.device)

    def _sample_single(
        self, prompt_cfg: dict[str, Any], step: int
    ) -> Image.Image | SampleArtifact:
        """Generate one sample from a prompt config entry.

        Returns a :class:`PIL.Image.Image` for image families or a
        :class:`SampleArtifact` for video families — whatever the family's
        ``decode_latents`` produces.

        Uses **phased GPU management** to minimise peak VRAM:
        1. TE → GPU → encode prompt → TE → CPU  (free VRAM)
        2. Denoise with only the transformer on GPU
        3. VAE → GPU → decode → VAE → CPU

        Args:
            prompt_cfg: Dict with ``prompt``, ``seed``, ``width``, ``height``,
                ``num_inference_steps``, ``guidance_scale``.
            step: Current training step (unused in base, available for hooks).
        """
        prompt = prompt_cfg.get("prompt", "")
        seed = int(prompt_cfg.get("seed", 42))
        width = int(prompt_cfg.get("width", 1024))
        height = int(prompt_cfg.get("height", 1024))
        num_steps = int(prompt_cfg.get("num_inference_steps", 20))
        guidance = float(prompt_cfg.get("guidance_scale", 3.5))

        # ── Phase 1: Encode prompt (TE on GPU) ──
        # Prefer driver.get_text_encoders() (new interface) with fallback
        # to pipeline._get_text_encoders() for backward compatibility.
        driver = getattr(self.pipeline, "driver", None)
        if driver is not None:
            te_names = list(driver.get_text_encoders().keys())
        else:
            te_names = list(self.pipeline._get_text_encoders().keys())
        te_moved = self._ensure_on_gpu(te_names)
        text_emb = self.encode_prompt(prompt)
        self._offload_to_cpu(te_moved)

        # ── Phase 2: Denoise (only transformer on GPU) ──
        # Stash the full prompt config so edit/kontext samplers can read
        # control_images without changing the abstract denoise() signature.
        # Non-edit samplers ignore it.
        self._active_prompt_cfg = prompt_cfg
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = self._create_initial_noise(width, height, generator)
        latents = self.denoise(noise, text_emb, num_steps, guidance, seed)

        # ── Phase 3: Decode latents (VAE on GPU) ──
        vae_moved = self._ensure_on_gpu(["vae"])
        artifact = self.decode_latents(latents)
        self._offload_to_cpu(vae_moved)

        return artifact

    def _persist_artifact(
        self,
        artifact: Image.Image | SampleArtifact,
        output_dir: Path,
        index: int,
        displayed_step: int,
        final: bool,
    ) -> Path:
        """Write a decoded sample to disk and return its path.

        - :class:`PIL.Image.Image` → ``.png`` (filename pattern unchanged from
          the original inline ``image.save`` — ``sample_{index:02d}_final.png``
          for the final sample, ``sample_{index:02d}_step{step:06d}.png``
          otherwise).
        - :class:`SampleArtifact` → ``.mp4`` via ``VideoFrameLoader`` (lazy
          import so importing this module never drags PyAV), muxing
          ``artifact.audio`` when present.
        """
        if final:
            stem = f"sample_{index:02d}_final"
        else:
            stem = f"sample_{index:02d}_step{displayed_step:06d}"

        if isinstance(artifact, SampleArtifact):
            # Lazy import: keeps PyAV out of this module's import graph.
            from app.engine.components.video import VideoFrameLoader

            path = output_dir / f"{stem}.mp4"
            VideoFrameLoader().encode_video(
                artifact.frames, artifact.audio, artifact.fps, str(path)
            )
            return path

        # Default: still image → PNG (original behavior, pattern unchanged).
        path = output_dir / f"{stem}.png"
        artifact.save(path)
        return path

    def _broadcast_sample(
        self,
        path: Path,
        step: int,
        index: int,
        expanded_cfg: dict[str, Any],
        raw_prompt: str = "",
        media_type: str = "image",
    ) -> None:
        """Broadcast a ``sample_generated`` event via the logging system.

        The WebSocket log handler forwards this to the frontend, which
        auto-refreshes the job samples gallery.  ``media_type`` is ``"image"``
        (default, preserves existing behavior) or ``"video"`` so the UI can
        render ``<img>`` vs ``<video>`` for the persisted artifact.
        """
        self.logger.info(
            "sample_generated",
            step=step,
            index=index,
            prompt=raw_prompt[:200],
            expanded_prompt=expanded_cfg.get("prompt", "")[:200],
            path=str(path),
            media_type=media_type,
            seed=expanded_cfg.get("seed", 42),
            guidance_scale=expanded_cfg.get("guidance_scale", 3.5),
        )
