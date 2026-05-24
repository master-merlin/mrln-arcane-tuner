"""HiDream-O1 sampler — extends ``GenericSamplingPipeline``.

HiDream-O1's vendored ``generate_image(...)`` is a monolithic end-to-end
inference call (no separate encode_prompt / denoise / decode_latents phases),
so we override ``_sample_single`` directly rather than implementing the
generic 3-phase lifecycle hooks. The abstract hooks are stubbed to raise
``NotImplementedError`` — they're never called because ``_sample_single``
short-circuits them.

The base ``GenericSamplingPipeline.generate_samples(step)`` still drives:
- eval mode + gradient-checkpointing disable
- EMA swap
- wildcard expansion of ``[triggerword]`` / ``[captionprefix]``
- output dir resolution
- PNG save + WebSocket broadcast
- restore training state

Default sampling constants reflect the FULL variant per the HF model card:
50 inference steps, guidance scale 5.0. The Dev (distilled) variant uses
28 / 1.0 — deferred to a follow-up PR.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
from PIL import Image

from app.engine.core.sampling import GenericSamplingPipeline

from .vendor import pipeline as vendored_pipeline

logger = structlog.get_logger(__name__)

DEFAULT_STEPS_FULL: int = 50
DEFAULT_GUIDANCE_FULL: float = 5.0
DEFAULT_WIDTH: int = 1024
DEFAULT_HEIGHT: int = 1024


class HiDreamO1Sampler(GenericSamplingPipeline):
    """Sample images via the vendored monolithic ``generate_image``.

    Overrides ``_sample_single`` instead of implementing the 3-phase
    lifecycle hooks because HiDream-O1 doesn't expose encode/denoise/decode
    separately — its custom unified transformer does all three internally.
    """

    def __init__(self, pipeline: Any) -> None:
        super().__init__(pipeline)
        self._processor: Any = None  # lazy-loaded on first sample

    # ── Main override ────────────────────────────────────────────────────

    def _sample_single(
        self,
        prompt_cfg: dict[str, Any],
        step: int,
    ) -> Image.Image:
        """Generate one sample via the vendored ``generate_image()``."""
        prompt = prompt_cfg.get("prompt", "")
        seed = int(prompt_cfg.get("seed", 42))
        width = int(prompt_cfg.get("width", DEFAULT_WIDTH))
        height = int(prompt_cfg.get("height", DEFAULT_HEIGHT))
        num_steps = int(prompt_cfg.get("num_inference_steps", DEFAULT_STEPS_FULL))
        guidance = float(prompt_cfg.get("guidance_scale", DEFAULT_GUIDANCE_FULL))

        # Lazy-load processor on first sample.
        processor = self._get_processor()
        model = self.pipeline._get_primary_model()

        self.logger.info(
            "hidream_o1.sampler.generate_start",
            step=step + 1,
            prompt_chars=len(prompt),
            steps=num_steps,
            guidance=guidance,
            width=width,
            height=height,
            seed=seed,
        )

        result = vendored_pipeline.generate_image(
            model=model,
            processor=processor,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            seed=seed,
            use_flash_attn=False,    # safer default — flash-attn may not be installed
            use_sage_attn=False,
        )

        # generate_image returns either a single PIL image or a list. Normalize.
        if isinstance(result, list):
            image = result[0]
        else:
            image = result

        self.logger.info(
            "hidream_o1.sampler.generate_complete",
            step=step + 1,
            size=(image.width, image.height),
        )
        return image

    # ── Lazy processor ───────────────────────────────────────────────────

    def _get_processor(self) -> Any:
        """Lazy-load AutoProcessor from the HF repo (cached after first call)."""
        if self._processor is not None:
            return self._processor

        from transformers import AutoProcessor

        # Try the trainer's processor field first (if the loader path
        # populated it); fall back to AutoProcessor.from_pretrained.
        trainer_processor = getattr(self.pipeline, "processor", None)
        if trainer_processor is not None:
            self._processor = trainer_processor
            return self._processor

        repo_id = "HiDream-ai/HiDream-O1-Image"
        defn = getattr(self.pipeline, "definition", None)
        if defn is not None:
            components = getattr(defn, "components", None) or {}
            unet_spec = (
                components.get("unet") if isinstance(components, dict) else None
            )
            if unet_spec is not None:
                repo_id = (
                    getattr(unet_spec, "repo", None)
                    or getattr(unet_spec, "path", None)
                    or repo_id
                )

        self.logger.info(
            "hidream_o1.sampler.loading_processor",
            repo_id=repo_id,
        )
        self._processor = AutoProcessor.from_pretrained(
            repo_id, trust_remote_code=False,
        )
        return self._processor

    # ── Abstract-hook stubs ──────────────────────────────────────────────
    # These exist solely to satisfy ABC instantiation. They are never
    # invoked because we override _sample_single above.

    def encode_prompt(self, prompt: str) -> Any:  # pragma: no cover
        raise NotImplementedError(
            "HiDream-O1 uses monolithic generate_image — see _sample_single override.",
        )

    def denoise(
        self,
        noise: torch.Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Any:  # pragma: no cover
        raise NotImplementedError(
            "HiDream-O1 uses monolithic generate_image — see _sample_single override.",
        )

    def decode_latents(self, latents: Any) -> Image.Image:  # pragma: no cover
        raise NotImplementedError(
            "HiDream-O1 uses monolithic generate_image — see _sample_single override.",
        )

    def _create_initial_noise(
        self,
        width: int,
        height: int,
        generator: torch.Generator,
    ) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError(
            "HiDream-O1 uses monolithic generate_image — see _sample_single override.",
        )
