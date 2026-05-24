"""HiDream-O1 sampler — pixel-space text-to-image generation.

Wraps the vendored ``pipeline.generate_image(...)`` call in
``asyncio.to_thread`` so the WebSocket-driven training loop isn't
blocked. The HiDream-O1 model is pixel-space (no VAE decode), so the
vendored helper returns PIL images directly.

Default sampling constants reflect the FULL variant per the HF model
card: 50 inference steps, guidance scale 5.0. The Dev (distilled)
variant uses 28 / 1.0 — that variant is deferred to a follow-up PR.

The vendored ``generate_image`` exposes many knobs (resolution,
scheduler choice, num_samples, etc.). We forward kwargs as-is and let
the family definition YAML supply defaults — see Task 14.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from .vendor import pipeline as vendored_pipeline

logger = structlog.get_logger(__name__)

DEFAULT_STEPS_FULL: int = 50
DEFAULT_GUIDANCE_FULL: float = 5.0
DEFAULT_WIDTH: int = 1024
DEFAULT_HEIGHT: int = 1024


class HiDreamO1Sampler:
    """Generate samples via the vendored HiDream-O1 pipeline."""

    def __init__(self, driver: Any, definition: Any, processor: Any = None):
        """Args:
            driver: A ``HiDreamO1Driver`` instance with the loaded model.
            definition: ``ModelDefinition`` (used to read defaults like
                ``steps_default``, ``guidance_scale_default`` from
                ``architecture_params`` if present).
            processor: Optional processor for handling image/text formatting.
                If not provided, will be loaded from the HF repo if needed.
        """
        self.driver = driver
        self.definition = definition
        self.processor = processor
        self.logger = structlog.get_logger(self.__class__.__name__)

    async def sample(
        self,
        prompts: list[str],
        *,
        steps: int | None = None,
        guidance_scale: float | None = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Generate one image per prompt and return PIL images.

        Args:
            prompts: List of text prompts.
            steps: Inference steps. Falls back to definition's
                ``steps_default`` or ``DEFAULT_STEPS_FULL``.
            guidance_scale: CFG scale. Falls back to definition's
                ``guidance_scale_default`` or ``DEFAULT_GUIDANCE_FULL``.
            width: Output width (must be patch-aligned: multiple of 32).
            height: Output height (must be patch-aligned: multiple of 32).
            seed: Optional RNG seed.
            **kwargs: Forwarded to ``generate_image`` (e.g.,
                ``num_samples``, ``scheduler``).

        Returns:
            One PIL image per prompt (typed ``Any`` to keep PIL import
            optional — caller iterates and saves them).
        """
        arch_params = getattr(self.definition, "architecture_params", {}) or {}
        steps = steps or arch_params.get("steps_default") or DEFAULT_STEPS_FULL
        guidance_scale = (
            guidance_scale
            or arch_params.get("guidance_scale_default")
            or DEFAULT_GUIDANCE_FULL
        )

        model = self.driver.get_primary_model()
        processor = self.processor

        # If processor is not provided, lazy-load it from the HF repo
        if processor is None:
            def _load_processor():
                from transformers import AutoProcessor
                repo = (
                    getattr(
                        getattr(self.definition, "components", {}) or {}, "unet", None,
                    ) or {}
                )
                repo_id = (
                    getattr(repo, "repo", None)
                    or getattr(repo, "path", None)
                    or "HiDream-ai/HiDream-O1-Image"
                )
                return AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)

            processor = _load_processor()

        def _blocking_generate() -> list[Any]:
            outputs: list[Any] = []
            for prompt in prompts:
                self.logger.info(
                    "hidream_o1.sampler.generate",
                    prompt_chars=len(prompt),
                    steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                )
                # The vendored generate_image signature requires processor.
                result = vendored_pipeline.generate_image(
                    model=model,
                    processor=processor,
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    seed=seed,
                    **kwargs,
                )
                # generate_image returns a single PIL image per prompt
                # (pixel-space, no VAE decode).
                outputs.append(result)
            return outputs

        return await asyncio.to_thread(_blocking_generate)
