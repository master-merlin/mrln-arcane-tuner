"""Qwen-Image 2 LoRA saver -- diffusers PEFT keys, the definition's licence.

Key format: ``transformer.{module}.lora_A/B.weight`` -- the diffusers layout
that ``QwenImageLoraLoaderMixin.load_lora_weights`` applies as it is. That
mixin is what ``QwenImage21Pipeline`` inherits (upstream
``pipeline_qwenimage21.py:25,159``), the external consumer of LANE-132's GPU
UAT; ``tests/test_qwen_image2_training.py`` loads a saved file through it.

``modelspec.license`` carries the definition's ``license`` (Qwen-Image-2.1 is
non-commercial), through ``lora_metadata.license_metadata`` -- absent when the
definition declares none.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.engine.core.pipeline.saver_base import GenericLoRASaver
from app.engine.utils.lora_metadata import license_metadata


class QwenImage2Saver(GenericLoRASaver):
    """Saves the Qwen-Image 2.x transformer LoRA as safetensors."""

    architecture_name = "qwen_image2"
    key_prefix = "transformer."

    def __init__(self, license_text: str | None = None) -> None:
        self.license_text = license_text

    def save(
        self,
        components: dict[str, Any],
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        merged = {**(metadata or {}), **license_metadata(self.license_text)}
        super().save(components, path, metadata=merged or None)
