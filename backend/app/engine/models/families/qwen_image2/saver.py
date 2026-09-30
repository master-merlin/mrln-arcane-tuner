"""Qwen-Image 2 LoRA saver -- diffusers PEFT keys.

Key format: ``transformer.{module}.lora_A/B.weight`` -- the diffusers layout
that ``QwenImageLoraLoaderMixin.load_lora_weights`` applies as it is. That
mixin is what ``QwenImage21Pipeline`` inherits (upstream
``pipeline_qwenimage21.py:25,159``), the external consumer of LANE-132's GPU
UAT; ``tests/test_qwen_image2_training.py`` loads a saved file through it.

Licence: Qwen-Image-2.1 weights are non-commercial (Qwen Research License) --
see the README licence table, not model metadata.
"""

from __future__ import annotations

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class QwenImage2Saver(GenericLoRASaver):
    """Saves the Qwen-Image 2.x transformer LoRA as safetensors."""

    architecture_name = "qwen_image2"
    key_prefix = "transformer."
