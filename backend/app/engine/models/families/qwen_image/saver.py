"""Qwen-Image LoRA saver -- saves LoRA weights as safetensors.

Key format: ``diffusion_model.{module}.lora_A/B.weight``.
Same ai-toolkit-compatible format used by FLUX families.
"""

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class QwenImageSaver(GenericLoRASaver):
    """Saves Qwen-Image model weights as LoRA safetensors."""

    architecture_name = "qwen_image"
