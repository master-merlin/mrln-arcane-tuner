"""LongCat-Image LoRA saver -- saves LoRA weights as safetensors.

Key format: ``diffusion_model.{module}.lora_A/B.weight``.
Same ai-toolkit-compatible format used by the FLUX families.  No upstream
LoRA loader mixin exists for LongCat-Image — these canonical keys are the
format of record (pinned by test_longcat_image_lora_portability.py).
"""

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class LongCatImageSaver(GenericLoRASaver):
    """Saves LongCat-Image model weights as LoRA safetensors."""

    architecture_name = "longcat_image"
