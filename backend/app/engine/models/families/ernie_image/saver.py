"""ERNIE-Image LoRA saver -- saves LoRA weights as safetensors.

Output key format::

    diffusion_model.{module}.lora_A.weight
    diffusion_model.{module}.lora_B.weight

This matches Ostris's ``ai-toolkit`` ERNIE-Image LoRA convention
(``key.replace("transformer.", "diffusion_model.")``) and is the
de-facto ComfyUI-compatible format for ERNIE-Image LoRAs.
"""

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class ErnieImageSaver(GenericLoRASaver):
    """Saves ERNIE-Image LoRA weights as ComfyUI-compatible safetensors."""

    architecture_name = "ernie_image"
