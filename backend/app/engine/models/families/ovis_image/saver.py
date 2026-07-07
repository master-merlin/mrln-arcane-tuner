"""Ovis-Image LoRA saver — saves LoRA weights as safetensors.

Key format: ``diffusion_model.{module}.lora_A/B.weight``.
Same ai-toolkit-compatible format used by the FLUX/zimage/krea2 families.
No upstream LoRA loader mixin exists for Ovis — this format is the
format of record (414 keys for the full checkpoint; pinned by
``test_ovis_image_lora_portability.py``).
"""

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class OvisImageSaver(GenericLoRASaver):
    """Saves Ovis-Image model weights as LoRA safetensors."""

    architecture_name = "ovis_image"
