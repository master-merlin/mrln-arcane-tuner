"""DreamLite LoRA saver — saves LoRA weights as safetensors.

Key format: ``diffusion_model.{module}.lora_A/B.weight``.
Same ai-toolkit-compatible format used by the FLUX/zimage/krea2/ovis
families. No upstream LoRA loader mixin exists for DreamLite — this
format is the format of record (624 keys for the full checkpoint; pinned
by ``test_dreamlite_lora_portability.py``).
"""

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class DreamLiteSaver(GenericLoRASaver):
    """Saves DreamLite model weights as LoRA safetensors."""

    architecture_name = "dreamlite"
