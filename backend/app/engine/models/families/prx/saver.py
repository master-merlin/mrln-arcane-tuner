"""PRX LoRA saver -- saves LoRA weights as safetensors.

Key format: ``diffusion_model.{module}.lora_A/B.weight``.
Same ai-toolkit-compatible format used by the FLUX families. No usable
upstream LoRA loader mixin exists for PRX (PRXPipeline's legacy
LoraLoaderMixin only targets unet/text_encoder) — these canonical keys are
the format of record (pinned by test_prx_lora_portability.py: 192 keys for
the 16-block sft checkpoint).
"""

from app.engine.models.families.prx_shared import PRXSharedLoRASaver


class PRXSaver(PRXSharedLoRASaver):
    """Saves latent-PRX model weights as LoRA safetensors."""

    architecture_name = "prx"
