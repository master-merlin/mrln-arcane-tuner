"""FLUX.1 LoRA saver — extract PEFT weights to ComfyUI-compatible format.

Output format: ``diffusion_model.{diffusers_key}.lora_A/B.weight``
Uses raw ai-toolkit format (no Kohya conversion, no alpha keys).
"""

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class Flux1Saver(GenericLoRASaver):
    """Save FLUX.1 LoRA weights for ComfyUI inference."""

    architecture_name = "flux1"
