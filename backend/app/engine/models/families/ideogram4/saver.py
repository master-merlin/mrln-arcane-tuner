"""Ideogram 4 LoRA saver -- generic ComfyUI / ai-toolkit safetensors format."""
from __future__ import annotations

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class IdeogramV4Saver(GenericLoRASaver):
    """Save Ideogram 4 LoRA weights in the generic distribution format."""

    architecture_name = "ideogram4"
