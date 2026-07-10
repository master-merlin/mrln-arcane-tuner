"""Boogu-Image LoRA saver — saves LoRA weights as safetensors.

Key format: ``diffusion_model.{module}.lora_A/B.weight`` (house
ai-toolkit-compatible convention — same as krea2/dreamlite/etc). Because
Boogu's vendored ``BooguImageTransformer2DModel`` uses diffusers' own
``Attention`` class (separate ``to_q``/``to_k``/``to_v`` Linears, no fused
qkv), ``GenericLoRASaver``'s plain PEFT-module-path derivation already
produces the correct per-module keys for every curated
``lora_targetable_modules`` entry — no Boogu-specific override is needed
here. See ``lora_ecosystem.py`` for the (separate) ecosystem-portability
mapping to/from the non-diffusers fused-qkv convention that
``BooguImageLoraLoaderMixin`` expects.
"""

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class BooguImageSaver(GenericLoRASaver):
    """Saves Boogu-Image model weights as LoRA safetensors."""

    architecture_name = "boogu_image"
