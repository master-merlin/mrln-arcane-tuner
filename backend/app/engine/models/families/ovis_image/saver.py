"""Ovis-Image LoRA saver — saves LoRA weights as safetensors.

Key format: ``transformer.{module}.lora_A/B.weight``.

Ovis is a Flux-architecture MMDiT. ComfyUI detects and loads it as a
``comfy.model_base.Flux`` model (``comfy/model_detection.py``: the
``double_blocks.*.img_mlp.gate_proj`` + ``txt_norm`` branch, ``# Ovis
model``). Its LoRA ``key_map`` is therefore built by the Flux handler in
``comfy/lora.py::model_lora_keys_unet`` from ``comfy.utils.flux_to_diffusers``,
which registers entries keyed ``transformer.<diffusers_module>`` (and bare
``<diffusers_module>``) — NOT ``diffusion_model.<module>`` with diffusers
module names. ComfyUI pairs the ``diffusion_model.`` prefix with BFL-native
``double_blocks.*``/``single_blocks.*`` names via its generic block, so our
diffusers ``transformer_blocks.*`` names under ``diffusion_model.`` matched
nothing and applied a silent zero-effect LoRA (the UAT bug). ``transformer.``
is the diffusers/PEFT/SimpleTuner prefix ComfyUI's ``flux_to_diffusers`` route
maps onto every one of our 207 curated modules.

Format of record: ``transformer.{module}.lora_A/B.weight`` (414 keys for the
full checkpoint; pinned by ``test_ovis_image_lora_portability.py``).
"""

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class OvisImageSaver(GenericLoRASaver):
    """Saves Ovis-Image model weights as LoRA safetensors.

    Overrides ``key_prefix`` to ``"transformer."`` so the shipped file loads
    in stock ComfyUI's Flux LoRA path (see module docstring).
    """

    architecture_name = "ovis_image"
    key_prefix = "transformer."
