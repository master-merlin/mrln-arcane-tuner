"""FLUX.1 LoRA saver — extract PEFT weights to ComfyUI-compatible format.

Output format: ``transformer.{diffusers_module}.lora_A/B.weight``
(raw PEFT lora_A/B, no Kohya conversion, no alpha keys).

We train against diffusers ``FluxTransformer2DModel`` and therefore export
diffusers module names (``transformer_blocks.*``, ``single_transformer_
blocks.*``). ComfyUI's Flux model is BFL-native (``double_blocks.*``/
``single_blocks.*``); its LoRA ``key_map`` maps diffusers-named LoRAs ONLY
via the ``transformer.<module>``/bare entries that
``comfy.utils.flux_to_diffusers`` registers
(``comfy/lora.py::model_lora_keys_unet``). The ``diffusion_model.`` prefix is
paired exclusively with BFL-native names, so the previous
``diffusion_model.<diffusers_module>`` output matched NOTHING and silently
applied a zero-effect LoRA in ComfyUI — same root cause as ovis_image (which
ComfyUI loads through this very Flux path). Pinned by
``test_flux1_lora_portability.py``.
"""

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class Flux1Saver(GenericLoRASaver):
    """Save FLUX.1 LoRA weights for ComfyUI inference.

    Overrides ``key_prefix`` to ``"transformer."`` so the shipped file loads
    through stock ComfyUI's Flux LoRA path (see module docstring).
    """

    architecture_name = "flux1"
    key_prefix = "transformer."
