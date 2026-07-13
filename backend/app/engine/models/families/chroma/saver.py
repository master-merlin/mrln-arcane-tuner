"""Chroma LoRA saver — extract PEFT weights to ComfyUI-compatible format.

Output format: ``transformer.{diffusers_module}.lora_A/B.weight``
(raw PEFT lora_A/B, no Kohya conversion, no alpha keys) — the SAME
``transformer.`` convention as ``flux1``/``ovis_image``/``krea2``.

ComfyUI route decision (evidence, verified against stock ComfyUI master,
2026-07-13):

1. ``comfy/model_detection.py`` detects a Chroma checkpoint inside the Flux
   detection block by the presence of ``distilled_guidance_layer.*`` keys
   (line 291) and sets ``image_model = "chroma"`` — the surrounding Flux
   block ALSO sets ``hidden_size`` / ``depth`` / ``depth_single_blocks``
   (lines 272-289) from the checkpoint, so Chroma's ``unet_config`` carries
   all three (3072 / 19 / 38 for the real checkpoints).
2. ``comfy/model_base.py`` line 2134: ``class Chroma(Flux):`` — ComfyUI's
   Chroma model class SUBCLASSES ``comfy.model_base.Flux``. Therefore the
   ``isinstance(model, comfy.model_base.Flux)`` branch in
   ``comfy/lora.py::model_lora_keys_unet`` FIRES for Chroma, runs
   ``comfy.utils.flux_to_diffusers(unet_config)`` (which reads exactly the
   depth/depth_single_blocks/hidden_size keys set above) and registers
   ``key_map`` entries keyed ``transformer.<diffusers_module>`` (plus
   lycoris/onetrainer/DiffSynth variants) — the same route that maps
   flux1's and ovis_image's LoRAs.
3. Net effect: our ``transformer.{diffusers_module}.lora_A/B.weight`` keys
   are EXACTLY what stock ComfyUI's Chroma LoRA path maps — the file loads
   and applies through the native UNETLoader + LoraLoader nodes, same as
   flux1/ovis. As with those families, the ``diffusion_model.`` prefix is
   paired only with BFL-native ``double_blocks.*``/``single_blocks.*``
   names via the generic block — never emit it with diffusers names.

The same file also loads in diffusers-native tooling: ``ChromaPipeline``
subclasses ``FluxLoraLoaderMixin`` (``pipeline_chroma.py`` line 153), whose
``load_lora_weights()`` expects this ``transformer.<module>.lora_A/B.weight``
PEFT convention. Pinned by ``test_chroma_lora_portability.py``.
"""

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class ChromaSaver(GenericLoRASaver):
    """Save Chroma LoRA weights for ComfyUI + diffusers inference.

    Overrides ``key_prefix`` to ``"transformer."`` so the shipped file loads
    through stock ComfyUI's Flux LoRA path (Chroma subclasses Flux there —
    see module docstring) and diffusers' ``FluxLoraLoaderMixin``.
    """

    architecture_name = "chroma"
    key_prefix = "transformer."
