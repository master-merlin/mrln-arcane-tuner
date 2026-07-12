"""Chroma LoRA saver — extract PEFT weights to a diffusers-portable format.

Output format: ``transformer.{diffusers_module}.lora_A/B.weight``
(raw PEFT lora_A/B, no Kohya conversion, no alpha keys) — the SAME
``transformer.`` convention as ``flux1``/``ovis_image``/``krea2``.

ComfyUI route decision (evidence, fetched live from
``github.com/comfyanonymous/ComfyUI`` main branch, 2026-07-13):

1. ``comfy/model_detection.py`` detects a Chroma checkpoint by the presence
   of ``distilled_guidance_layer.*`` keys and sets ``unet_config["image_model"]
   = "chroma"``; ``comfy/supported_models.py``'s ``Chroma`` class then builds
   a ``comfy.model_base.Chroma`` instance for it (a class DISTINCT from
   ``comfy.model_base.Flux``).
2. ``comfy/lora.py::model_lora_keys_unet`` — the function that builds the
   LoRA ``key_map`` ComfyUI's loaders actually use — has bespoke
   ``isinstance(model, comfy.model_base.X)`` branches for Flux, Ovis (via the
   Flux branch), Krea2, QwenImage, Lumina2/Z-Image, Kandinsky5, ErnieImage,
   HiDream, SD3, AuraFlow, PixArt, ... but **NO branch for
   ``comfy.model_base.Chroma`` at all**. The unconditional top-of-function
   ``comfy.utils.unet_to_diffusers(model.model_config.unet_config)`` call
   also contributes NOTHING for Chroma — that helper only maps classic
   UNet-style configs (it returns ``{}`` immediately when
   ``"num_res_blocks" not in unet_config``, which is true for any DiT config
   including Chroma's).
3. Net effect: **stock ComfyUI has no diffusers-format LoRA key mapping for
   Chroma at all.** The ONLY route that resolves is the function's generic
   top-level direct match against ``diffusion_model.<key>`` where ``<key>``
   is ComfyUI's OWN internal ``comfy.ldm.chroma.model`` module's parameter
   name — which is BFL-NATIVE (``double_blocks.*``/``single_blocks.*``,
   matching the community single-file ``Chroma1-HD.safetensors`` checkpoint
   layout), NOT the diffusers ``transformer_blocks.*``/
   ``single_transformer_blocks.*`` names our ``ChromaTransformer2DModel``-
   trained LoRA uses.

Given that gap, we still emit the diffusers/PEFT ``transformer.`` convention
(matching every other family in this codebase) rather than hand-rolling a
BFL-native key/tensor remapper (fusing separate ``to_q``/``to_k``/``to_v``
LoRA pairs into a single ``img_attn.qkv`` slice-compatible form is a much
larger, easy-to-get-subtly-wrong undertaking with no reference checkpoint to
verify against). This format IS verified to load: ``ChromaPipeline``
subclasses ``FluxLoraLoaderMixin`` (``pipeline_chroma.py`` line 153), whose
``load_lora_weights()`` expects exactly this ``transformer.<module>.lora_A/
B.weight`` PEFT convention — so the saved file loads correctly via
``pipe.load_lora_weights(...)`` in diffusers-native tooling (our own
inference path included). **Stock ComfyUI's native UNETLoader + LoraLoader
nodes will NOT auto-apply this file for Chroma** (no key_map entries exist)
— unlike the flux1/ovis fix this saver's format does NOT resolve ComfyUI
compatibility for Chroma; that would require either a BFL-native re-export
(future work) or a ComfyUI-side ``comfy.model_base.Chroma`` branch in
``model_lora_keys_unet`` (upstream fix, not ours to make). Pinned by
``test_chroma_lora_portability.py``.
"""

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class ChromaSaver(GenericLoRASaver):
    """Save Chroma LoRA weights in the diffusers/PEFT ``transformer.`` format.

    See module docstring for the ComfyUI-compatibility caveat (Chroma has no
    diffusers-key mapping in stock ComfyUI today).
    """

    architecture_name = "chroma"
    key_prefix = "transformer."
