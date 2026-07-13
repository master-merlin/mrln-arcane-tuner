"""OmniGen2 LoRA saver — diffusers-canonical PEFT keys, upstream-loadable.

Output format: ``transformer.{module_path}.lora_A/B.weight`` (raw PEFT
lora_A/B, no Kohya conversion, no alpha keys) — the same ``transformer.``
convention as flux1/ovis_image/chroma/lumina2.

Format decision (evidence, fetched 2026-07-13):

1. **Primary target — the upstream ``omnigen2`` inference stack** (the
   controller-designated consumer): ``VectorSpaceLab/OmniGen2``'s
   ``omnigen2/pipelines/lora_pipeline.py::OmniGen2LoraLoaderMixin`` is a
   verbatim CogVideoX-template diffusers LoRA mixin —
   ``_lora_loadable_modules = ["transformer"]`` / ``transformer_name =
   "transformer"`` (L52-60), ``save_lora_weights`` packs via
   ``cls.pack_weights(transformer_lora_layers, cls.transformer_name)``
   (L308-312, i.e. ``transformer.``-prefixed keys), and
   ``load_lora_weights`` -> ``load_lora_into_transformer`` consumes exactly
   that prefix. A file in this saver's format round-trips through
   ``OmniGen2Pipeline.load_lora_weights()`` unmodified. (The mixin's
   ``lora_state_dict`` additionally accepts ``diffusion_model.``-prefixed
   files via ``_convert_non_diffusers_lumina2_lora_to_diffusers``, L157-160
   — a fallback we don't need to emit.)

2. **ComfyUI status (honest gap note)**: stock ComfyUI DOES ship native
   OmniGen2 support (``comfy/ldm/omnigen/omnigen2.py`` mirrors the
   diffusers module names 1:1 — ``layers.N.attn.to_q`` etc., verified by
   direct fetch), and ``comfy/lora.py::model_lora_keys_unet`` carries an
   Omnigen2 branch (L313-317)::

       if isinstance(model, comfy.model_base.Omnigen2):
           for k in sdk:
               if k.startswith("diffusion_model.") and k.endswith(".weight"):
                   key_lora = k[len("diffusion_model."):-len(".weight")]
                   key_map["{}".format(key_lora)] = k

   — it registers ONLY the BARE module-path key form (``layers.0.attn.
   to_q.lora_A.weight``), with no ``transformer.`` and no
   ``diffusion_model.`` LoRA-file variant (contrast the QwenImage branch
   directly below it, which registers all three). So this saver's
   ``transformer.``-prefixed file does NOT load in stock ComfyUI as-is; a
   one-line key rename (strip the ``transformer.`` prefix) makes it load.
   Given the controller's explicit directive ("match the LoRA format the
   UPSTREAM omnigen2 package consumes"), upstream-loadability wins and the
   ComfyUI mismatch is documented rather than papered over with a
   double-emit. Pinned by ``test_omnigen2_lora_portability.py``.

Note the target-module naming: our LoRA surface (attention + feed-forward
Linears) is a superset of upstream's own LoRA recipe (train.py L262 targets
attention only) — upstream's PEFT-based loader handles any subset of
``transformer.*`` keys, so the extra feed-forward adapters load fine there.
"""

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class OmniGen2Saver(GenericLoRASaver):
    """Save OmniGen2 LoRA weights for the upstream diffusers-style stack.

    Uses ``key_prefix = "transformer."`` — the prefix
    ``OmniGen2LoraLoaderMixin.load_lora_weights()`` consumes (see module
    docstring for the ComfyUI bare-key gap note).
    """

    architecture_name = "omnigen2"
    key_prefix = "transformer."
