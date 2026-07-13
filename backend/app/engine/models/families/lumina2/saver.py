"""Lumina2 LoRA saver — extract PEFT weights in diffusers-canonical format.

Output format: ``transformer.{diffusers_module}.lora_A/B.weight`` (raw PEFT
lora_A/B, no Kohya conversion, no alpha keys) — the SAME ``transformer.``
convention as ``flux1``/``ovis_image``/``chroma``.

ComfyUI route decision (evidence, verified against stock ComfyUI master via
a direct fetch of ``comfy/lora.py``/``comfy/utils.py``/``comfy/model_base.py``/
``comfy/supported_models.py``, 2026-07-13 — NOT summarized, the exact files
were downloaded and grepped):

1. ``comfy/lora.py::model_lora_keys_unet`` DOES carry a Lumina2-specific
   branch (line ~340)::

       if isinstance(model, comfy.model_base.Lumina2):
           diffusers_keys = comfy.utils.z_image_to_diffusers(
               model.model_config.unet_config, output_prefix="diffusion_model."
           )
           for k in diffusers_keys:
               if k.endswith(".weight"):
                   to = diffusers_keys[k]
                   key_lora = k[:-len(".weight")]
                   key_map["diffusion_model.{}".format(key_lora)] = to
                   key_map["transformer.{}".format(key_lora)] = to
                   key_map["lycoris_{}".format(key_lora.replace(".", "_"))] = to
                   key_map[key_lora] = to

   So a ``transformer.<key_lora>`` candidate IS registered — at first glance
   this looks like our exact convention. It is NOT.

2. ``comfy/utils.py::z_image_to_diffusers`` (the function that builds
   ``diffusers_keys`` above, verbatim, lines 755-819) constructs its LEFT-
   hand candidate key strings from a DIFFERENT module-naming convention than
   diffusers' own ``Lumina2Transformer2DModel``::

       k = "{}.attention.".format(prefix_from)              # NOT ".attn."
       key_map["{}to_q.{}".format(k, end)] = ...
       block_map = {
           "attention.norm_q.weight": "attention.q_norm.weight",
           "attention_norm1.weight": "attention_norm1.weight",   # NOT "norm1"
           "attention_norm2.weight": "attention_norm2.weight",   # NOT "norm2"
           "feed_forward.w1.weight": "feed_forward.w1.weight",   # NOT "linear_1"
           "feed_forward.w2.weight": "feed_forward.w2.weight",   # NOT "linear_2"
           "feed_forward.w3.weight": "feed_forward.w3.weight",   # NOT "linear_3"
           ...
       }

   This is the ORIGINAL Alpha-VLLM checkpoint's OWN native module naming
   (the repo also ships a non-diffusers ``consolidated.00-of-01.pth``
   checkpoint in this exact layout) — the function's name references
   "diffusers" because it is shared with Z-Image (an architecturally
   related, newer Tongyi model whose own native checkpoint format uses the
   SAME ``attention``/``feed_forward.w1-3``/``attention_normN`` naming), not
   because it matches HF diffusers' ``Lumina2Transformer2DModel``.

3. Live-instantiating ``diffusers.Lumina2Transformer2DModel`` and
   introspecting ``named_modules()`` (verified 2026-07-13, this repo's
   venv, diffusers 0.39.0) proves the ACTUAL module names are
   ``layers.N.attn.to_q/to_k/to_v/to_out.0``, ``layers.N.feed_forward.
   linear_1/linear_2/linear_3``, ``layers.N.norm1``, ``layers.N.norm2``,
   ``layers.N.ffn_norm1``, ``layers.N.ffn_norm2`` — none of which appear
   anywhere in ``z_image_to_diffusers``'s key_map (which expects
   ``attention.*``, ``feed_forward.w1/w2/w3``, ``attention_norm1/2``).

4. Net effect: NONE of the four key_map variants ComfyUI's Lumina2 branch
   registers (``diffusion_model.``, ``transformer.``, ``lycoris_``, bare)
   ever match a diffusers-native Lumina2 LoRA's real module names — the
   ``isinstance(model, comfy.model_base.Lumina2)`` branch exists for
   ComfyUI's OWN native/BFL-style Lumina2 model implementation (which loads
   the original ``consolidated.*.pth``-format checkpoint), not for this
   family's HF-diffusers-trained LoRA. This is a genuine stock-ComfyUI gap
   for the diffusers checkpoint route — not a bug in this saver (chroma1
   precedent: documented as a known upstream gap, same rigor).

5. Given that gap, this saver follows the brief's documented fallback:
   diffusers-canonical keys (``transformer.`` + module names exactly as
   PEFT/``named_modules()`` produce them) so ``Lumina2Pipeline.
   load_lora_weights()`` — ``Lumina2LoraLoaderMixin`` (``venv/Lib/site-
   packages/diffusers/loaders/lora_pipeline.py`` line 3851:
   ``_lora_loadable_modules = ["transformer"]``) — loads the file correctly.
   Pinned by ``test_lumina2_lora_portability.py``.
"""

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class Lumina2Saver(GenericLoRASaver):
    """Save Lumina-Image-2.0 LoRA weights for diffusers-native inference.

    Uses ``key_prefix = "transformer."`` — the diffusers-canonical PEFT
    convention (see module docstring for why stock ComfyUI's Lumina2
    isinstance branch does NOT map this format despite superficially
    registering a ``transformer.`` key variant).
    """

    architecture_name = "lumina2"
    key_prefix = "transformer."
