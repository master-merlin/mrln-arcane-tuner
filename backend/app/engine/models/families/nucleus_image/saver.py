"""Nucleus-Image LoRA saver — extract PEFT weights in diffusers-canonical
format.

Output format: ``transformer.{diffusers_module}.lora_A/B.weight`` (raw PEFT
lora_A/B, no Kohya conversion, no alpha keys) — the SAME ``transformer.``
convention as ``flux1``/``ovis_image``/``chroma``/``lumina2``.

ComfyUI + diffusers-mixin route decision (evidence, verified directly
against the installed diffusers 0.39.0 package AND a live fetch of stock
ComfyUI master's ``comfy/lora.py``/``comfy/supported_models.py``,
2026-07-13 — NOT summarized, the exact files/sources were read):

1. **``NucleusMoEImagePipeline`` has NO LoRA loader mixin at all** (confirmed
   by reading ``diffusers/pipelines/nucleusmoe_image/
   pipeline_nucleusmoe_image.py``: ``class NucleusMoEImagePipeline(
   DiffusionPipeline)`` — a single base class, no ``*LoraLoaderMixin``).
   Grepping ``diffusers/loaders/lora_pipeline.py`` for ``"Nucleus"`` returns
   ZERO matches — there is no ``NucleusMoEImageLoraLoaderMixin`` anywhere in
   diffusers 0.39.0. So ``pipe.load_lora_weights(...)`` does not exist for
   this family yet, unlike (e.g.) ``Lumina2LoraLoaderMixin`` for lumina2.

2. **The TRANSFORMER itself, however, IS directly LoRA-loadable.**
   ``NucleusMoEImageTransformer2DModel`` inherits ``PeftAdapterMixin``
   (confirmed: ``transformer_nucleusmoe_image.py`` line 727 class
   declaration). ``PeftAdapterMixin.load_lora_adapter`` (``diffusers/
   loaders/peft.py`` line 80) takes a ``prefix: str = "transformer"``
   argument and — read verbatim, lines 200-202 — filters AND STRIPS exactly
   that prefix from the incoming state dict::

       if prefix is not None:
           state_dict = {
               k.removeprefix(f"{prefix}."): v
               for k, v in state_dict.items() if k.startswith(f"{prefix}.")
           }

   This means a file saved with this saver's ``transformer.`` key
   convention is loadable TODAY, without any pipeline-level mixin, via::

       transformer.load_lora_adapter("nucleus_lora.safetensors")  # prefix="transformer" is the default

   i.e. the SAME ``transformer.`` key format used across this codebase
   (flux1/ovis_image/chroma/lumina2) is not just a stylistic convention
   here — it is the literal string the default ``prefix`` argument expects
   and strips. No saver-side change would be needed even if diffusers later
   adds a ``NucleusMoEImageLoraLoaderMixin`` (every prior pipeline-level
   mixin in this codebase's experience — e.g. lumina2's — uses the same
   ``_lora_loadable_modules = ["transformer"]`` convention).

3. **ComfyUI has ZERO Nucleus support at all** — a STRICTLY LARGER gap than
   lumina2's (which has a stub ``isinstance(model, comfy.model_base.
   Lumina2)`` branch in ``comfy/lora.py`` that simply doesn't match the
   diffusers-native module names). A live fetch of stock ComfyUI master's
   ``comfy/lora.py`` and ``comfy/supported_models.py`` (2026-07-13) found
   NO occurrence of "nucleus"/"Nucleus"/"NucleusMoE" (case-insensitive) in
   either file — no model class, no key-mapping branch, nothing. A LoRA
   saved by this saver cannot currently be loaded in ComfyUI at all; that
   will require either a future ComfyUI custom node or upstream ComfyUI
   support, not a saver-side fix. Documented honestly rather than papered
   over (chroma1/lumina2 precedent for known upstream gaps).

4. Given (1)-(3), this saver follows the codebase's established fallback:
   diffusers-canonical keys (``transformer.`` + module names exactly as
   PEFT/``named_modules()`` produce them) — directly consumable by
   ``NucleusMoEImageTransformer2DModel.load_lora_adapter()`` today, and
   forward-compatible with any future pipeline-level mixin. Pinned by
   ``test_nucleus_image_lora_portability.py``.
"""

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class NucleusImageSaver(GenericLoRASaver):
    """Save Nucleus-Image LoRA weights for diffusers-native inference.

    Uses ``key_prefix = "transformer."`` — the diffusers-canonical PEFT
    convention, and the literal string
    ``NucleusMoEImageTransformer2DModel.load_lora_adapter()``'s default
    ``prefix="transformer"`` argument strips (see module docstring for
    the full evidence-cited decision, including the ComfyUI gap).
    """

    architecture_name = "nucleus_image"
    key_prefix = "transformer."
