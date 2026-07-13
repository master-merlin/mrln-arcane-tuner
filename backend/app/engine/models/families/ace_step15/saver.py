"""ACE-Step 1.5 LoRA saver.

Saved key format (ai-toolkit / ComfyUI-custom-node convention, via
``GenericLoRASaver``)::

    diffusion_model.layers.{i}.self_attn.to_q.lora_A/B.weight
    diffusion_model.layers.{i}.cross_attn.to_out.0.lora_A/B.weight

diffusers 0.39.0 ships NO ``AceStepLoraLoaderMixin`` (verified —
``AceStepPipeline.load_lora_weights`` does not exist), so there is no
diffusers-native LoRA format to translate to/from — unlike Kandinsky5's real
upstream mixin. The ai-toolkit-convention ``diffusion_model.`` prefix is the
most defensible portable choice: it is what the community ComfyUI training
node packs identified in the recon report (``Comfyui_SN_AceStepTrainer``,
``ComfyUI-FL-AceStep-Training``) both target, and it matches this house's
default ``GenericLoRASaver.key_prefix``.

ComfyUI LoRA *loading* is a documented, actively-developed feature for
ACE-Step 1.5 (see recon report §8) but was NOT live-verified against a real
trained checkpoint from this family (no GPU UAT in this task) — flagged
honestly in the C1 report rather than asserted.
"""

from __future__ import annotations

from app.engine.core.pipeline.saver_base import GenericLoRASaver


class AceStep15Saver(GenericLoRASaver):
    """Saves ACE-Step 1.5 LoRA weights as ai-toolkit-format safetensors."""

    architecture_name = "ace_step15"
