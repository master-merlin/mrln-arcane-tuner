"""ACE-Step 1.5 loader — manifest-driven via GenericComponentLoader.

Components (diffusers-format repo — verified against
``ACE-Step/acestep-v15-xl-turbo-diffusers``'s ``model_index.json``, which
ships ``model_index.json`` + one subfolder per module, so the full pipeline is
``from_pretrained``-loadable in a single call; see the driver module
docstring for how this checkpoint choice was verified against the installed
diffusers 0.39.0 package):

- ``tokenizer``          : ``AutoTokenizer`` (Qwen3 tokenizer)
- ``text_encoder``       : ``Qwen3Model`` — Qwen3-Embedding-0.6B's checkpoint
  declares ``architectures: ["Qwen3ForCausalLM"]`` (it is a causal-LM
  checkpoint repurposed for embeddings), but only the bare backbone
  (``last_hidden_state``) is ever used — the diffusers pipeline's own
  ``encode_prompt`` calls ``self.text_encoder(input_ids=...).last_hidden_state``
  with no LM head. Declaring the bare ``Qwen3Model`` class (what
  ``AutoModel.from_pretrained`` would resolve to for ``model_type: "qwen3"``
  anyway) pins the exact class for the TE loading-contract test instead of
  leaving it underspecified behind ``AutoModel`` — full contextual encoding
  for the text/caption prompt; ALSO reused for lyrics via its
  embedding-lookup layer only, see ``driver.encode_condition``.
- ``vae``                : ``AutoencoderOobleck`` (48kHz stereo, 25Hz latent
  rate) — kept fp32 for encode/decode precision, matching the video-VAE policy
  used elsewhere (WAN/LTX-2)
- ``condition_encoder``  : ``AceStepConditionEncoder`` — folds text + lyric +
  timbre embeddings into the final cross-attention conditioning sequence
  (lives in ``diffusers.pipelines.ace_step.modeling_ace_step``, not the
  top-level ``diffusers`` namespace)
- ``unet``               : ``AceStepTransformer1DModel`` (the DiT decoder,
  mapped to the house-standard ``"unet"`` component key)

Deliberately EXCLUDED from the manifest:
- ``scheduler`` (``FlowMatchEulerDiscreteScheduler``) — the trainer drives its
  own flow-match timestep sampling/noise-blend via the shared
  ``TimestepSampler``/``NoiseInterpolation`` components (see the driver
  module docstring); loading the scheduler object would be dead weight.
- ``audio_tokenizer`` / ``audio_token_detokenizer`` (``AceStepAudioTokenizer``
  / ``AceStepAudioTokenDetokenizer``) — the FSQ semantic-code codec used ONLY
  for the ``cover``/``lego`` audio-to-audio tasks (see recon report §1). Plain
  text2music LoRA training never touches them.
"""

from __future__ import annotations

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader


class AceStep15Loader(GenericComponentLoader):
    """Load ACE-Step 1.5 components from a diffusers-format repository."""

    def get_component_manifest(
        self, definition: ModelDefinition
    ) -> list[ComponentSpec]:
        return [
            # -- Tokenizer (Qwen3) --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoTokenizer",
                subfolder="tokenizer",
                candidates=["tokenizer"],
                is_torch_model=False,
                fallback_to_root=True,
            ),
            # -- Text encoder (Qwen3-Embedding-0.6B, bare backbone) --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Qwen3Model",
                subfolder="text_encoder",
                candidates=["text_encoder"],
                fallback_to_root=True,
            ),
            # -- VAE (AutoencoderOobleck, 48kHz stereo) — fp32 precision --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderOobleck",
                subfolder="vae",
                candidates=["vae"],
                dtype_override=torch.float32,
                fallback_to_root=True,
            ),
            # -- Condition encoder (text+lyric+timbre -> cross-attn sequence) --
            ComponentSpec(
                key="condition_encoder",
                hf_class="diffusers.pipelines.ace_step.modeling_ace_step.AceStepConditionEncoder",
                subfolder="condition_encoder",
                candidates=["condition_encoder"],
                fallback_to_root=True,
            ),
            # -- Transformer (DiT decoder) -> mapped to "unet" --
            ComponentSpec(
                key="unet",
                hf_class="diffusers.AceStepTransformer1DModel",
                subfolder="transformer",
                candidates=["transformer"],
                fallback_to_root=True,
            ),
        ]
