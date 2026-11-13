"""Shared MiniMax-H3 code (no ``family.py`` and no definitions: not a family itself).

Consumed by the ``minimax_h3`` video family and the ``minimax_h3_image``
image family. Both models share a text encoder presentation, a VAE pixel
space, a flow-match convention and a native LoRA layout, so each piece lives
here once:

- :mod:`text` - the Qwen3-VL layer tap (raw tokens, no special tokens, one
  caption at a time), the text-cache key with its family prefix passed in,
  and the text-encoder release without a host copy.
- :mod:`pixel_adapter` - the ``[-1, 1]`` <-> ImageNet-space VAE wrapper.
- :mod:`flow` - the inverted flow-match convention (``t = 1 - sigma``,
  ``x_t = t*x0 + (1-t)*noise``, target ``v = x0 - noise``).
- :mod:`lora_artifact` - the PEFT-to-artifact LoRA core: QKV fusing, the
  ``diffusion_model.`` prefix and the scale folded into ``lora_B``; the
  module-name map is passed in by each family.

Nothing here names a family, a definition id or a checkpoint.
"""
