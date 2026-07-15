"""Bernini-R model family (ByteDance) — renderer-only video-EDIT LoRA training.

Bernini-R is built entirely from stock Wan components (UMT5-xxl text encoder +
stock-key ``WanTransformer3DModel`` + Wan2.1 ``AutoencoderKLWan``). Its only
novelty is data-side conditioning: clean condition-video latents are
token-sequence concatenated with the noisy target latents, disambiguated by a
``source_id`` rotary phase, with the prediction consumed for target tokens only.
See ``vendor/transformer_forward.py`` for the packed forward adapter.
"""
