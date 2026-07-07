"""Kandinsky 5.0 model family registration.

Kandinsky 5.0 is a flow-match video DiT (``Kandinsky5Transformer3DModel``,
diffusers 0.39) with a DUAL external text encoder — Qwen2.5-VL (last hidden
layer, chat-template crop) for sequence embeddings plus CLIP ViT-L pooled
embeddings into ``pooled_projections`` — and the HunyuanVideo VAE
(16 latent channels, spatial 8x, temporal 4x, scalar sf 0.476986).

Family-wide quirks (see the driver for the load-bearing details):

- **Channels-LAST latents** ``(B, F, H, W, C)`` at the transformer boundary —
  unlike every other family; the driver transposes at ``prepare_latents``.
- ``cu_seqlens`` (int32 cumulative lengths) instead of a padding mask for the
  Qwen text stream.
- ``visual_cond=True`` on BOTH shipped checkpoints (even the T2V Lite sft) —
  the input concat is ``[latents, visual_cond, mask]`` on the last dim.
"""

from app.engine.core.definitions import ModelFamily


class Kandinsky5Family(ModelFamily):
    """Kandinsky 5.0 (T2V Lite 2B / I2V Pro 19B) logic provider."""

    family_name = "kandinsky5"
    archetype = "latent_diffusion"

    # Merged into the capability descriptor by ``resolve_capabilities``.
    #  - is_video: batches are 5D video clips (data path collates [B,C,F,H,W]).
    #  - has_image_encoder: False — I2V conditions through the LATENT path
    #    (frame 0 = image latent + visual_cond concat), no CLIP-vision tower.
    #
    # NOTE: ``supports_train_te`` is intentionally NOT overridden (the
    # latent_diffusion archetype already defaults it False; the redundant
    # override is reserved for SDXL by a project invariant test).
    capability_overrides = {
        "is_video": True,
        "has_image_encoder": False,
        "native_fps": 24,
    }

    def get_trainer_class(self):
        """Both T2V and I2V share one trainer; the driver branches on the
        definition's ``mode`` architecture param internally."""
        from .trainer import Kandinsky5Trainer

        return Kandinsky5Trainer
