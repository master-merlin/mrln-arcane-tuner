"""PRX Pixel model family registration (pixel-space Photoroom PRXPixel)."""

from app.engine.core.definitions import ModelFamily


class PRXPixelFamily(ModelFamily):
    """PRX Pixel implementation of the ModelFamily logic provider.

    ~7B pixel-space cross-attention DiT (24 PRXBlocks, hidden 3584,
    bottleneck img_in, resolution embeds) by Photoroom with a Qwen3-VL
    text backbone TE (~1.7B) and NO VAE — the transformer denoises raw
    RGB and predicts the clean image x0. Transformer-level code shared
    with the latent ``prx`` sibling via ``prx_shared``.
    """

    family_name = "prx_pixel"
    archetype = "pixel_transformer"

    def get_trainer_class(self):
        # Lazy import: keeps family discovery lightweight.
        from .trainer import PRXPixelTrainer

        return PRXPixelTrainer
