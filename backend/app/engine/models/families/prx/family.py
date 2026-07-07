"""PRX model family registration (latent-space Photoroom PRX)."""

from app.engine.core.definitions import ModelFamily


class PRXFamily(ModelFamily):
    """PRX implementation of the ModelFamily logic provider.

    ~1.2B cross-attention DiT (16 PRXBlocks, hidden 1792, fused QKV/KV
    projections) by Photoroom with a T5Gemma encoder TE (~2.6B) and the
    Flux-style 16-channel AutoencoderKL. Latent-space variant; transformer
    code shared with the future pixel-space sibling via ``prx_shared``.
    """

    family_name = "prx"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        # Lazy import: keeps family discovery lightweight.
        from .trainer import PRXTrainer

        return PRXTrainer
