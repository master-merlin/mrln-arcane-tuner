"""MiniMax-H3 model family registration.

H3 is a 33B dense single-stream DiT that denoises video and stereo audio
JOINTLY in one pass. Unlike LTX-2 — which has separate audio_attn / cross-modal
attention modules that can be adapted independently — H3's audio and video
tokens traverse the SAME attention and FFN blocks. There is therefore ONE LoRA
target set covering both modalities, and ``audio.loss_weight`` is the only
lever that separates them. See the driver for the targeting consequence.
"""

from app.engine.core.definitions import ModelFamily


class MiniMaxH3Family(ModelFamily):
    """MiniMax-H3 implementation of the ModelFamily logic provider."""

    family_name = "minimax_h3"
    archetype = "latent_diffusion"

    capability_overrides = {
        "is_video": True,
        "has_audio": True,
        # fl2va takes first/last frames; ref2va takes reference images/video.
        "has_image_encoder": True,
        # Single-stream — no MoE / dual-expert scheduling.
        "dual_expert": False,
        # NOTE: ``supports_train_te`` is NOT overridden — the ``latent_diffusion``
        # archetype default (False) already matches: the 48 GB Qwen3-VL TE must
        # never train. Only SDXL is allowed to put this key in
        # ``capability_overrides`` (enforced by
        # ``tests/test_family_archetypes.py::test_only_sdxl_overrides_train_te``);
        # repeating the archetype default here would trip that guard for no
        # behavioral change.
        "te_cache": True,
        "latent_cache": True,
        "supports_block_swap": True,
    }

    def get_trainer_class(self):
        # Lazy import: keeps family discovery lightweight (avoids pulling in
        # the full GenericTrainingPipeline / driver module graph just to
        # register the family).
        from .trainer import MiniMaxH3Trainer

        return MiniMaxH3Trainer
