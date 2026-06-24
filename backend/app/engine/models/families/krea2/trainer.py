"""Krea2 Trainer — minimal skeleton for family registration.

Phase 1: this module must import cleanly during registry discovery.
Driver, sampler, and saver arrive in Phases 2–3 and are lazy-imported
inside their respective methods to avoid circular/missing imports at
module load time.
"""

from app.engine.core.pipeline import GenericTrainingPipeline


class Krea2Trainer(GenericTrainingPipeline):
    """Krea-2 LoRA trainer skeleton.

    28-layer MMDiT with Qwen3-VL text encoder, AutoencoderKLQwenImage VAE,
    and flow-matching noise schedule.  Full driver/sampler/saver arrive in
    Phases 2–3.
    """

    # -- Setup --

    def _setup_family(self) -> None:
        """Initialize Krea2-specific loader, saver, and driver.

        Driver / sampler / saver are lazy-imported here so that registry
        discovery (which just imports this module) never trips on a missing
        Phase 2/3 module.
        """
        from .loader import Krea2Loader  # noqa: PLC0415
        self.loader = Krea2Loader(self.device)

        # Saver arrives in Phase 2/3 — raise informative error at runtime.
        raise NotImplementedError(
            "Krea2Trainer._setup_family: driver/saver arrive in Phase 2/3. "
            "Do not launch a real training run until those phases are merged."
        )

    def _create_sampler(self):
        """Sampler arrives in Phase 2/3."""
        return None
