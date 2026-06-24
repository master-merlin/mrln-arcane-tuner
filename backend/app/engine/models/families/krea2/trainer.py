"""Krea2 Trainer — family-specific trainer for Krea-2 (Phase 2).

Phase 1 vendored the transformer + conditioning helpers and scaffolded the
family / loader / definitions.  Phase 2 (this commit) wires in Krea2Driver.

Driver, sampler, and saver are lazy-imported inside ``_setup_family`` so
that registry discovery (which merely imports this module) never trips on a
missing Phase 3+ module.
"""

from app.engine.core.pipeline import GenericTrainingPipeline


class Krea2Trainer(GenericTrainingPipeline):
    """Krea-2 LoRA trainer.

    28-layer MMDiT with Qwen3-VL 12-layer stacked text encoder,
    AutoencoderKLQwenImage VAE, and flow-matching noise schedule.
    """

    # -- Setup --

    def _setup_family(self) -> None:
        """Initialize Krea2-specific loader and driver.

        All family-specific classes are lazy-imported here so that registry
        discovery (which just imports this module) never trips on missing
        Phase 3+ modules.
        """
        from .loader import Krea2Loader  # noqa: PLC0415
        from .driver import Krea2Driver  # noqa: PLC0415

        self.loader = Krea2Loader(self.device)
        self.driver = Krea2Driver(self.definition, self.device)

    def _create_sampler(self):
        """Sampler arrives in Phase 3 — return None for now."""
        return None
