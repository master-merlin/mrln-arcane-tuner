"""OvisImageTrainer — family-specific trainer for Ovis-Image.

Driver, loader, sampler, and saver are lazy-imported inside the hooks so
that registry discovery (which merely imports this module) never trips on
a module still under construction.
"""

from __future__ import annotations

import structlog

from app.engine.core.pipeline import GenericTrainingPipeline

logger = structlog.get_logger(__name__)


class OvisImageTrainer(GenericTrainingPipeline):
    """Ovis-Image LoRA trainer.

    ~7.4B Flux-style MMDiT (6 double + 27 single blocks) with a single
    Qwen3 text encoder (hidden 2048), AutoencoderKL VAE, and
    flow-matching noise schedule.
    """

    # -- Setup --

    def _setup_family(self) -> None:
        """Initialize Ovis-Image-specific loader, driver, and saver."""
        from .loader import OvisImageLoader  # noqa: PLC0415
        from .driver import OvisImageDriver  # noqa: PLC0415
        from .saver import OvisImageSaver  # noqa: PLC0415

        self.loader = OvisImageLoader(self.device)
        self.driver = OvisImageDriver(self.definition, self.device)
        self.saver = OvisImageSaver()

    def _create_sampler(self):
        """Create an OvisImageSampler if sampling is configured."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import OvisImageSampler  # noqa: PLC0415

            return OvisImageSampler(self)
        return None
