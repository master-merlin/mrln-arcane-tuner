import importlib
from unittest.mock import MagicMock

import pytest
import torch
from app.engine.strategies.sigma_schedule import shifted_sigmas
from app.engine.strategies.sigma_tracker import SigmaTracker


class TestShiftedSigmas:
    def test_identity_when_shift_one(self):
        out = shifted_sigmas(10, 1.0)
        assert torch.allclose(out, torch.linspace(1.0, 0.0, 11))

    def test_endpoints_fixed(self):
        out = shifted_sigmas(8, 3.0)
        assert abs(float(out[0]) - 1.0) < 1e-6
        assert abs(float(out[-1]) - 0.0) < 1e-6

    def test_shift_gt_one_pushes_toward_high_noise(self):
        lin = shifted_sigmas(100, 1.0)
        sh = shifted_sigmas(100, 3.0)
        # At every interior point the shifted schedule sits ABOVE linear (more
        # high-noise mass) for s>1.
        assert (sh[1:-1] >= lin[1:-1] - 1e-6).all()
        assert float(sh[1:-1].mean()) > float(lin[1:-1].mean())

    def test_monotonic_descending(self):
        out = shifted_sigmas(50, 5.0)
        assert (out[:-1] - out[1:] >= -1e-6).all()


class TestSigmaTracker:
    def test_mean_and_bins(self):
        tr = SigmaTracker(bins=10)
        # timesteps in [0,1000]; a batch all at 0.9*1000 → decile index 8.
        tr.update(torch.full((100,), 900.0), scale=1000.0)
        s = tr.summary()
        assert s["n"] == 100
        assert abs(s["mean"] - 0.9) < 1e-3
        assert s["deciles"][8] == 1.0

    def test_empty_safe(self):
        assert SigmaTracker().summary()["n"] == 0


def _clock_trainer(family, trainer_name, driver_name, t):
    """A real family trainer and driver, with one training step's other hooks stubbed.

    Zero latents and unit noise: what ``add_noise`` returns is the noise fraction it applied.
    """
    root = f"app.engine.models.families.{family}"
    trainer_cls = getattr(importlib.import_module(f"{root}.trainer"), trainer_name)
    driver_cls = getattr(importlib.import_module(f"{root}.driver"), driver_name)

    definition = MagicMock()
    definition.family = family
    definition.lora_targetable_modules = []
    definition.architecture_params = {}

    trainer = trainer_cls.__new__(trainer_cls)
    trainer.driver = driver_cls(definition, torch.device("cpu"))
    trainer._sigma_tracker = SigmaTracker()
    trainer.make_training_noise = torch.ones_like
    trainer.prepare_latents_for_training = lambda latents: latents
    trainer.prepare_noise_for_training = lambda noise: noise
    trainer.sample_timesteps = lambda batch_size, latents: torch.full((batch_size,), t)
    trainer.forward_pass = lambda noisy, timesteps, text_emb, batch: noisy
    trainer._compute_step_loss = lambda pred, target, *rest: pred.mean()
    return trainer


CLOCKS = [
    ("boogu_image", "BooguImageTrainer", "BooguImageDriver", 0.75),
    ("omnigen2", "OmniGen2Trainer", "OmniGen2Driver", 0.75),
    ("minimax_h3", "MiniMaxH3Trainer", "MiniMaxH3Driver", 0.75),
    ("wan22", "Wan22Trainer", "Wan22Driver", 250.0),
]


class TestTrackerClock:
    @pytest.mark.parametrize("family,trainer_name,driver_name,t", CLOCKS)
    def test_the_mean_is_the_noise_fraction_add_noise_applies(
        self, family, trainer_name, driver_name, t
    ):
        trainer = _clock_trainer(family, trainer_name, driver_name, t)

        step = trainer.forward_loss_step(torch.zeros(2, 4, 8, 8), None, {}, 1)

        applied = float(step.noisy_input.mean())
        assert abs(applied - 0.25) < 1e-4
        assert abs(trainer._sigma_tracker.summary()["mean"] - applied) < 1e-4

    @pytest.mark.parametrize(
        "family,trainer_name",
        [
            ("boogu_image.trainer_edit", "BooguImageEditTrainer"),
            ("omnigen2.trainer_edit", "OmniGen2EditTrainer"),
        ],
    )
    def test_an_edit_trainer_has_its_family_s_clock(self, family, trainer_name):
        module = importlib.import_module(f"app.engine.models.families.{family}")
        assert getattr(module, trainer_name).timesteps_count_clean_fraction is True

    def test_the_base_clock_is_the_default(self):
        from app.engine.core.pipeline.pipeline_base import PipelineBaseMixin

        assert PipelineBaseMixin.timesteps_count_clean_fraction is False

    def test_a_trainer_on_the_train_mixin_alone_keeps_the_base_clock(self):
        from app.engine.core.pipeline.pipeline_train import PipelineTrainMixin

        class Bare(PipelineTrainMixin):
            make_training_noise = staticmethod(torch.ones_like)

            def prepare_latents_for_training(self, latents):
                return latents

            def prepare_noise_for_training(self, noise):
                return noise

            def sample_timesteps(self, batch_size, latents):
                return torch.full((batch_size,), 250.0)

            def add_noise(self, latents, noise, timesteps):
                return noise

            def forward_pass(self, noisy, timesteps, text_emb, batch):
                return noisy

            def compute_target(self, latents, noise, timesteps):
                return noise

            def _compute_step_loss(self, pred, target, *rest):
                return pred.mean()

        trainer = Bare()
        trainer._sigma_tracker = SigmaTracker()

        trainer.forward_loss_step(torch.zeros(2, 4, 8, 8), None, {}, 1)

        assert trainer._sigma_tracker.summary()["mean"] == 0.25
