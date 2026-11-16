"""The throwaway warm-up: one forward and backward on the largest bucket, and nothing kept.

A family that routes per optimizer step gets one bucket per accumulation window, so the VRAM
warm-up can't be a training window of the largest bucket (that would train it more often). It
is a throwaway batch the iterator marks (:class:`WarmUpBatch`): the REAL ``train()`` runs it
through the micro-step's own path, so the allocator reserves the peak, then discards it. This
file drives that loop with and without a warm-up and checks the warm-up leaves no trace:
parameters, optimizer state, step counters, logged losses and the random streams the training
draws use are the same, and the routing hook is held for the warm-up forward only.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from app.engine.core.pipeline.pipeline_train import PipelineTrainMixin, WarmUpBatch

STEPS, ACCUM = 3, 2


class _Trainer(PipelineTrainMixin):
    """The real loop, family hooks stubbed; the loss reads the training noise and a random
    augmentation draw, so a moved random stream changes the trained parameters."""

    def __init__(self, tmp_path, *, warm_up: bool):
        self.logger = MagicMock()
        self.config = {
            "max_train_steps": STEPS,
            "train_batch_size": 1,
            "gradient_accumulation_steps": ACCUM,
            "cache_latents": False,
            "save_every_n_steps": 0,
            "sample_before_training": False,
            "noise_offset": 0.0,
        }
        self.inventory = [{"id": i} for i in range(8)]
        self.device = torch.device("cpu")
        self.autocast_dtype = torch.float32
        self.use_amp = False
        self.scaler = torch.amp.GradScaler("cpu", enabled=False)
        torch.manual_seed(0)
        self._model = torch.nn.Linear(4, 4)
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr=0.1, momentum=0.9)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)
        self.ema_handler = MagicMock()
        self.global_step = 0
        self.sampler = None
        self._log_writer = None
        self._aug_h_flip = False
        self._aug_v_flip = False
        self.checkpoint_manager = MagicMock()
        self.checkpoint_manager.output_dir = str(tmp_path)
        self.logger_component = MagicMock()
        self.logger_component.last_step_time = time.time()
        self.latent_manager = MagicMock()
        self.latent_manager.encode_and_cache_batch.side_effect = lambda images, **kw: images.mean(dim=(2, 3))[:, :1].repeat(1, 4)
        self.warm_up = warm_up
        self.forwards: list[tuple[str, bool]] = []  # (batch id, routing held)
        self.routing_held = False
        self.held_count = 0

    @contextlib.contextmanager
    def routing_paused(self):
        self.routing_held = True
        self.held_count += 1
        try:
            yield
        finally:
            self.routing_held = False

    def _get_primary_model(self):
        return self._model

    def _iter_training_batches(self, batch_size):
        if self.warm_up:
            yield WarmUpBatch([{"id": "warm", "size": 16}])
        for i in range(STEPS * ACCUM):
            yield [{"id": str(i), "size": 8}]

    def _get_batch(self, batch_items, decode_pixels=True):
        size = batch_items[0]["size"]
        return {"images": torch.full((1, 3, size, size), float(size)), "ids": [batch_items[0]["id"]],
                "paths": ["x.png"], "captions": ["a cat"]}

    def _load_control_latents(self, batch):
        pass

    def _attach_conditioning(self, batch, latents):
        pass

    def encode_text(self, captions, dtype, batch=None):
        return torch.zeros(1, 4)

    def prepare_latents_for_training(self, latents):
        return latents

    def prepare_noise_for_training(self, noise):
        return noise

    def sample_timesteps(self, batch_size, latents):
        return torch.rand(batch_size) + random.random() + float(np.random.rand())

    def add_noise(self, latents, noise, timesteps):
        return latents + noise * timesteps[:, None]

    def forward_pass(self, noisy_input, timesteps, text_emb, batch):
        self.forwards.append((batch["ids"][0], self.routing_held))
        return self._model(noisy_input)

    def compute_target(self, latents, noise, timesteps):
        return torch.zeros_like(latents)

    def _compute_step_loss(self, pred, target, timesteps, batch, grad_accum):
        return (pred - target).pow(2).mean() / grad_accum

    def _build_trainable_components(self):
        return {}

    def get_te_cache(self):
        return None

    def _build_cache_manifest(self):
        return None


def _run(tmp_path, warm_up: bool) -> _Trainer:
    t = _Trainer(tmp_path, warm_up=warm_up)
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    with patch("app.core.db.DatabaseEngine.get_instance", side_effect=RuntimeError("no db in test")):
        asyncio.run(t.train())
    t.after = (torch.rand(3), random.random(), float(np.random.rand()))
    return t


def _logged(trainer: _Trainer) -> list:
    """Every logged step (step, loss, lr, extras) but the wall-clock throughput."""
    calls = []
    for call in trainer.logger_component.log_step.call_args_list:
        extra = {k: v for k, v in (call.kwargs.get("extra") or {}).items() if k != "samples_per_sec"}
        calls.append((call.args, extra))
    return calls


def test_the_warm_up_runs_first_through_the_model_with_routing_held(tmp_path):
    t = _run(tmp_path, warm_up=True)
    assert t.forwards[0] == ("warm", True), t.forwards
    assert [f for f in t.forwards[1:] if f[1]] == [], "routing held for a training batch"
    assert [i for i, _ in t.forwards[1:]] == [str(i) for i in range(STEPS * ACCUM)]
    assert t.held_count == 1


def test_the_warm_up_leaves_no_trace(tmp_path):
    with_warm, without = _run(tmp_path / "a", warm_up=True), _run(tmp_path / "b", warm_up=False)
    for a, b in zip(with_warm._model.parameters(), without._model.parameters(), strict=True):
        assert torch.equal(a, b), "the warm-up changed the trained parameters"
    sa, sb = with_warm.optimizer.state_dict(), without.optimizer.state_dict()
    assert sa["param_groups"] == sb["param_groups"]
    for key in sb["state"]:
        assert torch.equal(sa["state"][key]["momentum_buffer"], sb["state"][key]["momentum_buffer"])
    assert with_warm.lr_scheduler.state_dict() == without.lr_scheduler.state_dict()
    assert with_warm.global_step == without.global_step
    assert _logged(with_warm) == _logged(without), "the warm-up changed what was logged"
    assert torch.equal(with_warm.after[0], without.after[0]) and with_warm.after[1:] == without.after[1:], (
        "the warm-up moved a random stream the training draws use")
    # A warm-up gradient kept into the first step's accumulation would move the parameters
    # and the momentum above.


def test_the_warm_up_adds_nothing_to_the_sigma_statistics(tmp_path, monkeypatch):
    from app.engine.strategies import sigma_tracker

    updates: list[int] = []
    monkeypatch.setattr(sigma_tracker.SigmaTracker, "update",
                        lambda self, timesteps, **kwargs: updates.append(timesteps.numel()))
    _run(tmp_path / "a", warm_up=True)
    with_warm, updates[:] = list(updates), []
    _run(tmp_path / "b", warm_up=False)
    assert with_warm == updates == [1] * (STEPS * ACCUM)


def test_a_trainer_without_the_hook_runs_the_warm_up_too(tmp_path):
    """A family with no routing to hold (no ``routing_paused``) still warms and discards."""
    class _Plain(_Trainer):
        routing_paused = None

    t = _Plain(tmp_path / "plain", warm_up=True)
    with patch("app.core.db.DatabaseEngine.get_instance", side_effect=RuntimeError("no db in test")):
        asyncio.run(t.train())
    reference = _run(tmp_path / "reference", warm_up=False)
    assert t.forwards[0][0] == "warm" and t.global_step == reference.global_step
    assert [i for i, _ in t.forwards[1:]] == [i for i, _ in reference.forwards]
