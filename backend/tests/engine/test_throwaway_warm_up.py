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
import pytest
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


# ── A cache-clearing expert swap is a re-warm event (both dual-expert drivers) ──────────────

SMALL_W, LARGE_W, REFERENCE_W = 8, 64, 32
# One latent width per accumulation window. The reference predicts LOW, so the first (LARGE,
# HIGH) window already swaps; every later change of width swaps again.
WINDOWS = [LARGE_W, SMALL_W, LARGE_W, LARGE_W, SMALL_W]


def _shape_router():
    """p_high is 1 for a latent wider than the reference, 0 below it: the width alone decides."""
    from app.engine.models.families.wan22.expert_router import ExpertRouter

    def draw(n: int, latents=None) -> torch.Tensor:
        wide = latents is not None and latents.shape[-1] > REFERENCE_W
        return torch.full((n,), 999.0 if wide else 0.0)

    return ExpertRouter(boundary=0.875, timestep_cfg={"timestep_sampling": "radc"}, seed=0, mc_samples=64,
                        timestep_draw=draw, reference_latent_shape=(1, 16, 1, REFERENCE_W, REFERENCE_W))


class _Expert(torch.nn.Module):
    """An expert whose ``.to()`` moves nothing: the swap's order is what's under test."""

    def __init__(self):
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def to(self, *args, **kwargs):  # noqa: A003
        return self


class _Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(()))


class _Wan22Defn:
    architecture_params = {"mode": "t2v", "te.max_length": 512, "moe.boundary_ratio": 0.875}
    lora_targetable_modules: list[str] = []
    defaults: dict = {"num_frames": 81}


class _BerniniDefn:
    architecture_params = {"mode": "t2v", "dual_expert": True, "switch_dit_boundary": 0.875}
    lora_targetable_modules: list[str] = []


def _wan22():
    from app.engine.models.families.wan22.driver import Wan22Driver
    from app.engine.models.families.wan22.trainer import Wan22Trainer

    return Wan22Driver(_Wan22Defn(), torch.device("cpu")), Wan22Trainer


def _bernini_r():
    from app.engine.models.families.bernini_r.driver import BerniniRDriver
    from app.engine.models.families.bernini_r.trainer import BerniniRTrainer

    return BerniniRDriver(_BerniniDefn(), torch.device("cpu")), BerniniRTrainer


DRIVERS = pytest.mark.parametrize("family", [_wan22, _bernini_r], ids=["wan22", "bernini_r"])


class _RoutedTrainer(_Trainer):
    """The real loop over a real dual-expert driver and the family trainer's own routing hooks.
    ``events`` records each ``_swap_to``, the ``empty_cache`` inside it, and each forward."""

    def __init__(self, tmp_path, *, family, swap_mode: str, warm_up: bool):
        super().__init__(tmp_path, warm_up=warm_up)
        self.config["max_train_steps"] = len(WINDOWS)
        self._model = _Scale()
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr=0.1, momentum=0.9)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)
        self.latent_manager.encode_and_cache_batch.side_effect = (
            lambda images, **kw: images.mean(dim=1, keepdim=True).unsqueeze(2).repeat(1, 16, 1, 1, 1))
        self.driver, trainer_cls = family()
        self.driver.transformer_high, self.driver.transformer_low = _Expert(), _Expert()
        self.driver.configure_swap_mode(swap_mode)
        self.driver.set_router(_shape_router())
        self.events: list[tuple[str, dict]] = []
        for name in ("_routing_driver", "routing_paused", "route_training_batch", "take_rewarm_request"):
            hook = getattr(trainer_cls, name, None)
            setattr(self, name, None if hook is None else hook.__get__(self))
        original = self.driver._swap_to

        def swap_to(expert):
            self.events.append(("swap", {"expert": expert}))
            with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
                    torch.cuda, "empty_cache", side_effect=lambda: self.events.append(("empty_cache", {}))):
                original(expert)

        self.driver._swap_to = swap_to

    def _iter_training_batches(self, batch_size):
        if self.warm_up:
            yield WarmUpBatch([{"id": "warm", "size": LARGE_W}])
        for w, size in enumerate(WINDOWS):
            for k in range(ACCUM):
                yield [{"id": f"{w}.{k}", "size": size}]

    def sample_timesteps(self, batch_size, latents):
        self.driver.route_for_batch(latents)  # as both families' sample_timesteps do
        return torch.rand(batch_size)

    def add_noise(self, latents, noise, timesteps):
        return latents + noise * timesteps.view(-1, *([1] * (latents.dim() - 1)))

    def forward_pass(self, noisy_input, timesteps, text_emb, batch):
        self.events.append(("forward", {
            "warm": batch["ids"][0] == "warm", "width": noisy_input.shape[-1],
            "expert": self.driver.active_expert, "paused": self.driver.router.paused}))
        return noisy_input * self._model.w


def _routed_run(tmp_path, family, *, swap_mode: str, warm_up: bool) -> _RoutedTrainer:
    t = _RoutedTrainer(tmp_path, family=family, swap_mode=swap_mode, warm_up=warm_up)
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    with patch("app.core.db.DatabaseEngine.get_instance", side_effect=RuntimeError("no db in test")):
        asyncio.run(t.train())
    t.after = (torch.rand(3), random.random(), float(np.random.rand()))
    return t


def _forwards(t: _RoutedTrainer, *, warm: bool) -> list[dict]:
    return [e for kind, e in t.events if kind == "forward" and e["warm"] == warm]


@DRIVERS
def test_after_every_cache_clearing_swap_the_next_forward_is_a_warm_up_of_the_largest_shape(tmp_path, family):
    t = _routed_run(tmp_path, family, swap_mode="swap", warm_up=True)
    clears = [i for i, (kind, _) in enumerate(t.events) if kind == "empty_cache"]
    assert len(clears) == 4, t.events  # into HIGH, LOW, HIGH, LOW
    for i in clears:
        swapped_to = t.events[i - 1][1]["expert"]
        following = next(e for kind, e in t.events[i + 1:] if kind == "forward")
        assert following["warm"] and following["width"] == LARGE_W, (i, t.events)
        assert following["expert"] == swapped_to and following["paused"], (i, t.events)
    training = _forwards(t, warm=False)
    assert [f["width"] for f in training] == [w for w in WINDOWS for _ in range(ACCUM)]
    assert all(f["expert"] == ("high" if f["width"] == LARGE_W else "low") for f in training), training
    assert not any(f["paused"] for f in training)
    assert t.global_step == len(WINDOWS) - 1  # the last step's index: no warm-up counted a step


@DRIVERS
@pytest.mark.parametrize("swap_mode", ["swap", "resident"])
def test_the_epoch_start_warm_up_runs_on_the_expert_the_first_window_trains(tmp_path, family, swap_mode):
    """The reference predicts LOW; the first window is LARGE (HIGH). The warm-up runs after the
    window's own batch has routed, so on HIGH, and the router decides from that batch alone."""
    t = _routed_run(tmp_path, family, swap_mode=swap_mode, warm_up=True)
    first = next(e for kind, e in t.events if kind == "forward")
    assert first["warm"] and first["expert"] == _forwards(t, warm=False)[0]["expert"] == "high", t.events


@DRIVERS
def test_resident_experts_warm_only_where_the_iterator_asks(tmp_path, family):
    t = _routed_run(tmp_path, family, swap_mode="resident", warm_up=True)
    assert not [kind for kind, _ in t.events if kind in ("swap", "empty_cache")]
    assert len(_forwards(t, warm=True)) == 1


@DRIVERS
def test_the_re_warms_leave_no_trace(tmp_path, family):
    """Warm-up off draws no warm-up batch, so it has nothing to re-warm with: the same swaps, no
    throwaway micro-step. Everything the training keeps must be the same."""
    with_warm = _routed_run(tmp_path / "a", family, swap_mode="swap", warm_up=True)
    without = _routed_run(tmp_path / "b", family, swap_mode="swap", warm_up=False)
    assert len(_forwards(with_warm, warm=True)) == 4 and not _forwards(without, warm=True)
    assert [e for e in with_warm.events if e[0] == "swap"] == [e for e in without.events if e[0] == "swap"]
    assert torch.equal(with_warm._model.w, without._model.w), "a warm-up changed the trained parameter"
    sa, sb = with_warm.optimizer.state_dict(), without.optimizer.state_dict()
    assert torch.equal(sa["state"][0]["momentum_buffer"], sb["state"][0]["momentum_buffer"])
    assert with_warm.lr_scheduler.state_dict() == without.lr_scheduler.state_dict()
    assert with_warm.global_step == without.global_step
    assert _logged(with_warm) == _logged(without)
    assert torch.equal(with_warm.after[0], without.after[0]) and with_warm.after[1:] == without.after[1:]
