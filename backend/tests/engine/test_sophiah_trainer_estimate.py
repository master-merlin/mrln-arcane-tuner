"""SophiaH through the real ``train()`` loop, which calls plain ``loss.backward()``.

No test here builds a graph: the trainer never does, and one test that passed
``create_graph=True`` once hid that SophiaH couldn't train at all.
"""
from __future__ import annotations

import pytest
import torch

from app.engine.core.pipeline import pipeline_train
from app.engine.factories.optimizer import OptimizerFactory
from tests.engine.test_nan_window_skip import _run, _ScriptedTrainer


class _Trainer(_ScriptedTrainer):
    """Real model, real optimizer, a fixed batch, and noise that reaches the loss."""

    def __init__(
        self, tmp_path, *, steps, period=3, grad_accum=1, optimizer_type="SophiaH",
        second_layer_from=None, num_samples=1,
    ):
        super().__init__(tmp_path, grad_accum=1, loss_plan=[None])
        self.config["max_train_steps"] = steps
        self.config["gradient_accumulation_steps"] = grad_accum
        torch.manual_seed(0)
        self._model = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Tanh(), torch.nn.Linear(4, 4)
        )
        self._second = torch.nn.Linear(4, 4)
        self._second_layer_from = second_layer_from
        self._latents = torch.randn(1, 4)
        self.latent_manager.encode_and_cache_batch.return_value = self._latents
        params = list(self._model.parameters()) + list(self._second.parameters())
        self.optimizer = OptimizerFactory.create(
            optimizer_type, params, lr=0.001,
            config={"sophia_update_period": period, "sophia_num_samples": num_samples},
        )
        self.lr_scheduler = None
        self.ema_handler = None
        self.scaler = _NoScaler()
        self.stream: list[float] = []

    def on_epoch_end(self, epoch):
        pass

    def _get_batch(self, batch_items, decode_pixels=True):
        self.stream.append(torch.rand(1).item())
        return super()._get_batch(batch_items, decode_pixels)

    def add_noise(self, latents, noise, timesteps):
        return latents + 0.1 * noise

    def forward_pass(self, noisy_input, timesteps, text_emb, batch):
        out = self._model(noisy_input)
        if self._second_layer_from is not None and self.global_step >= self._second_layer_from:
            out = out + self._second(noisy_input)
        return out

    def compute_target(self, latents, noise, timesteps):
        return torch.ones_like(latents)

    def _compute_step_loss(self, pred, target, timesteps, batch, grad_accum):
        return (pred - target).pow(2).mean() / grad_accum


class _NoScaler:
    def is_enabled(self):
        return False


class _Scaler:
    """Behaves as GradScaler does on the paths the loop uses, on the CPU."""

    def __init__(self, factor):
        self.factor = factor

    def is_enabled(self):
        return True

    def scale(self, loss):
        return loss * self.factor

    def get_scale(self):
        return self.factor

    def unscale_(self, optimizer):
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.div_(self.factor)

    def step(self, optimizer, **kwargs):
        optimizer.step(**kwargs)

    def update(self):
        pass


def _spy_on_estimates(monkeypatch, trainer, result=None):
    """Record each estimate's step and one call of its gradient function."""
    calls = []
    real = pipeline_train.estimate_hessian_diagonal

    def spy(params, grads, **kwargs):
        calls.append({"step": trainer.global_step, "grads": grads(), "params": list(params), **kwargs})
        return result(trainer.global_step) if result else real(params, grads, **kwargs)

    monkeypatch.setattr(pipeline_train, "estimate_hessian_diagonal", spy)
    return calls


def test_sophiah_trains_past_several_periods_on_a_plain_backward(tmp_path):
    t = _Trainer(tmp_path, steps=8, period=3)
    before = [p.detach().clone() for p in t._model.parameters()]

    _run(t)

    moments = [t.optimizer.state[p]["hessian_moment"] for p in t._model.parameters()]
    assert all(torch.isfinite(m).all() for m in moments)
    assert any(m.abs().sum() > 0 for m in moments), "no estimate reached the optimizer"
    assert any(not torch.equal(p, b) for p, b in zip(t._model.parameters(), before))


def test_the_estimate_runs_only_on_due_steps(tmp_path, monkeypatch):
    t = _Trainer(tmp_path, steps=8, period=3)
    calls = _spy_on_estimates(monkeypatch, t)
    _run(t)
    assert [c["step"] for c in calls] == [0, 3, 6]


def test_a_period_of_one_estimates_every_step(tmp_path, monkeypatch):
    t = _Trainer(tmp_path, steps=3, period=1)
    calls = _spy_on_estimates(monkeypatch, t)
    _run(t)
    assert [c["step"] for c in calls] == [0, 1, 2]


def test_the_samples_and_distribution_reach_the_estimate(tmp_path, monkeypatch):
    t = _Trainer(tmp_path, steps=1, period=1, num_samples=3)
    calls = _spy_on_estimates(monkeypatch, t)
    _run(t)
    assert calls[0]["num_samples"] == 3
    assert calls[0]["distribution"] == "gaussian"


def _gradient_seen(tmp_path, monkeypatch, *, grad_accum, scaler):
    t = _Trainer(tmp_path, steps=1, period=1, grad_accum=grad_accum)
    t.scaler = scaler
    t.add_noise = lambda latents, noise, timesteps: latents  # the same loss in each run
    calls = _spy_on_estimates(monkeypatch, t)
    _run(t)
    return calls[0]["grads"]


def test_the_gradient_is_of_the_plain_loss_under_accumulation_and_scaling(tmp_path, monkeypatch):
    """The Hessian must not scale with the accumulation or the AMP scaler's factor."""
    plain = _gradient_seen(tmp_path, monkeypatch, grad_accum=1, scaler=_NoScaler())
    accumulated = _gradient_seen(tmp_path, monkeypatch, grad_accum=4, scaler=_NoScaler())
    scaled = _gradient_seen(tmp_path, monkeypatch, grad_accum=1, scaler=_Scaler(1024.0))

    for reference, a, s in zip(plain, accumulated, scaled):
        assert reference.abs().sum() > 0
        assert torch.allclose(a, reference, rtol=1e-4, atol=1e-6)
        assert torch.allclose(s, reference, rtol=1e-4, atol=1e-6)


def test_the_estimate_leaves_the_window_s_gradients_alone(tmp_path, monkeypatch):
    t = _Trainer(tmp_path, steps=1, period=1)
    seen = {}
    real_step = t.optimizer.step

    def step(*args, **kwargs):
        seen["grads"] = [p.grad.clone() for p in t._model.parameters()]
        return real_step(*args, **kwargs)

    t.optimizer.step = step
    no_estimate = _Trainer(tmp_path, steps=1, period=1)
    no_estimate_seen = {}
    real_no_estimate_step = no_estimate.optimizer.step

    def no_estimate_step(*args, **kwargs):
        no_estimate_seen["grads"] = [p.grad.clone() for p in no_estimate._model.parameters()]
        return real_no_estimate_step(*args, **kwargs)

    no_estimate.optimizer.step = no_estimate_step
    torch.manual_seed(5)
    _run(t)
    monkeypatch.setattr(pipeline_train, "estimate_hessian_diagonal", lambda *a, **k: None)
    torch.manual_seed(5)
    _run(no_estimate)

    for a, b in zip(seen["grads"], no_estimate_seen["grads"]):
        assert torch.equal(a, b)


def test_a_non_finite_estimate_is_skipped_and_training_continues(tmp_path, monkeypatch):
    t = _Trainer(tmp_path, steps=4, period=3)
    _spy_on_estimates(monkeypatch, t, result=lambda _step: None)

    _run(t)

    warnings = [c.args[0] for c in t.logger.warning.call_args_list if c.args]
    assert warnings.count("sophiah_hessian_estimate_skipped") == 2


def test_the_training_stream_is_the_one_it_would_be_without_the_estimate(tmp_path, monkeypatch):
    with_estimate = _Trainer(tmp_path, steps=7, period=2)
    torch.manual_seed(9)
    _run(with_estimate)

    without = _Trainer(tmp_path, steps=7, period=2)
    monkeypatch.setattr(pipeline_train, "estimate_hessian_diagonal", lambda *a, **k: None)
    torch.manual_seed(9)
    _run(without)

    assert with_estimate.stream == without.stream


def test_the_sigma_histogram_does_not_count_the_estimate_s_passes(tmp_path, monkeypatch):
    from app.engine.strategies import sigma_tracker

    updates = []
    monkeypatch.setattr(
        sigma_tracker.SigmaTracker, "update", lambda self, timesteps: updates.append(1)
    )
    t = _Trainer(tmp_path, steps=4, period=1)
    _run(t)
    assert len(updates) == 4


def test_a_parameter_first_reached_later_is_left_alone_and_then_trains(tmp_path, monkeypatch):
    """A dual-expert family's inactive expert: no gradient yet, no perturbation, and
    no state left in a shape that breaks its first real step."""
    t = _Trainer(tmp_path, steps=8, period=2, second_layer_from=5)
    calls = _spy_on_estimates(monkeypatch, t)
    second = list(t._second.parameters())
    before = [p.detach().clone() for p in second]

    _run(t)

    early = calls[0]["params"]
    assert not any(p is q for p in early for q in second)
    assert any(p is q for p in calls[-1]["params"] for q in second)
    assert any(not torch.equal(p, b) for p, b in zip(second, before))


def test_a_resumed_run_estimates_on_its_next_due_step(tmp_path, monkeypatch):
    first = _Trainer(tmp_path, steps=4, period=3)
    _run(first)

    resumed = _Trainer(tmp_path, steps=8, period=3)
    resumed._model.load_state_dict(first._model.state_dict())
    resumed.optimizer.load_state_dict(first.optimizer.state_dict())
    resumed.global_step = 4
    calls = _spy_on_estimates(monkeypatch, resumed)

    _run(resumed)

    assert [c["step"] for c in calls] == [6]


def test_another_optimizer_never_estimates(tmp_path, monkeypatch):
    t = _Trainer(tmp_path, steps=4, optimizer_type="AdamW")
    calls = _spy_on_estimates(monkeypatch, t)
    _run(t)
    assert calls == []


def test_the_factory_keeps_the_library_from_estimating_on_its_own():
    """Its own estimate needs the graph the trainer doesn't build."""
    x = torch.randn(8, 4)
    model = torch.nn.Linear(4, 2)
    optimizer = OptimizerFactory.create(
        "SophiaH", model.parameters(), lr=0.001, config={"sophia_update_period": 2}
    )
    assert optimizer.hessian_period == 2

    for _ in range(6):
        optimizer.zero_grad()
        model(x).pow(2).mean().backward()
        optimizer.step()


def test_an_unreached_parameter_keeps_no_state_until_it_has_a_gradient(tmp_path):
    """State made early would stay on the device the parameter had then."""
    t = _Trainer(tmp_path, steps=4, period=2)
    _run(t)
    assert all(len(t.optimizer.state[p]) == 0 for p in t._second.parameters())


@pytest.mark.xdist_group("gpu")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_an_inactive_expert_moved_to_the_gpu_takes_its_first_step():
    """A dual-expert family swaps an expert in after estimates have already run."""
    active = torch.nn.Parameter(torch.ones(4, device="cuda"))
    inactive = torch.nn.Parameter(torch.ones(4))
    optimizer = OptimizerFactory.create(
        "SophiaH", [active, inactive], lr=0.001, config={"sophia_update_period": 1}
    )
    active.grad = torch.ones(4, device="cuda")
    optimizer.step(hessian=[torch.ones(4, device="cuda"), None])

    inactive.data = inactive.data.to("cuda")
    inactive.grad = torch.ones(4, device="cuda")
    active.grad = None
    before = inactive.detach().clone()
    optimizer.step()

    assert not torch.equal(inactive, before)
