"""WAN 2.2 TI2V-5B trainer seams: per-step i2v gate + frame-0 loss exclusion.

The single ``mode: both`` definition needs the JOB config (``video_mode``,
``first_frame_conditioning_probability``) to decide i2v engagement per step —
config the driver doesn't have — so :meth:`Wan22Ti2v5bTrainer._attach_conditioning`
(ltx2 precedent) Bernoulli-gates ``driver._i2v_active`` each step. And because
the conditioning frame's flow-match target is a closed-form function of
``noisy`` alone (t=0 → the model was shown no noise to undo), it must be
EXCLUDED from the loss (:meth:`_compute_step_loss`) — an answer-leak class
identical in kind to ltx2/kandinsky5's first-frame-token exclusion.
"""

from __future__ import annotations

import torch

from app.engine.models.families.wan22_ti2v_5b.driver import Wan22Ti2v5bDriver
from app.engine.models.families.wan22_ti2v_5b.trainer import Wan22Ti2v5bTrainer

Z_DIM = 48


class _Defn:
    architecture_params = {"mode": "both", "te.max_length": 512}
    lora_targetable_modules: list[str] = []


def _trainer_shell(config: dict) -> Wan22Ti2v5bTrainer:
    t = object.__new__(Wan22Ti2v5bTrainer)
    t.device = torch.device("cpu")
    t.config = config
    t.driver = Wan22Ti2v5bDriver(_Defn(), torch.device("cpu"))
    return t


# ── _attach_conditioning: the per-step Bernoulli gate ───────────────────────


def test_attach_conditioning_t2v_run_never_engages():
    t = _trainer_shell({"video_mode": "t2v", "first_frame_conditioning_probability": 1.0})
    for _ in range(20):
        t._attach_conditioning({}, torch.randn(1, Z_DIM, 3, 4, 4))
        assert t.driver._i2v_active is False


def test_attach_conditioning_i2v_run_probability_one_always_engages():
    t = _trainer_shell({"video_mode": "i2v", "first_frame_conditioning_probability": 1.0})
    for _ in range(20):
        t._attach_conditioning({}, torch.randn(1, Z_DIM, 3, 4, 4))
        assert t.driver._i2v_active is True


def test_attach_conditioning_i2v_run_probability_zero_never_engages():
    t = _trainer_shell({"video_mode": "i2v", "first_frame_conditioning_probability": 0.0})
    for _ in range(20):
        t._attach_conditioning({}, torch.randn(1, Z_DIM, 3, 4, 4))
        assert t.driver._i2v_active is False


def test_attach_conditioning_default_probability_is_half():
    """Base default (no explicit config key) matches the documented 0.5 mix."""
    t = _trainer_shell({"video_mode": "i2v"})
    torch.manual_seed(0)
    hits = 0
    n = 400
    for _ in range(n):
        t._attach_conditioning({}, torch.randn(1, Z_DIM, 3, 4, 4))
        hits += int(t.driver._i2v_active)
    rate = hits / n
    assert 0.35 < rate < 0.65, f"expected ~0.5 engagement rate, got {rate}"


# ── _compute_step_loss: frame-0 tokens excluded only when engaged ──────────


def test_loss_not_engaged_matches_base_mse():
    t = _trainer_shell({"video_mode": "t2v"})
    t.driver._i2v_active = False
    pred = torch.randn(2, Z_DIM, 3, 4, 4)
    target = torch.randn(2, Z_DIM, 3, 4, 4)
    timesteps = torch.tensor([400.0, 600.0])

    loss = t._compute_step_loss(pred, target, timesteps, {}, grad_accum=1)
    expected = torch.nn.functional.mse_loss(pred.float(), target.float())
    assert torch.allclose(loss, expected)


def test_loss_engaged_excludes_frame_zero():
    t = _trainer_shell({"video_mode": "i2v"})
    t.driver._i2v_active = True
    pred = torch.randn(2, Z_DIM, 3, 4, 4)
    target = torch.randn(2, Z_DIM, 3, 4, 4)
    # Corrupt frame 0 badly — if it leaked into the loss this would dominate it.
    pred_corrupted = pred.clone()
    pred_corrupted[:, :, 0] = 999.0
    timesteps = torch.tensor([400.0, 600.0])

    loss_clean = t._compute_step_loss(pred, target, timesteps, {}, grad_accum=1)
    loss_corrupted = t._compute_step_loss(
        pred_corrupted, target, timesteps, {}, grad_accum=1
    )
    assert torch.allclose(loss_clean, loss_corrupted), (
        "frame-0 corruption must not affect the loss when i2v is engaged"
    )
    expected = torch.nn.functional.mse_loss(
        pred[:, :, 1:].float(), target[:, :, 1:].float()
    )
    assert torch.allclose(loss_clean, expected)


def test_loss_engaged_f1_still_includes_the_only_frame():
    """F=1 guard: with no frame to exclude, the still's loss is NOT dropped."""
    t = _trainer_shell({"video_mode": "i2v"})
    t.driver._i2v_active = True
    pred = torch.randn(2, Z_DIM, 1, 4, 4)
    target = torch.randn(2, Z_DIM, 1, 4, 4)
    timesteps = torch.tensor([400.0, 600.0])

    loss = t._compute_step_loss(pred, target, timesteps, {}, grad_accum=1)
    expected = torch.nn.functional.mse_loss(pred.float(), target.float())
    assert torch.allclose(loss, expected)


# ── End-to-end dispatch pin: _attach_conditioning -> add_noise -> forward_pass
# -> _compute_step_loss, exactly the order pipeline_train.py runs them ──────


def test_end_to_end_i2v_step_pins_frame_zero_clean_and_excludes_it_from_loss():
    t = _trainer_shell({"video_mode": "i2v", "first_frame_conditioning_probability": 1.0})

    class _Fake:
        def __call__(self, hidden_states, timestep, encoder_hidden_states,
                      encoder_hidden_states_image=None, return_dict=False):
            return (hidden_states.clone(),)  # identity "prediction"

    t.driver.transformer = _Fake()

    latents = torch.randn(1, Z_DIM, 3, 4, 4)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([500.0])
    text = torch.randn(1, 8, 16)

    batch: dict = {}
    t._attach_conditioning(batch, latents)
    assert t.driver._i2v_active is True

    noisy = t.driver.add_noise(latents, noise, timesteps)
    assert torch.equal(noisy[:, :, 0], latents[:, :, 0])  # frame 0 pinned clean

    pred = t.driver.forward_pass(noisy, timesteps, text, batch)
    target = noise - latents  # base compute_target formula

    # Frame 0's "prediction" (== noisy's frame 0 == latents' frame 0) is a
    # trivial identity, nowhere near noise-latents — proving it must be
    # excluded, then proving the trainer's loss actually excludes it.
    loss = t._compute_step_loss(pred, target, timesteps, batch, grad_accum=1)
    expected = torch.nn.functional.mse_loss(
        pred[:, :, 1:].float(), target[:, :, 1:].float()
    )
    assert torch.allclose(loss, expected)
