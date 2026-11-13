"""MiniMax-H3-Image under the shared contrastive guidance, on CPU with a tiny DiT.

The family supplies only its clock: ``noise_fraction(t) = 1 - t`` (the transformer's
``t = 1 - sigma``, ``t = 1`` clean). With that:

* ``t = 0`` (pure noise) trains at ``s_eff = s`` and ``t = 1`` (clean) at ``s_eff = 1`` under
  the ``sigma`` schedule, the same ``s_eff`` a family on a 0-1000 clock gets at the same noise;
* the blank pass runs the family's own forward on the same noisy input and timestep, so the
  same per-step head, without grad, with the blank prompt;
* ``setup()`` takes the ``sigma`` schedule for this family, and ``s = 1`` trains bit for bit
  as without guidance.
"""

from __future__ import annotations

import asyncio
from typing import Any

import torch

from app.engine.core.contrastive_guidance import guidance_scales
from app.engine.core.definitions import ResolvedDistillation
from tests.engine.families.test_minimax_h3_image_trainer import (
    _HeadSpy,
    _latents,
    _text,
    _with_model,
    bare_trainer,
    stand_in_definition,
)
from tests.engine.families.test_minimax_h3_image_vendor import TRANSFORMER_TINY, _tiny_transformer

BLOCK = {
    "step": True,
    "guidance": True,
    "guidance_scale": 3.5,
    "training_adapter": None,
    "default_handling": "contrastive_guidance",
}


def _trainer(**config: Any):
    trainer = bare_trainer(stand_in_definition(distillation=BLOCK), timestep_sampling="fixed_sigmas", **config)
    asyncio.run(trainer.setup())
    _with_model(trainer, _tiny_transformer())
    trainer.text_cache[""] = torch.randn(2, TRANSFORMER_TINY["text_dim"], generator=torch.Generator().manual_seed(9))
    return trainer


def test_the_clock_gives_sigma():
    trainer = _trainer()
    torch.testing.assert_close(trainer.noise_fraction(torch.tensor([0.0, 0.25, 1.0])), torch.tensor([1.0, 0.75, 0.0]))


def test_noise_trains_at_s_and_clean_at_one_under_the_sigma_schedule():
    trainer = _trainer()
    s_eff = guidance_scales(3.5, "sigma", 2, sigma=trainer.noise_fraction(torch.tensor([0.0, 1.0])))
    torch.testing.assert_close(s_eff, torch.tensor([3.5, 1.0]))


def test_the_same_noise_on_another_clock_gives_the_same_scale():
    """A 0-1000 clock (sigma = t / 1000) at t = 750 is this family's t = 0.25: sigma 0.75."""
    trainer = _trainer()
    h3 = guidance_scales(3.5, "sigma", 1, sigma=trainer.noise_fraction(torch.tensor([0.25])))
    thousand = guidance_scales(3.5, "sigma", 1, sigma=torch.tensor([750.0]) / 1000.0)
    torch.testing.assert_close(h3, thousand)
    torch.testing.assert_close(h3, torch.tensor([1.0 + 2.5 * 0.75]))


def test_setup_takes_the_sigma_schedule_for_this_family():
    trainer = _trainer(distillation_guidance_schedule="sigma")
    assert trainer.distillation == ResolvedDistillation("contrastive_guidance", 3.5, "sigma")


def test_the_blank_pass_uses_the_same_head_and_timestep_without_grad():
    trainer = _trainer()
    spy = _HeadSpy(trainer.driver.transformer)
    calls = []
    forward = trainer.forward_pass

    def recording(noisy_input, timesteps, text, batch):
        calls.append({"noisy_input": noisy_input, "timesteps": timesteps, "text": text, "grad": torch.is_grad_enabled()})
        return forward(noisy_input, timesteps, text, batch)

    trainer.forward_pass = recording
    torch.manual_seed(0)
    step = trainer.forward_loss_step(_latents(2), _text(), {"captions": ["a", "b"], "caption_dropped": [False, False]}, 1)
    assert [c["grad"] for c in calls] == [True, False]
    assert calls[1]["noisy_input"] is calls[0]["noisy_input"] and calls[1]["timesteps"] is calls[0]["timesteps"]
    blank_rows = calls[1]["text"].embeddings
    assert torch.equal(blank_rows[0, :2].cpu(), trainer.text_cache[""].to(blank_rows.dtype))
    # Two items each pass: one head and one timestep for all four forwards.
    assert len(spy.calls) == 4
    assert len({c["head"] for c in spy.calls}) == 1
    assert all(torch.equal(c["timestep"], spy.calls[0]["timestep"]) for c in spy.calls)
    assert spy.calls[0]["head"] == trainer.driver.head_for_sigma(1.0 - float(step.timesteps[0]))
    v_data = step.prepared_latents - step.prepared_noise
    assert not torch.allclose(step.target, v_data.float()), "the guided target is the data target"
    step.loss.backward()


def test_scale_one_trains_bit_for_bit_as_without_guidance():
    guided = _trainer(distillation_guidance_scale=1.0)
    plain = _trainer(distillation_handling="none")
    plain.driver.transformer.load_state_dict(guided.driver.transformer.state_dict())
    results = []
    for trainer in (guided, plain):
        torch.manual_seed(0)
        results.append(trainer.forward_loss_step(_latents(2), _text(), {"captions": ["a", "b"]}, 1))
    assert torch.equal(results[0].loss, results[1].loss)
    assert torch.equal(results[0].target, results[1].target)
