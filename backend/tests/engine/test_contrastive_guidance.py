"""Contrastive guidance, the shared loss (``core/contrastive_guidance.py`` in
``pipeline_train.forward_loss_step``).

Each step with contrastive guidance on: ``u = f(x_t, sigma, E_blank)`` under no_grad on the
grad pass's own ``noisy_input`` and timesteps; ``s_eff`` is ``s`` (``constant``) or
``1 + (s - 1) * sigma`` (``sigma``), and 1 for an item whose caption was dropped;
``y = u + s_eff * (v_data - u)`` in fp32, detached, replaces the target the family's own
``_compute_step_loss`` receives. What this file holds it to, through the real
``forward_loss_step`` of a stub family:

* ``y`` is the formula by hand, for both schedules; an item with ``s_eff == 1`` gets ``v_data``
  itself, bit for bit, even where ``u + (v_data - u)`` would round;
* the blank pass runs under no_grad, on the identical tensors, with the blank embedding, which
  is encoded once per job; one backward pass;
* ``s = 1`` trains exactly as without guidance, and so does a family with its own loss
  override (kandinsky5) that has no distillation block;
* the ``sigma`` schedule reads sigma through the family's ``noise_fraction`` (any clock), and is
  refused at setup for a family without one; a family whose prediction is no tensor of the
  target's shape is refused at its first step;
* the batch keeps its caption-dropout mask, so a dropped caption that kept a prefix or a
  persistent trigger word still counts as dropped.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from app.engine.core.contrastive_guidance import guidance_scales, guided_target
from app.engine.core.definitions import ModelDefinition, ResolvedDistillation
from app.engine.core.pipeline import GenericTrainingPipeline

LATENT = (2, 3, 4)  # [B, C, L]


class _Family(GenericTrainingPipeline):
    """A stub family on the 0-1000 clock: sigma = t / 1000. Its "model" adds a learnable
    offset to the noisy input, and the text embedding's mean."""

    def _setup_family(self):  # pragma: no cover - the tests build it by hand
        pass

    def prepare_latents_for_training(self, latents):
        return latents

    def prepare_noise_for_training(self, noise):
        return noise

    def sample_timesteps(self, batch_size, latents=None):
        return torch.full((batch_size,), 250.0)

    def add_noise(self, latents, noise, timesteps):
        sigma = (timesteps / 1000.0).reshape(-1, 1, 1)
        return (1 - sigma) * latents + sigma * noise

    def compute_target(self, latents, noise, timesteps):
        return latents - noise

    def forward_pass(self, noisy_input, timesteps, text_embeddings, batch):
        self.calls.append(
            {
                "noisy_input": noisy_input,
                "timesteps": timesteps,
                "text": text_embeddings,
                "grad": torch.is_grad_enabled(),
            }
        )
        return noisy_input * self.weight + text_embeddings.mean(dim=-1, keepdim=True)

    def encode_text(self, captions, dtype, batch=None):
        self.encoded.append(list(captions))
        return torch.stack([torch.full((LATENT[2],), float(len(c)) + 0.5) for c in captions]).unsqueeze(1)

    def noise_fraction(self, timesteps):
        return timesteps.float() / 1000.0


def _family(cls=_Family, distillation: ResolvedDistillation | None = None, **config: Any):
    trainer = object.__new__(cls)
    trainer.config = {"noise_offset": 0.0, **config}
    trainer.device = torch.device("cpu")
    trainer.autocast_dtype = torch.float32
    trainer.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    trainer.weight = torch.nn.Parameter(torch.tensor(1.5))
    trainer.calls = []
    trainer.encoded = []
    trainer.definition = None
    if distillation is not None:
        trainer.distillation = distillation
    return trainer


def _latents(seed: int = 0) -> torch.Tensor:
    return torch.randn(*LATENT, generator=torch.Generator().manual_seed(seed))


def _step(trainer, *, dropped=None, seed: int = 0):
    batch: dict[str, Any] = {"captions": ["a cat", "a dog"]}
    if dropped is not None:
        batch["caption_dropped"] = dropped
    text = trainer.encode_text(batch["captions"], torch.float32)
    trainer.encoded.pop()  # the captions' own encode; only the loop's blank encodes stay
    torch.manual_seed(seed)
    return trainer.forward_loss_step(_latents(), text, batch, 1)


def _cg(scale: float, schedule: str = "constant") -> ResolvedDistillation:
    return ResolvedDistillation("contrastive_guidance", scale, schedule)


# ── The target ───────────────────────────────────────────────────────────────


def test_the_scales_per_item():
    torch.testing.assert_close(guidance_scales(3.0, "constant", 2), torch.tensor([3.0, 3.0]))
    sigma = torch.tensor([0.25, 1.0])
    torch.testing.assert_close(guidance_scales(3.0, "sigma", 2, sigma=sigma), torch.tensor([1.5, 3.0]))
    torch.testing.assert_close(guidance_scales(3.0, "sigma", 2, sigma=torch.tensor([0.5])), torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(
        guidance_scales(3.0, "constant", 3, dropped=[False, True, False]), torch.tensor([3.0, 1.0, 3.0])
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"schedule": "sigma"}, "needs each item's sigma"),
        ({"schedule": "sigma", "sigma": torch.tensor([0.5, 1.5])}, "outside"),
        ({"schedule": "sigma", "sigma": torch.tensor([0.1, 0.2, 0.3])}, "3 sigmas"),
        ({"schedule": "constant", "dropped": [True]}, "dropout mask"),
        ({"schedule": "linear"}, "unknown schedule"),
    ],
)
def test_bad_scale_inputs_are_refused(kwargs, match):
    schedule = kwargs.pop("schedule")
    with pytest.raises(ValueError, match=match):
        guidance_scales(3.0, schedule, 2, **kwargs)


def test_the_target_is_the_formula_by_hand():
    u, v = _latents(1), _latents(2)
    s = torch.tensor([3.0, 1.5])
    y = guided_target(u, v, s)
    assert y.dtype == torch.float32 and not y.requires_grad
    for i in range(2):
        torch.testing.assert_close(y[i], u[i] + float(s[i]) * (v[i] - u[i]), atol=0, rtol=0)


def test_an_item_at_scale_one_gets_the_data_target_exactly_where_the_formula_would_round():
    """A large ``u`` and a small ``v``: ``v - u`` loses ``v``'s low bits in fp32, so
    ``u + (v - u)`` is not ``v``."""
    u = torch.full((2, 1, 4), 3.0e7)
    v = torch.tensor([0.1, 0.7, 1.3, 2.9]).reshape(1, 1, 4).expand(2, 1, 4).clone()
    naive = u + 1.0 * (v - u)
    assert not torch.equal(naive, v), "the inputs must expose the cancellation, or the test proves nothing"
    y = guided_target(u, v, torch.tensor([1.0, 2.0]))
    assert torch.equal(y[0], v[0])
    assert not torch.equal(y[1], v[1])


def test_a_blank_prediction_of_another_shape_is_refused():
    with pytest.raises(ValueError, match="blank prediction"):
        guided_target(torch.zeros(2, 3), torch.zeros(2, 4), torch.ones(2))


# ── Through the shared loop ──────────────────────────────────────────────────


def test_the_step_trains_toward_the_guided_target():
    trainer = _family(distillation=_cg(3.0))
    seen = {}
    loss_hook = trainer._compute_step_loss

    def spy(pred, target, timesteps, batch, grad_accum):
        seen["pred"], seen["target"] = pred, target
        return loss_hook(pred, target, timesteps, batch, grad_accum)

    trainer._compute_step_loss = spy
    step = _step(trainer)
    grad_call, blank_call = trainer.calls
    with torch.no_grad():
        u = blank_call["noisy_input"] * trainer.weight + blank_call["text"].mean(dim=-1, keepdim=True)
    v_data = step.prepared_latents - step.prepared_noise
    expected = u + 3.0 * (v_data - u)
    torch.testing.assert_close(seen["target"], expected, atol=1e-6, rtol=0)
    assert seen["pred"] is step.pred
    torch.testing.assert_close(step.loss, torch.nn.functional.mse_loss(step.pred, expected), atol=1e-6, rtol=0)


def test_the_blank_pass_runs_without_grad_on_the_identical_tensors_with_the_blank_prompt():
    trainer = _family(distillation=_cg(3.0))
    step = _step(trainer)
    assert len(trainer.calls) == 2
    grad_call, blank_call = trainer.calls
    assert grad_call["grad"] is True
    assert blank_call["grad"] is False, "the blank pass must not build a graph"
    assert blank_call["noisy_input"] is grad_call["noisy_input"] is step.noisy_input
    assert blank_call["timesteps"] is grad_call["timesteps"]
    assert torch.equal(blank_call["text"], trainer.encode_text(["", ""], torch.float32))
    assert not step.target.requires_grad
    step.loss.backward()  # one backward, through the grad pass only
    assert trainer.weight.grad is not None and float(trainer.weight.grad) != 0.0


def test_the_blank_embedding_is_encoded_once_per_job():
    trainer = _family(distillation=_cg(3.0))
    for seed in range(5):
        _step(trainer, seed=seed)
    assert [c for c in trainer.encoded if c == ["", ""]] == [["", ""]], trainer.encoded


def test_scale_one_trains_exactly_as_without_guidance():
    guided = _family(distillation=_cg(1.0))
    plain = _family(distillation=ResolvedDistillation("none", None, None))
    a, b = _step(guided), _step(plain)
    assert torch.equal(a.loss, b.loss)
    assert torch.equal(a.target, b.target)
    assert len(guided.calls) == 1, "no blank pass is needed when every item trains at scale 1"


def test_a_trainer_without_a_resolved_handling_trains_as_before():
    with_none = _family(distillation=ResolvedDistillation("none", None, None))
    bare = _family()  # setup() never ran: no ``distillation`` attribute at all
    assert torch.equal(_step(with_none).loss, _step(bare).loss)


def test_a_dropped_caption_trains_on_the_data_target_and_the_others_on_the_guided_one():
    trainer = _family(distillation=_cg(3.0))
    step = _step(trainer, dropped=[True, False])
    v_data = (step.prepared_latents - step.prepared_noise).float()
    assert torch.equal(step.target[0], v_data[0])
    assert not torch.allclose(step.target[1], v_data[1])


def test_the_sigma_schedule_reads_sigma_through_the_familys_clock():
    """A 0-1000 clock at t = 250 is sigma 0.25: s_eff = 1 + (3 - 1) * 0.25 = 1.5."""
    trainer = _family(distillation=_cg(3.0, "sigma"))
    step = _step(trainer)
    _, blank_call = trainer.calls
    with torch.no_grad():
        u = blank_call["noisy_input"] * trainer.weight + blank_call["text"].mean(dim=-1, keepdim=True)
    v_data = step.prepared_latents - step.prepared_noise
    torch.testing.assert_close(step.target, u + 1.5 * (v_data - u), atol=1e-6, rtol=0)


def test_a_prediction_that_is_not_a_tensor_of_the_targets_shape_is_refused_naming_the_family():
    class _Tuple(_Family):
        def forward_pass(self, noisy_input, timesteps, text_embeddings, batch):
            return (noisy_input, None)

    trainer = _family(_Tuple, distillation=_cg(3.0))
    with pytest.raises(ValueError, match="_Tuple"):
        _step(trainer)


# ── Setup ────────────────────────────────────────────────────────────────────


def _definition() -> ModelDefinition:
    return ModelDefinition(
        id="cg-test",
        family="sdxl",
        name="CG test",
        distillation={
            "step": True,
            "guidance": True,
            "guidance_scale": 3.5,
            "training_adapter": None,
            "default_handling": "contrastive_guidance",
        },
    )


def test_the_sigma_schedule_is_refused_at_setup_for_a_family_without_a_clock():
    class _NoClock(_Family):
        noise_fraction = GenericTrainingPipeline.noise_fraction

    trainer = _family(_NoClock, distillation_guidance_schedule="sigma")
    trainer.definition = _definition()
    with pytest.raises(ValueError, match="noise_fraction"):
        trainer._setup_distillation()
    trainer.config["distillation_guidance_schedule"] = "constant"
    trainer._setup_distillation()
    assert trainer.distillation == _cg(3.5, "constant")


def test_a_family_with_a_clock_takes_the_sigma_schedule_at_setup():
    trainer = _family(distillation_guidance_schedule="sigma")
    trainer.definition = _definition()
    trainer._setup_distillation()
    assert trainer.distillation == _cg(3.5, "sigma")


def test_the_base_pipeline_has_no_clock():
    with pytest.raises(NotImplementedError, match="noise_fraction"):
        GenericTrainingPipeline.noise_fraction(object.__new__(_Family), torch.zeros(1))


# ── A family with its own loss and no block (kandinsky5) ─────────────────────


def test_kandinsky5_computes_the_identical_loss_for_one_step():
    from app.engine.models.families.kandinsky5.trainer import Kandinsky5Trainer

    class _K5(Kandinsky5Trainer):
        prepare_latents_for_training = _Family.prepare_latents_for_training
        prepare_noise_for_training = _Family.prepare_noise_for_training
        sample_timesteps = _Family.sample_timesteps
        add_noise = _Family.add_noise
        compute_target = _Family.compute_target
        forward_pass = _Family.forward_pass
        encode_text = _Family.encode_text

    trainer = _family(_K5)
    trainer.driver = SimpleNamespace(_i2v_conditioning_engaged=lambda: True)  # its override slices frame 0
    trainer.definition = None
    trainer._setup_distillation()
    assert trainer.distillation.handling == "none"
    step = _step(trainer)
    expected = Kandinsky5Trainer._compute_step_loss(trainer, step.pred, step.target, step.timesteps, {}, 1)
    assert torch.equal(step.loss, expected)
    assert torch.equal(step.target, step.prepared_latents - step.prepared_noise)
    assert len(trainer.calls) == 1


# ── The caption-dropout mask ─────────────────────────────────────────────────


def _item(tmp_path, *, dropout: float, prefix: str = "") -> dict[str, Any]:
    return {
        "path": str(tmp_path / "x.png"),
        "id": "x.png",
        "caption": "a red fox",
        "dataset_name": "ds",
        "dataset_path": str(tmp_path),
        "prefix": prefix,
        "dropout_rate": dropout,
        "use_captions": True,
        "use_model_aware_captions": False,
        "cache_dir": str(tmp_path / "cache"),
        "is_video": False,
    }


@pytest.mark.parametrize(
    ("config", "prefix", "kept"),
    [
        ({"global_triggerword": "ohwx", "persist_triggerword_on_dropout": True}, "", "ohwx"),
        ({}, "photo", "photo"),
        ({}, "", ""),
    ],
    ids=["persistent-trigger", "prefix", "empty"],
)
def test_the_batch_keeps_which_captions_were_dropped(tmp_path, config, prefix, kept):
    trainer = _family(**config)
    batch = trainer._get_batch(
        [_item(tmp_path, dropout=1.0, prefix=prefix), _item(tmp_path, dropout=0.0, prefix=prefix)],
        decode_pixels=False,
    )
    assert batch["captions"][0] == kept
    assert batch["caption_dropped"] == [True, False]
