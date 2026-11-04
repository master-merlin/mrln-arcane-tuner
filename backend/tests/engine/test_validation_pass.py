"""The validation pass: comparable numbers, and no trace in the run it measures."""
import asyncio
import functools
import json
import random
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from app.engine.core.pipeline import GenericTrainingPipeline, validation_pass
from app.engine.core.pipeline.pipeline_train import PipelineTrainMixin
from app.engine.core.pipeline.validation_split import split_holdout
from app.engine.models.registry import ModelRegistry

LATENT = 4
BANDS = 4


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(LATENT, LATENT)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        return self.dropout(self.linear(x))


class _Latents:
    """Latents derived from the image id, with no cache on disk."""

    def load_cached_latents(self, ids, cache_dirs, source_paths=None, device=None, dtype=None, **_):
        seed = sum(ord(c) for c in ids[0])
        return torch.ones(1, LATENT, dtype=dtype, device=device) * (seed % 7 + 1) / 8.0

    def encode_and_cache_batch(self, *_args, **_kwargs):
        raise AssertionError("the pass must read the cache")


class _Pipeline(GenericTrainingPipeline):
    family_name = "stub"
    validation_timestep_domain = (0.0, 1000.0)

    def _setup_family(self):  # pragma: no cover - never called by the shell
        pass

    def add_noise(self, latents, noise, timesteps):
        sigma = (timesteps / 1000.0).reshape(-1, *[1] * (latents.dim() - 1)).to(latents)
        return (1 - sigma) * latents + sigma * noise

    def prepare_latents_for_training(self, latents):
        return latents

    def prepare_noise_for_training(self, noise):
        return noise

    def forward_pass(self, noisy_input, timesteps, text_emb, batch):
        self.forwards.append((batch["ids"][0], float(timesteps[0])))
        return self._model(noisy_input)

    def compute_target(self, latents, noise, timesteps):
        return noise - latents

    def encode_text(self, captions, dtype, batch=None):
        self.captions.append(captions[0])
        return None

    def sample_timesteps(self, batch_size, latents=None):
        raise AssertionError("the pass must not draw from the configured sampler")

    def _compute_step_loss(self, pred, target, timesteps, batch, grad_accum):
        raise AssertionError("the pass must not apply the family's loss")

    def _get_primary_model(self):
        return self._model

    def build_batch_extra(self, items):
        return {}


def _items(tmp_path, count=6, repeats=1):
    items = []
    for i in range(count):
        path = tmp_path / f"img{i}.png"
        pattern = np.random.default_rng(i).integers(0, 255, (9, 9, 3), dtype=np.uint8)
        Image.fromarray(pattern).resize((16, 16), Image.Resampling.NEAREST).save(path)
        masked = tmp_path / f"masked{i}.png"
        Image.fromarray(pattern).resize((16, 16), Image.Resampling.NEAREST).save(masked)
        item = {
            "path": str(path),
            "id": f"img{i}.png",
            "caption": f"plain caption {i}",
            "dataset_name": "ds",
            "dataset_path": str(tmp_path),
            "prefix": "",
            "dropout_rate": 0.9,
            "use_captions": True,
            "use_model_aware_captions": False,
            "target_w": 16,
            "target_h": 16,
            "orig_w": 16,
            "orig_h": 16,
            "cache_dir": str(tmp_path / "cache"),
            "variant": "original",
            "is_video": False,
            "has_masked": True,
            "masked_path": str(masked),
            "masked_caption": f"masked caption {i}",
            "has_masked_caption": True,
            "masked_cache_dir": str(tmp_path / "cache_masked"),
            "original_weight": 0.5,
        }
        items.extend([item] * (1 if i else repeats))
    return items


@functools.lru_cache(maxsize=1)
def _definition():
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    return ModelRegistry._definitions["flux3-image-klein-base"]


def _pipeline(tmp_path, items=None, **config):
    t = object.__new__(_Pipeline)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        t._model = _Model()
    t._model.train()
    t.forwards = []
    t.captions = []
    t.device = torch.device("cpu")
    t.autocast_dtype = torch.float32
    t.use_amp = False
    t.logger = MagicMock()
    t._log_writer = MagicMock()
    t.definition = _definition()
    t.driver = SimpleNamespace()
    t.latent_manager = _Latents()
    t.config = {
        "cache_latents": True,
        "validation_enabled": True,
        "validation_split_seed": 3,
        "validation_timestep_bands": BANDS,
        **config,
    }
    t.checkpoint_manager = SimpleNamespace(output_dir=str(tmp_path))
    t._sigma_tracker = MagicMock()
    inventory = _items(tmp_path) if items is None else items
    t._validation_split = split_holdout(inventory, 0.34, 3, image_hash=lambda path: _spread_hash(path))
    return t


def _spread_hash(path):
    number = int("".join(c for c in path.rsplit("img", 1)[-1] if c.isdigit()) or 0)
    return (1 << 63) >> (number * 10) | (0xFF << (number * 3))


def _numbers(metrics):
    return metrics["band_loss"], metrics["mean"], metrics["worst_band"]


def _run(t, step=0):
    return validation_pass.run_validation_pass(t, t._validation_split, step)


def test_two_passes_with_unchanged_weights_give_identical_numbers(tmp_path):
    t = _pipeline(tmp_path)
    first = _run(t)
    # Whatever training did to the random streams in between.
    torch.rand(5)
    random.random()
    np.random.rand()
    second = _run(t, step=100)
    assert _numbers(first) == _numbers(second)
    assert len(first["band_loss"]) == BANDS
    assert all(np.isfinite(first["band_loss"]))


def test_numbers_hold_across_entry_states_repeats_and_inventory_order(tmp_path):
    base = _numbers(_run(_pipeline(tmp_path)))

    torch.manual_seed(999)
    random.seed(999)
    np.random.seed(999)
    repeated = _items(tmp_path, repeats=5)
    random.shuffle(repeated)

    assert _numbers(_run(_pipeline(tmp_path, items=repeated))) == base


def test_the_pass_is_sources_times_bands_forward_passes(tmp_path):
    t = _pipeline(tmp_path)
    _run(t)
    assert len(t.forwards) == len(t._validation_split.representatives) * BANDS
    assert len(t._validation_split.representatives) >= 2


def test_the_pass_uses_the_plain_caption_with_dropout_off(tmp_path):
    t = _pipeline(tmp_path)
    # The pass restores the streams, so each pass needs its own state to differ.
    for seed in range(12):
        random.seed(seed)
        _run(t)
    # The stub dropout rate is 0.9 and every image has a masked variant.
    expected = {item["caption"] for item in t._validation_split.representatives}
    assert set(t.captions) == expected


def test_the_noise_differs_per_image_and_not_per_pass(tmp_path):
    t = _pipeline(tmp_path)
    keys = [r["key"] for r in t._validation_split.representatives]
    like = torch.zeros(1, LATENT)
    a = validation_pass.fixed_noise(3, keys[0], like)
    assert torch.equal(a, validation_pass.fixed_noise(3, keys[0], like))
    assert not torch.equal(a, validation_pass.fixed_noise(3, keys[1], like))
    assert not torch.equal(a, validation_pass.fixed_noise(4, keys[0], like))


def test_the_pass_leaves_no_trace(tmp_path):
    t = _pipeline(tmp_path)
    t._model.linear.weight.grad = torch.ones_like(t._model.linear.weight)
    t._model.dropout.eval()  # a module the run keeps in its own mode
    weights = {k: v.clone() for k, v in t._model.state_dict().items()}
    gradient = t._model.linear.weight.grad.clone()
    torch.manual_seed(5)
    random.seed(5)
    np.random.seed(5)
    expected = (torch.rand(1), random.random(), np.random.rand())
    torch.manual_seed(5)
    random.seed(5)
    np.random.seed(5)

    _run(t)

    assert (torch.rand(1), random.random(), np.random.rand()) == expected
    assert all(torch.equal(v, weights[k]) for k, v in t._model.state_dict().items())
    assert torch.equal(t._model.linear.weight.grad, gradient)
    assert t._model.training is True
    assert t._model.dropout.training is False
    t._sigma_tracker.update.assert_not_called()


def test_the_model_is_in_eval_mode_and_builds_no_graph_during_the_pass(tmp_path):
    t = _pipeline(tmp_path)
    seen = []
    forward = t.forward_pass

    def spy(*args):
        seen.append((t._model.training, torch.is_grad_enabled()))
        return forward(*args)

    t.forward_pass = spy
    _run(t)
    assert set(seen) == {(False, False)}


def test_the_model_mode_comes_back_when_a_forward_fails(tmp_path):
    t = _pipeline(tmp_path)

    def boom(*_args):
        raise RuntimeError("forward failed")

    t.forward_pass = boom
    state = torch.get_rng_state()
    with pytest.raises(RuntimeError):
        _run(t)
    assert t._model.training is True
    assert torch.equal(torch.get_rng_state(), state)


def test_the_loss_is_plain_mse_over_the_band(tmp_path):
    t = _pipeline(tmp_path)
    t.compute_loss_weight = MagicMock(side_effect=AssertionError("no weighting"))
    metrics = _run(t)
    assert metrics["mean"] == pytest.approx(sum(metrics["band_loss"]) / BANDS)
    assert metrics["band_loss"][metrics["worst_band"]] == max(metrics["band_loss"])


def test_a_pass_appends_a_line_and_emits_an_event(tmp_path):
    t = _pipeline(tmp_path)
    _run(t, step=0)
    _run(t, step=100)
    lines = (tmp_path / "validation_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in lines] == [0, 100]
    record = json.loads(lines[0])
    assert set(record) >= {"step", "band_loss", "band_centres", "mean", "worst_band", "seconds", "images"}
    assert record["images"] == len(t._validation_split.representatives)
    assert [c.args[0] for c in t._log_writer.emit.call_args_list] == ["validation", "validation"]


def test_a_non_finite_loss_is_written_as_null(tmp_path):
    t = _pipeline(tmp_path)
    forward = t.forward_pass
    t.forward_pass = lambda *a: forward(*a) * float("nan")
    _run(t)
    record = json.loads((tmp_path / "validation_metrics.jsonl").read_text(encoding="utf-8"))
    assert record["band_loss"] == [None] * BANDS
    assert record["mean"] is None


# ── Through the real loop ────────────────────────────────────────────────────


def test_passes_are_due_at_zero_each_interval_and_the_last_step():
    due = [s for s in range(500) if validation_pass.is_due(s, 100, 500)]
    assert due == [0, 100, 200, 300, 400, 499]


class _LoopTrainer(PipelineTrainMixin):
    """The real train() over stub hooks; records the random stream at every step."""

    def __init__(self, tmp_path, steps, config=None):
        self.logger = MagicMock()
        self.config = {
            "max_train_steps": steps,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "cache_latents": False,
            "save_every_n_steps": 0,
            "sample_before_training": False,
            "noise_offset": 0.0,
            **(config or {}),
        }
        self.inventory = [{"id": 0}]
        self.device = torch.device("cpu")
        self.autocast_dtype = torch.float32
        self.use_amp = False
        self.scaler = MagicMock()
        self.scaler.is_enabled.return_value = False
        self._model = torch.nn.Linear(4, 4)
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr=0.01)
        self.lr_scheduler = MagicMock()
        self.ema_handler = None
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
        self.latent_manager.encode_and_cache_batch.side_effect = lambda *a, **k: torch.randn(1, 4)
        self.stream = []

    def _get_primary_model(self):
        return self._model

    def _iter_training_batches(self, batch_size):
        while True:
            yield [{"id": 0}]

    def _get_batch(self, batch_items, decode_pixels=True):
        return {"images": torch.randn(1, 3, 8, 8), "ids": [0], "paths": ["x.png"], "captions": ["c"]}

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
        self.stream.append(torch.rand(1).item())
        return torch.zeros(batch_size)

    def add_noise(self, latents, noise, timesteps):
        return latents

    def forward_pass(self, noisy_input, timesteps, text_emb, batch):
        return self._model(noisy_input)

    def compute_target(self, latents, noise, timesteps):
        return torch.zeros_like(latents)

    def _compute_step_loss(self, pred, target, timesteps, batch, grad_accum):
        return (pred - target).pow(2).mean()

    def _build_trainable_components(self):
        return {}

    def get_te_cache(self):
        return None

    def _build_cache_manifest(self):
        return None

    def on_epoch_end(self, epoch):
        pass


def _train(trainer):
    with patch("app.core.db.DatabaseEngine.get_instance", side_effect=RuntimeError("no db in test")):
        asyncio.run(trainer.train())


def _fake_split(items=()):
    return SimpleNamespace(
        training_items=[{"id": 0}], holdout_keys=[], representatives=list(items), groups=[]
    )


def test_validation_off_runs_no_pass_and_writes_nothing(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("no pass may run with validation off")

    monkeypatch.setattr(validation_pass, "run_validation_pass", forbidden)
    t = _LoopTrainer(tmp_path, steps=3)
    _train(t)
    assert t.optimizer is not None
    assert not (tmp_path / "validation_metrics.jsonl").exists()
    assert not (tmp_path / "validation_split.json").exists()


def test_validation_off_leaves_the_random_stream_as_with_the_feature_unreached(tmp_path):
    torch.manual_seed(11)
    reached = _LoopTrainer(tmp_path, steps=5)
    _train(reached)

    torch.manual_seed(11)
    unreached = _LoopTrainer(tmp_path, steps=5)
    unreached._enter_validation_split = lambda: None
    unreached._maybe_validate = lambda step, max_steps: None
    _train(unreached)

    assert reached.stream == unreached.stream


class _MeasuredLoopTrainer(_LoopTrainer):
    """Adds what the real pass needs; its batch and its forward both draw randomness."""

    validation_timestep_domain = (0.0, 1000.0)

    def add_noise(self, latents, noise, timesteps):
        sigma = (timesteps / 1000.0).reshape(-1, *[1] * (latents.dim() - 1)).to(latents)
        return (1 - sigma) * latents + sigma * noise

    def forward_pass(self, noisy_input, timesteps, text_emb, batch):
        torch.rand(1)
        return self._model(noisy_input)


def test_validation_on_leaves_the_random_stream_as_with_it_off(tmp_path):
    torch.manual_seed(11)
    off = _MeasuredLoopTrainer(tmp_path, steps=5)
    _train(off)

    torch.manual_seed(11)
    on = _MeasuredLoopTrainer(
        tmp_path, steps=5, config={"validation_every_n_steps": 1, "validation_timestep_bands": 2}
    )
    on._validation_split = _fake_split([{"key": "ds/a.png", "id": "a.png"}])
    _train(on)

    assert on.stream == off.stream
    assert len((tmp_path / "validation_metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 5


def test_passes_run_at_the_cadence_through_the_real_loop(tmp_path):
    steps = []
    t = _LoopTrainer(tmp_path, steps=7, config={"validation_every_n_steps": 3})
    t._validation_split = _fake_split()
    with patch.object(validation_pass, "run_validation_pass", lambda _t, _s, step: steps.append(step)):
        _train(t)
    assert steps == [0, 3, 6]


def test_a_pass_stops_the_step_timer_while_it_runs(tmp_path):
    order = []
    t = _LoopTrainer(tmp_path, steps=1, config={"validation_every_n_steps": 1})
    t._validation_split = _fake_split()
    t.logger_component.pause_step_timer.side_effect = lambda: order.append("pause")
    t.logger_component.resume_step_timer.side_effect = lambda: order.append("resume")
    with patch.object(validation_pass, "run_validation_pass", lambda *_a: order.append("pass")):
        _train(t)
    assert order[:3] == ["pause", "pass", "resume"]


def test_a_resumed_run_takes_the_next_due_pass(tmp_path):
    steps = []
    t = _LoopTrainer(tmp_path, steps=9, config={"validation_every_n_steps": 3})
    t._validation_split = _fake_split()
    original = t._maybe_validate

    def from_step_four(step, max_steps):
        if step >= 4:
            original(step, max_steps)

    t._maybe_validate = from_step_four
    with patch.object(validation_pass, "run_validation_pass", lambda _t, _s, step: steps.append(step)):
        _train(t)
    assert steps == [6, 8]


# ── Round 1 of the code review ───────────────────────────────────────────────


class _StochasticEncoder:
    """Encodes as a VAE does, sampling its posterior from the global stream."""

    def encode_and_cache_batch(self, images, ids=None, cache_dirs=None, source_paths=None, **_):
        return torch.randn(1, LATENT)


def test_uncached_latents_are_the_same_in_every_pass(tmp_path):
    t = _pipeline(tmp_path, cache_latents=False)
    t.latent_manager = _StochasticEncoder()
    first = _numbers(_run(t))
    torch.rand(9)
    random.random()
    assert _numbers(_run(t, step=100)) == first


def test_two_images_get_different_uncached_latents(tmp_path):
    t = _pipeline(tmp_path, cache_latents=False)
    seen = []
    encoder = _StochasticEncoder()

    def recording(*args, **kwargs):
        seen.append(encoder.encode_and_cache_batch(*args, **kwargs))
        return seen[-1]

    t.latent_manager = SimpleNamespace(encode_and_cache_batch=recording)
    _run(t)
    assert len(seen) >= 2
    assert not torch.equal(seen[0], seen[1])


class _ScheduleFree:
    """Changes the model's weights between its modes, as a schedule-free optimizer does."""

    def __init__(self, model):
        self.model = model
        self.calls = []

    def eval(self):
        self.calls.append("eval")
        self.saved = self.model.linear.weight.detach().clone()
        with torch.no_grad():
            self.model.linear.weight.add_(1.0)

    def train(self):
        self.calls.append("train")
        with torch.no_grad():
            self.model.linear.weight.copy_(self.saved)


class _Ema:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def store_and_swap(self):
        self.calls.append("swap")
        self.saved = self.model.linear.weight.detach().clone()
        with torch.no_grad():
            self.model.linear.weight.add_(10.0)

    def restore(self):
        self.calls.append("restore")
        with torch.no_grad():
            self.model.linear.weight.copy_(self.saved)


def _weights_at_forward(t):
    seen = []
    forward = t.forward_pass

    def spy(*args):
        seen.append(t._model.linear.weight.detach().clone())
        return forward(*args)

    t.forward_pass = spy
    return seen


def test_the_pass_measures_a_schedule_free_optimizers_evaluation_weights(tmp_path):
    t = _pipeline(tmp_path)
    before = t._model.linear.weight.detach().clone()
    t.optimizer = _ScheduleFree(t._model)
    seen = _weights_at_forward(t)

    _run(t)

    assert all(torch.equal(w, before + 1.0) for w in seen)
    assert t.optimizer.calls == ["eval", "train"]
    assert torch.equal(t._model.linear.weight, before)


def test_the_pass_measures_the_ema_weights_when_ema_is_on(tmp_path):
    t = _pipeline(tmp_path)
    before = t._model.linear.weight.detach().clone()
    t.ema_handler = _Ema(t._model)
    seen = _weights_at_forward(t)

    _run(t)

    assert all(torch.equal(w, before + 10.0) for w in seen)
    assert t.ema_handler.calls == ["swap", "restore"]
    assert torch.equal(t._model.linear.weight, before)


def test_the_optimizer_and_ema_come_back_when_a_forward_fails(tmp_path):
    t = _pipeline(tmp_path)
    before = t._model.linear.weight.detach().clone()
    t.optimizer = _ScheduleFree(t._model)
    t.ema_handler = _Ema(t._model)

    def boom(*_args):
        raise RuntimeError("forward failed")

    t.forward_pass = boom
    with pytest.raises(RuntimeError):
        _run(t)

    assert t.optimizer.calls == ["eval", "train"]
    assert t.ema_handler.calls == ["swap", "restore"]
    assert torch.equal(t._model.linear.weight, before)


def test_an_optimizer_without_modes_is_left_alone(tmp_path):
    t = _pipeline(tmp_path)
    t.optimizer = torch.optim.SGD(t._model.parameters(), lr=0.01)
    assert _run(t)["images"] > 0


def test_the_optimizer_stays_out_of_eval_when_the_ema_swap_fails(tmp_path):
    t = _pipeline(tmp_path)
    before = t._model.linear.weight.detach().clone()
    t.optimizer = _ScheduleFree(t._model)
    t.ema_handler = _Ema(t._model)

    def boom():
        raise RuntimeError("swap failed")

    t.ema_handler.store_and_swap = boom
    with pytest.raises(RuntimeError):
        _run(t)

    assert t.optimizer.calls == ["eval", "train"]
    assert t.ema_handler.calls == []
    assert torch.equal(t._model.linear.weight, before)
