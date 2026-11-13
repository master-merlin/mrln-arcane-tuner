"""The MiniMax-H3-Image family's loader, driver and trainer (``minimax_h3_image``).

An image family of its own, never the ``minimax_h3`` video family: no audio, no frames, no
dual shift. What this file holds it to, on CPU, against tiny models built from the
checkpoint's own config fixture (``test_minimax_h3_image_vendor``'s overrides), never the
real weights:

* the head for a sigma: each of the 6 grid sigmas maps to its own head 0..5, any other sigma
  to the nearest one (DiffSynth-Studio ``minimax_h3_image.py:537-542``);
* one sigma per batch: the transformer takes one ``pdd_step_index`` per forward, so a batch
  of two items gets one sigma and one head, and over 600 seeded steps every head is used;
* the training step on a tiny random DiT: ``x_t = t*x0 + (1-t)*noise``, the target
  ``x0 - noise``, the head reaches the forward pass, the loss is the shared MSE over the
  target rows only, and a step changes only the LoRA weights;
* the packed ``[text | target]`` layout: the 3-axis rotary grid of the reference pipeline
  (diffusers PR #15230 ``before_denoise.py``), text tagged 1, image rows tagged 0;
* the refusals, each raised by ``_setup_family`` before any loader exists;
* video and audio files skipped loudly, with a count;
* the text path: the shared layer tap, the cache key under the family's own prefix, the
  cache-served ``encode_text`` and the release before the DiT loads;
* the loader: the deferred vendored DiT from ``transformer_diffusers``, the vendored VAE
  from ``vae_diffusers`` wrapped once in the shared pixel adapter, and the 6 PDD heads.

``stand_in_definition()`` carries the keys these modules read with the shipped YAML's values
(``test_the_stand_in_agrees_with_the_shipped_definition``), and a target list the tiny models
have; the API -> job -> trainer tests run the shipped definition itself.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.tests.h3_text_stubs import StubProcessor, StubQwen3VL
from tests.engine.families.test_minimax_h3_image_vendor import (
    TRANSFORMER_CONFIG,
    TRANSFORMER_TINY,
    VAE_CONFIG,
    VAE_TINY,
    _tiny_config,
    _tiny_transformer,
    _vendor,
)

FAMILY_PKG = "app.engine.models.families.minimax_h3_image"
DEFINITION_ID = "minimax-h3-image-t2i"

# modular_model_index.json `sample_sigmas` of MiniMaxAI/MiniMax-H3-Image at 00dd081e,
# without the terminal 0.0 (not a training level).
H3_IMAGE_SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132]
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


def stand_in_definition(**overrides: Any) -> ModelDefinition:
    """The keys the family's modules read, with the model's values (the YAML is a later task)."""
    data: dict[str, Any] = {
        "id": DEFINITION_ID,
        "family": "minimax_h3_image",
        "name": "MiniMax-H3-Image",
        "architecture_params": {
            "te.hidden_state_tap_index": 50,
            "te.max_length": 4096,
            "timestep.default_mode": "fixed_sigmas",
            "timestep.fixed_sigmas": list(H3_IMAGE_SIGMAS),
        },
        "lora_targetable_modules": list(LORA_TARGETS),
        "block_topology": [
            {"name": "transformer_blocks", "attr_path": "transformer_blocks", "count": 50},
            {"name": "refiner_blocks", "attr_path": "token_refiner.refiner_blocks", "count": 2},
        ],
    }
    data.update(overrides)
    return ModelDefinition(**data)


class _Log:
    """Records every log call (order-independent of structlog's global configuration)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def _record(self, event, **kw):
        self.events.append({"event": event, **kw})

    info = warning = debug = error = _record

    def named(self, event: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["event"] == event]


BASE_CONFIG = {
    "mixed_precision": "bf16",
    "te_quantization": "none",
    "cache_text_embeddings": True,
    "cache_latents": True,
    "sample_prompts": [],
}


def bare_trainer(definition: ModelDefinition | None = None, **config: Any):
    """The real trainer class, built without weights; ``_setup_family`` is NOT run."""
    from app.engine.models.families.minimax_h3_image.trainer import MiniMaxH3ImageTrainer

    trainer = object.__new__(MiniMaxH3ImageTrainer)
    trainer.definition = definition or stand_in_definition()
    trainer.config = {**BASE_CONFIG, **config}
    trainer.device = torch.device("cpu")
    trainer.logger = _Log()
    trainer.components = {}
    trainer.text_cache = {}
    trainer._te_unloaded = False
    trainer.epoch = 0
    trainer.global_step = 0
    trainer.max_train_steps = 100
    trainer.optimizer = None
    trainer.autocast_dtype = torch.float32
    return trainer


def set_up(definition: ModelDefinition | None = None, **config: Any):
    trainer = bare_trainer(definition, **config)
    trainer._setup_family()
    return trainer


def _driver(definition: ModelDefinition | None = None):
    from app.engine.models.families.minimax_h3_image.driver import MiniMaxH3ImageDriver

    return MiniMaxH3ImageDriver(definition or stand_in_definition(), torch.device("cpu"))


def _with_model(trainer, model: nn.Module) -> None:
    components = {"transformer": model, "unet": model}
    trainer.components = components
    trainer.driver.assign_components(components)
    trainer.transformer = model


class _HeadSpy:
    """Wraps the model's forward and records each call's head index and timestep."""

    def __init__(self, model: nn.Module) -> None:
        self.calls: list[dict[str, Any]] = []
        original = model.forward

        def forward(*args, **kwargs):
            self.calls.append(
                {"head": kwargs["pdd_step_index"], "timestep": kwargs["timestep"].detach().clone(),
                 "rows": kwargs["hidden_states"].shape[1]}
            )
            return original(*args, **kwargs)

        model.forward = forward


def _text(batch: int = 2, lengths: tuple[int, ...] = (3, 5), dim: int | None = None) -> TextEncoderOutput:
    dim = dim or TRANSFORMER_TINY["text_dim"]
    emb = torch.zeros(batch, max(lengths), dim)
    mask = torch.zeros(batch, max(lengths), dtype=torch.long)
    gen = torch.Generator().manual_seed(7)
    for i, n in enumerate(lengths):
        emb[i, :n] = torch.randn(n, dim, generator=gen)
        mask[i, :n] = 1
    return TextEncoderOutput(embeddings=emb, attention_mask=mask)


def _latents(batch: int = 2, hw: tuple[int, int] = (4, 6), seed: int = 3) -> torch.Tensor:
    return torch.randn(batch, 24, *hw, generator=torch.Generator().manual_seed(seed))


# ── The head for a sigma ──────────────────────────────────────────────────


def test_each_grid_sigma_maps_to_its_own_head():
    driver = _driver()
    assert driver.head_sigmas() == H3_IMAGE_SIGMAS
    for i, sigma in enumerate(H3_IMAGE_SIGMAS):
        assert driver.head_for_sigma(sigma) == i, f"sigma {sigma} must use head {i}"
        assert driver.head_for_sigma(torch.tensor([sigma])) == i


@pytest.mark.parametrize(
    ("sigma", "head"),
    [
        (0.99, 0),  # nearer 1.0 than 0.9675
        (0.98, 1),
        (0.93, 1),  # 0.0375 from 0.9675, 0.0425 from 0.8875
        (0.92, 2),
        (0.80, 3),
        (0.64, 4),
        (0.40, 4),
        (0.38, 5),
        (0.0, 5),  # the terminal sigma is not a head; the nearest one is the last
    ],
)
def test_a_sigma_between_grid_values_uses_the_nearest_head(sigma, head):
    assert _driver().head_for_sigma(sigma) == head


def test_the_head_grid_comes_from_the_definition_and_is_required():
    arch = dict(stand_in_definition().architecture_params)
    arch.pop("timestep.fixed_sigmas")
    with pytest.raises(ValueError, match="timestep.fixed_sigmas"):
        _driver(stand_in_definition(architecture_params=arch)).head_sigmas()


# ── One sigma per batch ───────────────────────────────────────────────────


def test_a_batch_of_two_draws_one_sigma():
    driver = _driver()
    config = {"timestep_sampling": "fixed_sigmas", "fixed_sigmas": list(H3_IMAGE_SIGMAS)}
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        t = driver.sample_timesteps(2, torch.device("cpu"), config, generator=gen)
        assert t.shape == (2,)
        assert float(t[0]) == float(t[1]), f"the two items got different timesteps {t.tolist()}"
        sigma = 1.0 - float(t[0])
        assert min(abs(sigma - s) for s in H3_IMAGE_SIGMAS) < 1e-6, f"sigma {sigma} is not a grid value"


def test_over_600_seeded_steps_every_head_is_used():
    driver = _driver()
    config = {"timestep_sampling": "fixed_sigmas", "fixed_sigmas": list(H3_IMAGE_SIGMAS)}
    gen = torch.Generator().manual_seed(0)
    state = torch.random.get_rng_state()
    heads = [
        driver.head_for_sigma(1.0 - driver.sample_timesteps(2, torch.device("cpu"), config, generator=gen))
        for _ in range(600)
    ]
    assert torch.equal(state, torch.random.get_rng_state()), "a seeded draw touched the global RNG"
    assert set(heads) == set(range(6)), f"heads used over 600 steps: {sorted(set(heads))}"


def test_another_mode_still_draws_one_sigma_per_batch():
    driver = _driver()
    gen = torch.Generator().manual_seed(1)
    t = driver.sample_timesteps(3, torch.device("cpu"), {"timestep_sampling": "uniform"}, generator=gen)
    assert t.shape == (3,) and len(set(t.tolist())) == 1
    assert 0.0 <= float(t[0]) <= 1.0


# ── The packed [text | target] layout ─────────────────────────────────────


def test_the_layout_is_text_then_target_with_the_reference_grid():
    driver = _driver()
    # latent 4 x 8 -> 2 x 4 target rows behind 3 text rows
    layout = driver.packed_layout(num_text=3, latent_height=4, latent_width=8, patch_size=(1, 2, 2))
    assert layout.text_indices.tolist() == [0, 1, 2]
    assert layout.image_indices.tolist() == list(range(3, 11))
    assert layout.token_tags.tolist() == [1, 1, 1] + [0] * 8
    assert layout.timestep_indices.tolist() == [0] * 11
    pos = layout.position_ids
    assert pos.dtype == torch.float64 and pos.shape == (11, 3)
    # text rows: t = 0, 1, 2; h = w = 0
    assert pos[:3].tolist() == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    # target rows: one frame at t = num_text; h/w aspect-normalised by sqrt(4*8) and scaled by 32
    assert pos[3:, 0].tolist() == [3.0] * 8
    sqrt_area = math.sqrt(32)
    h_left = (1 - 4 / sqrt_area) / 2
    w_left = (1 - 8 / sqrt_area) / 2
    h_grid = [(h_left + k * (4 / sqrt_area) / 2) * 32 for k in range(2)]
    w_grid = [(w_left + k * (8 / sqrt_area) / 4) * 32 for k in range(4)]
    expected = [[h, w] for h in h_grid for w in w_grid]
    assert torch.allclose(pos[3:, 1:], torch.tensor(expected, dtype=torch.float64), atol=1e-12)
    # a square canvas spans [0, 32)
    square = driver.packed_layout(num_text=1, latent_height=4, latent_width=4, patch_size=(1, 2, 2))
    assert square.position_ids[1:, 1].tolist() == [0.0, 0.0, 16.0, 16.0]


def test_patchify_round_trips():
    driver = _driver()
    lat = _latents(1, (4, 6))
    rows = driver.patchify(lat, (1, 2, 2))
    assert rows.shape == (1, 6, 96)
    assert torch.equal(driver.unpatchify(rows, 4, 6, (1, 2, 2)), lat)


# ── The training step on a tiny random DiT ────────────────────────────────


def test_noise_and_target_follow_the_inverted_convention():
    trainer = set_up()
    x0, noise = _latents(2, seed=1), _latents(2, seed=2)
    t = torch.tensor([0.25, 0.25])
    assert torch.allclose(trainer.add_noise(x0, noise, t), 0.25 * x0 + 0.75 * noise)
    assert torch.equal(trainer.compute_target(x0, noise, t), x0 - noise)


def test_the_head_of_the_drawn_sigma_reaches_the_forward_pass():
    trainer = set_up(timestep_sampling="fixed_sigmas")
    model = _tiny_transformer()
    spy = _HeadSpy(model)
    _with_model(trainer, model)
    for i, sigma in enumerate(H3_IMAGE_SIGMAS):
        t = torch.full((2,), 1.0 - sigma)
        with torch.no_grad():
            pred = trainer.forward_pass(_latents(2), t, _text(), {})
        assert pred.shape == (2, 24, 4, 6), "the prediction must be the target rows in latent shape"
        assert [c["head"] for c in spy.calls] == [i, i], f"sigma {sigma}: heads {[c['head'] for c in spy.calls]}"
        assert all(torch.allclose(c["timestep"], torch.tensor([1.0 - sigma])) for c in spy.calls)
        spy.calls.clear()


def test_a_batch_of_two_gets_one_sigma_and_one_head_and_a_target_only_loss():
    trainer = set_up(timestep_sampling="fixed_sigmas")
    model = _tiny_transformer()
    spy = _HeadSpy(model)
    _with_model(trainer, model)
    latents = _latents(2)
    torch.manual_seed(0)
    with torch.no_grad():
        step = trainer.forward_loss_step(latents, _text(lengths=(3, 5)), {"captions": ["a", "b"]}, 1)
    t = step.timesteps
    assert t.shape == (2,) and float(t[0]) == float(t[1]), f"one sigma per batch, got {t.tolist()}"
    assert len(spy.calls) == 2 and spy.calls[0]["head"] == spy.calls[1]["head"]
    assert spy.calls[0]["head"] == trainer.driver.head_for_sigma(1.0 - float(t[0]))
    # each item's forward carried exactly its own target rows (2 x 3 patches), no text rows
    assert [c["rows"] for c in spy.calls] == [6, 6]
    assert step.pred.shape == latents.shape
    assert torch.equal(step.target, step.prepared_latents - step.prepared_noise)
    expected = torch.nn.functional.mse_loss(step.pred.float(), step.target.float())
    assert torch.allclose(step.loss, expected), "the loss is the plain MSE over the target rows"


def test_items_at_two_different_sigmas_are_refused():
    trainer = set_up()
    _with_model(trainer, _tiny_transformer())
    with pytest.raises(ValueError, match="one sigma per batch"):
        trainer.forward_pass(_latents(2), torch.tensor([0.25, 0.5]), _text(), {})


def test_a_text_row_change_reaches_the_prediction_but_padding_does_not():
    """The item's TRUE text rows enter the sequence; the batch's padding rows never do."""
    trainer = set_up()
    _with_model(trainer, _tiny_transformer())
    lat = _latents(2)
    t = torch.full((2,), 0.25)
    text = _text(lengths=(3, 5))
    with torch.no_grad():
        base = trainer.forward_pass(lat, t, text, {})
        padded = TextEncoderOutput(embeddings=text.embeddings.clone(), attention_mask=text.attention_mask)
        padded.embeddings[0, 3:] = 99.0  # item 0's padding rows
        assert torch.equal(trainer.forward_pass(lat, t, padded, {}), base)
        changed = TextEncoderOutput(embeddings=text.embeddings.clone(), attention_mask=text.attention_mask)
        changed.embeddings[0, 0] += 1.0
        moved = trainer.forward_pass(lat, t, changed, {})
    assert not torch.equal(moved[0], base[0]) and torch.equal(moved[1], base[1])


def test_a_step_changes_only_the_lora_weights():
    from peft import LoraConfig, get_peft_model

    trainer = set_up(timestep_sampling="fixed_sigmas")
    model = get_peft_model(
        _tiny_transformer(), LoraConfig(r=4, lora_alpha=4, target_modules=list(LORA_TARGETS))
    )
    trainer._update_primary_model(model)
    trainer.components = {"transformer": model, "unet": model}
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all(
        "lora_" in n for n, p in model.named_parameters() if p.requires_grad
    ), "only LoRA parameters may be trainable"
    opt = torch.optim.SGD(trainable, lr=1.0)
    torch.manual_seed(0)
    step = trainer.forward_loss_step(_latents(2), _text(), {"captions": ["a", "b"]}, 1)
    step.loss.backward()
    opt.step()
    changed = {n for n, p in model.named_parameters() if not torch.equal(p.detach(), before[n])}
    assert changed, "the step changed nothing"
    assert all("lora_" in n for n in changed), f"non-LoRA weights changed: {sorted(n for n in changed if 'lora_' not in n)}"


# ── CLOBBER hooks: delegated explicitly ───────────────────────────────────


def test_the_trainer_delegates_every_clobber_hook_the_driver_overrides():
    from app.engine.core.hook_dispatch import driver_meaningfully_overrides
    from app.engine.models.families.minimax_h3_image.driver import MiniMaxH3ImageDriver
    from app.engine.models.families.minimax_h3_image.trainer import MiniMaxH3ImageTrainer
    from tests.engine.test_hook_wiring_meta import CLOBBER_HOOKS

    for hook in CLOBBER_HOOKS:
        if driver_meaningfully_overrides(MiniMaxH3ImageDriver, hook):
            assert hook in MiniMaxH3ImageTrainer.__dict__, f"{hook} is not delegated by the trainer"
    assert _driver().init_scheduler() is None


# ── The refusals, before any loader exists ────────────────────────────────


@pytest.mark.parametrize(
    ("config", "needle"),
    [
        ({"cache_text_embeddings": False}, "cache_text_embeddings"),
        ({"cache_latents": False}, "cache_latents"),
        ({"mixed_precision": "fp16"}, "mixed_precision"),
        ({"mixed_precision": "fp32"}, "mixed_precision"),
        ({"train_text_encoder": True}, "train_text_encoder"),
        ({"te_quantization": "int8"}, "te_quantization"),
        ({"quantization": "nf4"}, "quantization"),
        ({"quantization": "fp8"}, "quantization"),
        ({"control_inputs": 1}, "control_inputs"),
        ({"datasets": [{"dataset_name": "faces", "masking_enabled": True}]}, "masking_enabled"),
    ],
)
def test_each_refusal_raises_before_any_loader(config, needle):
    trainer = bare_trainer(**config)
    with pytest.raises(ValueError, match=needle):
        trainer._setup_family()
    assert getattr(trainer, "loader", None) is None and getattr(trainer, "driver", None) is None


def test_an_edit_definition_is_refused():
    trainer = bare_trainer(stand_in_definition(control_inputs=1))
    with pytest.raises(ValueError, match="control_inputs"):
        trainer._setup_family()


@pytest.mark.parametrize("quantization", ["none", "int8", None])
def test_the_supported_settings_pass(quantization):
    trainer = set_up(quantization=quantization, mixed_precision=None)
    assert trainer.config["mixed_precision"] == "bf16"
    assert trainer.loader.defer_transformer is True


def test_unset_precision_resolves_to_bf16():
    trainer = bare_trainer()
    trainer.config.pop("mixed_precision")
    trainer._setup_family()
    assert trainer.config["mixed_precision"] == "bf16"


# ── Video and audio files are skipped loudly ──────────────────────────────


def test_video_and_audio_items_are_skipped_with_a_count():
    trainer = set_up()
    trainer.inventory = [
        {"path": "d/a.png", "id": 1},
        {"path": "d/b.mp4", "id": 2, "is_video": True},
        {"path": "d/b.mp4", "id": 3, "is_video": True},
        {"path": "d/c.wav", "id": 4, "is_audio": True},
        {"path": "d/e.jpg", "id": 5},
    ]
    trainer._skip_non_image_items()
    assert [i["id"] for i in trainer.inventory] == [1, 5]
    summary = trainer.logger.named("minimax_h3_image_media_skipped")
    assert summary == [
        {"event": "minimax_h3_image_media_skipped", "skipped_video_files": 1, "skipped_audio_files": 1,
         "total_items": 2}
    ]
    assert len(trainer.logger.named("media_file_skipped")) == 2


def test_a_dataset_of_only_video_fails():
    trainer = set_up()
    trainer.inventory = [{"path": "d/b.mp4", "id": 2, "is_video": True}]
    with pytest.raises(ValueError, match="1 video file"):
        trainer._skip_non_image_items()


# ── The text path ─────────────────────────────────────────────────────────


def _text_components(driver):
    encoder, processor = StubQwen3VL(num_hidden_layers=64, hidden_size=8), StubProcessor()
    components = {"text_encoder": encoder, "tokenizer": processor}
    driver.assign_components(components)
    return components, encoder


def test_the_cache_key_carries_the_family_prefix_and_never_the_video_one():
    from app.engine.models.families.minimax_h3_shared import text as h3_text

    driver = _driver()
    _text_components(driver)
    key = driver.te_cache_key("a red fox")
    fp = h3_text.tokenizer_fingerprint(driver.tokenizer)
    assert key == f"minimax_h3_image|{DEFINITION_ID}|tap=50|tok={fp[:16]}|a red fox"
    assert not key.startswith("minimax_h3|")
    assert driver.te_cache_scope() == f"tap50-{fp[:16]}"


def test_the_text_tap_is_layer_50():
    driver = _driver()
    _text_components(driver)
    out = driver.encode_text(["a red fox", "b"], torch.float32)
    assert out.embeddings.shape == (2, 3, 8)
    assert out.attention_mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert torch.all(out.embeddings[0] == 50.0), "the embedding must be hidden_states[50]"


def test_encode_text_serves_the_cache_and_a_miss_after_release_raises(tmp_path):
    trainer = set_up()
    components, encoder = _text_components(trainer.driver)
    trainer.components = components
    trainer.inventory = []
    trainer._resolve_te_cache_dirs = lambda: [str(tmp_path)]
    trainer._pre_cache_text_embeddings()
    assert "" in trainer.text_cache, "the dropout prompt must be pre-cached"
    calls = encoder.calls
    trainer._offload_text_encoders()
    assert trainer.driver.text_encoder is None and "text_encoder" not in trainer.components
    out = trainer.encode_text(["", ""], torch.float32)
    assert isinstance(out, TextEncoderOutput) and out.embeddings.shape[0] == 2
    assert encoder.calls == calls, "a cached caption was encoded again"
    with pytest.raises(RuntimeError, match="not pre-cached"):
        trainer.encode_text(["never seen"], torch.float32)


def test_the_disk_cache_is_read_back_without_the_encoder(tmp_path):
    first = set_up()
    components, _ = _text_components(first.driver)
    first.components = components
    first.inventory = []
    first._resolve_te_cache_dirs = lambda: [str(tmp_path)]
    first._pre_cache_text_embeddings()

    second = set_up()
    components, encoder = _text_components(second.driver)
    second.components = components
    second.inventory = []
    second._resolve_te_cache_dirs = lambda: [str(tmp_path)]
    second._pre_cache_text_embeddings()
    assert encoder.calls == 0, "an entry on disk was encoded again"
    assert torch.equal(second.text_cache[""], first.text_cache[""])


def test_the_dit_never_loads_beside_the_encoder():
    trainer = set_up()
    _text_components(trainer.driver)
    with pytest.raises(RuntimeError, match="text encoder still resident"):
        trainer._materialise_transformer()


def test_update_primary_model_syncs_every_alias():
    trainer = set_up()
    old, new = _tiny_transformer(), nn.Linear(1, 1)
    _with_model(trainer, old)
    trainer._update_primary_model(new)
    assert trainer.driver.get_primary_model() is new
    assert trainer.components["unet"] is new and trainer.components["transformer"] is new
    assert trainer.transformer is new


def test_targets_and_topology_are_the_definitions_verbatim():
    driver = _driver()
    assert driver.get_lora_targets() == LORA_TARGETS
    assert driver.get_block_topology() == stand_in_definition().block_topology
    assert driver.get_te_lora_targets() == []
    assert driver.resolve_loading_dtype() == torch.bfloat16


# ── The loader ────────────────────────────────────────────────────────────


def test_the_manifest_defers_the_vendored_dit():
    from app.engine.models.families.minimax_h3_image.loader import MiniMaxH3ImageLoader

    defn = stand_in_definition()
    deferred = MiniMaxH3ImageLoader(torch.device("cpu"), defer_transformer=True).get_component_manifest(defn)
    eager = MiniMaxH3ImageLoader(torch.device("cpu")).get_component_manifest(defn)
    by_key = {s.key: s for s in eager}
    assert set(by_key) == {"tokenizer", "text_encoder", "vae", "transformer"}
    assert "transformer" not in {s.key for s in deferred}
    assert by_key["transformer"].subfolder == "transformer_diffusers"
    assert by_key["transformer"].hf_class == (
        f"{FAMILY_PKG}.vendor.transformer_minimax_h3_image.MiniMaxH3ImageTransformer2DModel"
    )
    assert by_key["vae"].subfolder == "vae_diffusers"
    assert by_key["vae"].hf_class == f"{FAMILY_PKG}.vendor.autoencoder_kl_minimax_h3_image.AutoencoderKLMiniMaxH3Image"
    assert by_key["text_encoder"].hf_class == "transformers.Qwen3VLForConditionalGeneration"
    assert by_key["text_encoder"].subfolder == "text_encoder"
    assert by_key["tokenizer"].is_torch_model is False


def test_the_vae_is_wrapped_once_in_the_shared_pixel_adapter():
    from app.engine.models.families.minimax_h3_image.loader import MiniMaxH3ImageLoader
    from app.engine.models.families.minimax_h3_shared.pixel_adapter import H3PixelAdaptedVAE

    vae = _vendor("AutoencoderKLMiniMaxH3Image").from_config(_tiny_config(VAE_CONFIG, VAE_TINY))
    wrapped = MiniMaxH3ImageLoader._post_load_vae(vae, stand_in_definition())
    assert isinstance(wrapped, H3PixelAdaptedVAE) and wrapped.inner is vae
    assert list(wrapped.config.latents_mean) == list(vae.config.latents_mean)


def test_the_six_pdd_heads_are_checked_against_the_head_grid():
    from app.engine.models.families.minimax_h3_image.loader import MiniMaxH3ImageLoader

    model = _tiny_transformer()
    assert MiniMaxH3ImageLoader._post_load_transformer(model, stand_in_definition()) is model
    short = _vendor("MiniMaxH3ImageTransformer2DModel").from_config(
        {**_tiny_config(TRANSFORMER_CONFIG, TRANSFORMER_TINY), "num_pdd_heads": 5}
    )
    with pytest.raises(ValueError, match="5 output heads"):
        MiniMaxH3ImageLoader._post_load_transformer(short, stand_in_definition())


def test_load_transformer_runs_the_single_spec_path(monkeypatch):
    from app.engine.models.families.minimax_h3_image.loader import MiniMaxH3ImageLoader

    loader = MiniMaxH3ImageLoader(torch.device("cpu"), defer_transformer=True)
    seen: dict[str, Any] = {}

    def single(spec, definition, root, dtype, device):
        seen.update(key=spec.key, sub=spec.subfolder, hook=spec.post_load_hook, root=root, dtype=dtype, device=device)
        return "model"

    monkeypatch.setattr(loader, "_load_single_spec", single)
    loader._root_path = "root"
    assert loader.load_transformer(stand_in_definition(), torch.bfloat16) == "model"
    assert seen == {"key": "transformer", "sub": "transformer_diffusers", "hook": "_post_load_transformer",
                    "root": "root", "dtype": torch.bfloat16, "device": "cpu"}


def test_the_family_registers_as_an_image_family():
    from app.engine.models.families.minimax_h3_image.family import MiniMaxH3ImageFamily
    from app.engine.models.families.minimax_h3_image.trainer import MiniMaxH3ImageTrainer

    assert MiniMaxH3ImageFamily.family_name == "minimax_h3_image"
    assert MiniMaxH3ImageFamily.archetype == "latent_diffusion"
    caps = MiniMaxH3ImageFamily.capability_overrides
    assert caps.get("is_video") is False and caps.get("has_audio") is False
    family = MiniMaxH3ImageFamily(stand_in_definition(), {})
    assert family.get_trainer_class() is MiniMaxH3ImageTrainer


def test_the_stand_in_agrees_with_the_shipped_definition():
    """The stand-in differs from the shipped YAML only where a tiny model needs it (the target
    list); every key it shares with the YAML carries the YAML's value."""
    from app.engine.models.registry import registry

    registry.initialize()
    shipped = registry.get_definition(DEFINITION_ID)
    assert shipped is not None and shipped.family == stand_in_definition().family
    for key, value in stand_in_definition().architecture_params.items():
        assert shipped.architecture_params[key] == value, key
    stand_in_stacks = [(g["attr_path"], g["count"]) for g in stand_in_definition().block_topology]
    assert stand_in_stacks == [(g["attr_path"], g["count"]) for g in shipped.block_topology]


# ── The latent-cache fingerprint ──────────────────────────────────────────


class _ConfigOnlyVAE:
    """A VAE seen only through what the fingerprint reads: its class name and its config."""

    def __init__(self, **config: Any) -> None:
        self.config = dict(config)


_VAE_CONFIG = {"latent_channels": 24, "latents_mean": [0.1] * 24, "latents_std": [1.2] * 24}


def _fingerprint(vae: Any = None, divisibility: int = 32) -> str:
    driver = _driver()
    driver.assign_components({"vae": vae if vae is not None else _ConfigOnlyVAE(**_VAE_CONFIG)})
    return driver.latent_cache_fingerprint(divisibility)


def test_the_latent_key_is_stable_and_changes_with_every_input(monkeypatch):
    from app.engine.models.families.minimax_h3_shared import pixel_adapter

    base = _fingerprint()
    assert base == _fingerprint() and len(base) == 8
    # the VAE normalisation statistics
    assert _fingerprint(_ConfigOnlyVAE(**{**_VAE_CONFIG, "latents_mean": [0.2] * 24})) != base
    assert _fingerprint(_ConfigOnlyVAE(**{**_VAE_CONFIG, "latents_std": [1.0] * 24})) != base
    # the bucket rule
    assert _fingerprint(divisibility=16) != base
    # the pixel adapter, read at call time
    monkeypatch.setattr(pixel_adapter, "PIXEL_ADAPTER_VERSION", pixel_adapter.PIXEL_ADAPTER_VERSION + 1)
    assert _fingerprint() != base


def test_the_latent_key_reads_through_the_pixel_adapter_to_the_vae():
    from app.engine.models.families.minimax_h3_shared.pixel_adapter import H3PixelAdaptedVAE

    vae = _vendor("AutoencoderKLMiniMaxH3Image").from_config(_tiny_config(VAE_CONFIG, VAE_TINY))
    wrapped = H3PixelAdaptedVAE(vae)
    assert _fingerprint(wrapped) == _fingerprint(vae)


def test_the_latent_key_needs_the_vae():
    with pytest.raises(RuntimeError, match="no VAE"):
        _driver().latent_cache_fingerprint(32)


def test_prepare_data_moves_every_latent_cache_dir_under_the_key(monkeypatch):
    """The production seam: the base ``prepare_data`` writes the inventory's cache dirs, and
    every later reader (coverage, pre-cache, batches) reads them from there."""
    import asyncio
    import os
    from types import SimpleNamespace

    from app.engine.core.pipeline import GenericTrainingPipeline

    async def base_prepare_data(self):
        self.bucket_manager = SimpleNamespace(divisibility=32)
        self.inventory = [
            {"path": "d/a.png", "cache_dir": os.path.join("ds", "latents", "1024", "original")},
            {"path": "d/b.png", "cache_dir": os.path.join("ds", "latents", "1024", "original"),
             "masked_cache_dir": os.path.join("ds", "latents", "1024", "masked")},
            {"path": "d/c.mp4", "is_video": True, "cache_dir": os.path.join("ds", "latents", "1024", "original")},
        ]

    monkeypatch.setattr(GenericTrainingPipeline, "prepare_data", base_prepare_data)
    trainer = set_up()
    trainer.components = {"vae": _ConfigOnlyVAE(**_VAE_CONFIG)}
    trainer.driver.assign_components(trainer.components)
    asyncio.run(trainer.prepare_data())

    key = trainer.driver.latent_cache_fingerprint(32)
    assert [i["path"] for i in trainer.inventory] == ["d/a.png", "d/b.png"]
    for item in trainer.inventory:
        assert item["cache_dir"] == os.path.join("ds", "latents", "1024", f"original-h3i{key}")
    assert trainer.inventory[1]["masked_cache_dir"] == os.path.join("ds", "latents", "1024", f"masked-h3i{key}")
    assert trainer.logger.named("minimax_h3_image_latent_cache_fingerprint") == [
        {"event": "minimax_h3_image_latent_cache_fingerprint", "fingerprint": key}
    ]
