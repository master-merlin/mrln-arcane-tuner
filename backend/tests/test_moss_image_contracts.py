"""LANE-165 -- the MOSS-Image (``moss_image``) family contract: ten tests, written RED first.

The tests are the spec's scenarios 2-9, 11 and 12 under the names the spec gives them (scenarios 1
and 10 are covered by the registry-wide count and VRAM coverage tests, which the plan's row 6
extends). They import ``app.engine.models.families.moss_image...`` which does not exist on this
row's head: every test fails with the marker ``LANE-165 moss_image contract`` (the red).

Everything runs on CPU against tiny configs and stubs, never the real weights (that is the GPU UAT).

The interface the later rows build (Evidence: the official pipeline
``.agent/workdir/moss/base_pipeline_moss_image.py``; the flux3 sibling ``test_flux3_contracts.py``):

* ``vendor.transformer_moss_image.MOSSImageTransformer`` -- the official class
  (``dim, ffn_dim, num_heads, num_layers, text_dim, freq_dim, latent_channels``);
  ``forward(hidden_states[B,C,H,W], timestep, encoder_hidden_states[B,S,D], encoder_attention_mask,
  return_dict, skip_layers)``; modules ``layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}``.
* ``driver.MossImageDriver(definition, device)`` -- ``assign_components``, ``encode_text(captions,
  dtype)`` -> ``TextEncoderOutput(embeddings, attention_mask)``, ``forward_pass`` (timestep passed
  through unchanged), ``get_lora_targets``, ``get_saver``. The text recipe reads
  ``architecture_params`` ``te.max_length`` (4096) and ``te.enable_thinking`` (true).
* ``trainer.MossImageTrainer`` -- ``sample_timesteps`` (sigma * 1000),
  ``encode_text`` (cache aware),
  ``_pre_cache_text_embeddings``.
* ``sampler.MossImageSampler`` -- ``_create_initial_noise``, ``denoise``, ``_flow_shift``
  (``scheduler.shift``); the prompt embedding is ``{"cond": TextEncoderOutput, "uncond": ...}``.
* ``saver.MossImageSaver`` -- kohya-layout keys with ``alpha / rank`` folded into ``lora_B``.
* ``family.MossImageFamily`` and ``definitions/{base,sft}.yaml``.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import structlog
import torch
import yaml

MARK = "LANE-165 moss_image contract"
FAMILY_PKG = "app.engine.models.families.moss_image"
FAMILY_DIR = (Path(__file__).resolve().parents[1] / "app" / "engine" / "models" / "families" / "moss_image")
BASE_ID = "moss-image-base"
SFT_ID = "moss-image-sft"

TEXT_DIM = 4096
TEXT_CAP = 4096
NUM_BLOCKS = 42
PROJECTIONS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
TARGET_NAME = re.compile(r"layers\.(\d+)\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)")
KEY_PATTERN = re.compile(
    r"diffusion_model\.layers\.(\d+)\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)\.lora_(A|B)\.weight"
)


# ── import / definition helpers: a missing module is a contract failure, not a collection error ──


def _load(module: str, attr: str):
    path = f"{FAMILY_PKG}.{module}"
    try:
        mod = importlib.import_module(path)
    except Exception as exc:  # noqa: BLE001 -- any import failure is the RED state of this contract
        pytest.fail(f"{MARK}: cannot import {path} ({type(exc).__name__}: {exc})")
    obj = getattr(mod, attr, None)
    if obj is None:
        pytest.fail(f"{MARK}: {path} has no {attr}")
    return obj


def _definition_data(definition_id: str) -> dict:
    paths = sorted((FAMILY_DIR / "definitions").glob("*.yaml"))
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("id") == definition_id:
            return data
    pytest.fail(f"{MARK}: no definition YAML with id {definition_id!r} under {FAMILY_DIR / 'definitions'}")


def _definition(definition_id: str = BASE_ID, **architecture_overrides):
    from app.engine.core.definitions import ModelDefinition

    definition = ModelDefinition(**_definition_data(definition_id))
    if architecture_overrides:
        params = {**definition.architecture_params, **architecture_overrides}
        definition = definition.model_copy(update={"architecture_params": params})
    return definition


def _tiny_transformer(num_layers: int = NUM_BLOCKS):
    cls = _load("vendor.transformer_moss_image", "MOSSImageTransformer")
    torch.manual_seed(0)
    return cls(dim=16, ffn_dim=16, num_heads=2, num_layers=num_layers, text_dim=8, freq_dim=8,
               latent_channels=8)


# ── stubs: Qwen chat-template tokenizer, Qwen3.5 text encoder, the VAE's BN stats, a spy DiT ──────


class _Encoding(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to(self, *_args, **_kwargs):
        return self


def _token_id(word: str) -> int:
    return 100 + sum((i + 1) * ord(c) for i, c in enumerate(word)) % 5000


class _StubTokenizer:
    """Whitespace tokenizer; pads on the left with id 0 only when the caller asks for padding."""

    pad_token_id = 0

    def __call__(self, text=None, return_tensors=None, padding=False, truncation=False,
                 max_length=None, **_):
        texts = [text] if isinstance(text, str) else list(text)
        rows = [[_token_id(w) for w in t.split()] for t in texts]
        if truncation and max_length is not None:
            rows = [r[:max_length] for r in rows]
        target = max(len(r) for r in rows) if padding else None
        ids, masks = [], []
        for r in rows:
            pad = max((target or 0) - len(r), 0)
            ids.append([self.pad_token_id] * pad + r)
            masks.append([0] * pad + [1] * len(r))
        if len({len(r) for r in ids}) > 1:
            raise ValueError("stub tokenizer: a ragged batch needs padding (as the real tokenizer)")
        return _Encoding(input_ids=torch.tensor(ids, dtype=torch.long),
                         attention_mask=torch.tensor(masks, dtype=torch.long))


class _StubProcessor:
    """The Qwen tokenizer surface the official pipeline uses: a chat template adding exactly ONE token."""

    def __init__(self):
        self.tokenizer = _StubTokenizer()
        self.template_calls: list[dict] = []

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        self.template_calls.append({"add_generation_prompt": add_generation_prompt, **kwargs})
        return f"<|u|> {conversation[-1]['content']}"

    def __call__(self, text=None, **kwargs):
        return self.tokenizer(text, **kwargs)


_RAMP = torch.arange(TEXT_DIM, dtype=torch.float32) * 1e-4


def _expected_hidden(ids: torch.Tensor) -> torch.Tensor:
    """The stub text encoder's final-layer hidden state for one unpadded row ``ids[S]`` -> ``[S, 4096]``."""
    pos = torch.arange(ids.shape[0], dtype=torch.float32)
    return ids.float()[:, None] * 1e-3 + pos[:, None] * 1e-5 + _RAMP[None]


class _StubQwen35(torch.nn.Module):
    """Records every call; any call once ``offloaded`` is set fails the test (a cache miss)."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.calls: list[dict] = []
        self.offloaded = False

    @property
    def dtype(self):
        return torch.float32

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        if self.offloaded:
            raise AssertionError(f"{MARK}: the text encoder was called after the offload (a cache miss)")
        self.calls.append({"input_ids": input_ids.detach().clone(),
                           "attention_mask": None if attention_mask is None else attention_mask.clone(),
                           "output_hidden_states": output_hidden_states, **kwargs})
        last = torch.stack([_expected_hidden(row) for row in input_ids])
        # Real Qwen: hidden_states[-1] is the final (normed) state == last_hidden_state; earlier ones differ.
        return SimpleNamespace(last_hidden_state=last, hidden_states=(last * 0.5, last * 0.25, last))


def _stub_vae():
    """``AutoencoderKLFlux2``'s BN surface: 128 running stats (32 latent ch x 2 x 2 patch), eps 1e-4."""
    bn = SimpleNamespace(running_mean=torch.zeros(128), running_var=torch.ones(128))
    return SimpleNamespace(bn=bn, config=SimpleNamespace(batch_norm_eps=1e-4, latent_channels=32),
                           dtype=torch.float32)


class _SpyTransformer(torch.nn.Module):
    """Records what the driver hands the DiT (the official signature); returns a fixed velocity."""

    def __init__(self, velocity=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16), requires_grad=False)
        self.config = SimpleNamespace(latent_channels=128, text_dim=TEXT_DIM, num_layers=NUM_BLOCKS)
        self.calls: list[dict] = []
        self.velocity = velocity

    @property
    def dtype(self):
        return torch.bfloat16

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask=None,
                return_dict=True, skip_layers=None):
        self.calls.append({
            "timestep": torch.as_tensor(timestep).detach().float().reshape(-1).clone(),
            "hidden_dtype": hidden_states.dtype,
            "context": encoder_hidden_states.detach().clone(),
            "mask": None if encoder_attention_mask is None else encoder_attention_mask.clone(),
            "skip_layers": None if not skip_layers else tuple(skip_layers),
            "autocast": torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"),
        })
        if self.velocity is not None:
            out = self.velocity(skip_layers).to(hidden_states.dtype)
        else:
            out = torch.zeros_like(hidden_states)
        return SimpleNamespace(sample=out) if return_dict else (out,)


class _RecordingLogWriter:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _text_output(batch: int = 1, length: int = 3):
    from app.engine.core.text_encoding import TextEncoderOutput

    return TextEncoderOutput(embeddings=torch.zeros(batch, length, TEXT_DIM),
                             attention_mask=torch.ones(batch, length, dtype=torch.bool))


def _driver(definition=None, spy=None):
    cls = _load("driver", "MossImageDriver")
    te, proc = _StubQwen35(), _StubProcessor()
    spy = spy if spy is not None else _SpyTransformer()
    driver = cls(definition if definition is not None else _definition(), torch.device("cpu"))
    driver.assign_components({
        "unet": spy, "vae": _stub_vae(), "text_encoder": te, "tokenizer": proc,
    })
    return driver, te, proc, spy


def _bare_trainer(driver, config: dict | None = None, inventory=(), tmp_path: Path | None = None):
    """A ``MossImageTrainer`` shell around a stubbed driver (no model load, no job)."""
    from app.engine.strategies.noise_interpolation import NoiseInterpolation

    cls = _load("trainer", "MossImageTrainer")
    trainer = object.__new__(cls)
    components = dict(driver.get_components())
    trainer.config = dict(config or {"mixed_precision": "fp32", "te_quantization": "none",
                                      "cache_text_embeddings": True, "sample_prompts": []})
    trainer.device = torch.device("cpu")
    trainer.definition = driver.definition
    trainer.driver = driver
    trainer.components = components
    trainer.transformer = components.get("unet")
    trainer.vae = components.get("vae")
    trainer.text_encoder = components.get("text_encoder")
    trainer.tokenizer = components.get("tokenizer")
    trainer.text_cache = {}
    trainer.inventory = list(inventory)
    trainer.logger = structlog.get_logger("moss-image-contract-test")
    trainer._log_writer = _RecordingLogWriter()
    trainer.noise_interpolation = NoiseInterpolation("linear")
    trainer.global_step = 0
    trainer.max_train_steps = 1
    if tmp_path is not None:  # one shared cache root; without it the real per-item resolution runs
        trainer._resolve_te_cache_dirs = lambda: [str(tmp_path)]
    return trainer


def _words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


# ── scenario 2: the timestep is the raw 0-1000 value, scaled exactly once ─────────────────────────


def test_moss_image_timestep_contract(monkeypatch):
    """Sigma 0.5 through the REAL ``sample_scaled`` -> driver 500.0 -> transformer 500.0 (unchanged)."""
    from app.engine.strategies.timestep_sampling import TimestepSampler

    monkeypatch.setattr(TimestepSampler, "sample",
                        staticmethod(lambda *a, **k: torch.tensor([0.5])))
    timesteps = TimestepSampler.sample_scaled("uniform", 1, torch.device("cpu"), {})
    assert float(timesteps[0]) == 500.0, f"{MARK}: sample_scaled(0.5) is {timesteps.tolist()}, not [500.0]"

    driver, _te, _proc, spy = _driver()
    trainer = _bare_trainer(driver)
    seen_by_driver: list[float] = []
    real_forward = driver.forward_pass

    def recording_forward(noisy_input, ts, text_embeddings, batch):
        seen_by_driver.append(float(ts.reshape(-1)[0]))
        return real_forward(noisy_input, ts, text_embeddings, batch)

    driver.forward_pass = recording_forward  # delegates to the real forward; only records the argument
    trainer.forward_pass(torch.randn(1, 128, 2, 2), timesteps, _text_output(), {})

    assert seen_by_driver == [500.0], f"{MARK}: the driver received {seen_by_driver}, expected [500.0]"
    assert spy.calls, f"{MARK}: the transformer was never called"
    got = spy.calls[-1]["timestep"].tolist()
    assert got == [500.0], (
        f"{MARK}: the transformer's forward received {got}; 0.5 means the driver divided by 1000, "
        "500000.0 means it multiplied by 1000 a second time")


# ── scenario 3: the training sigma distribution ───────────────────────────────────────────────────


def test_moss_image_timestep_distribution():
    """Factory path (LANE-167): family default and a user override, scenarios 2 and 3."""
    driver, _te, _proc, _spy = _driver()
    trainer = _bare_trainer(driver)

    torch.manual_seed(0)
    first = trainer.sample_timesteps(4096) / 1000.0
    torch.manual_seed(0)
    second = trainer.sample_timesteps(4096) / 1000.0
    assert torch.equal(first, second), f"{MARK}: the same seed gave two different sigma draws"
    assert bool(((first > 0) & (first < 1)).all()), f"{MARK}: sigmas must lie in (0, 1)"
    assert abs(float(first.median()) - 0.748) <= 0.02, (
        f"{MARK}: the family default median sigma {float(first.median()):.4f} is not 0.748")

    trainer.config["timestep_sampling"] = "logit_normal"
    trainer.config["logit_normal_mu"] = 0.0
    trainer.config["logit_normal_sigma"] = 1.0
    torch.manual_seed(0)
    user = trainer.sample_timesteps(4096) / 1000.0
    assert abs(float(user.median()) - 0.5) <= 0.02, (
        f"{MARK}: the user's logit-normal values did not win: median {float(user.median()):.4f}")


# ── scenario 4: interpolation direction and velocity target ───────────────────────────────────────


def test_moss_image_loss_target():
    driver, _te, _proc, _spy = _driver()
    trainer = _bare_trainer(driver)
    gen = torch.Generator().manual_seed(5)
    x0 = torch.randn(2, 128, 3, 3, generator=gen)
    noise = torch.randn(2, 128, 3, 3, generator=gen)
    timesteps = torch.full((2,), 250.0)  # sigma 0.25

    noisy = trainer.add_noise(x0, noise, timesteps)
    assert torch.allclose(noisy, 0.75 * x0 + 0.25 * noise, atol=1e-6), (
        f"{MARK}: x_t must be 0.75*x0 + 0.25*noise at sigma 0.25")
    assert not torch.allclose(noisy, 0.25 * x0 + 0.75 * noise, atol=1e-3), f"{MARK}: swapped interpolation"
    target = trainer.compute_target(x0, noise, timesteps)
    assert torch.allclose(target, noise - x0, atol=1e-6), f"{MARK}: the target must be noise - x0"
    assert not torch.allclose(target, x0 - noise, atol=1e-3), f"{MARK}: negated target"


# ── scenario 5: the sampler's precision and skip contract ─────────────────────────────────────────


def _sampler(definition=None, spy=None):
    cls = _load("sampler", "MossImageSampler")
    driver, _te, _proc, spy = _driver(definition, spy)
    pipeline = SimpleNamespace(config={}, device=torch.device("cpu"), definition=driver.definition,
                               driver=driver, transformer=spy, vae=driver.vae)
    return cls(pipeline), spy


def test_moss_image_sampler_precision_contract():
    gen = torch.Generator().manual_seed(3)
    v_cond = torch.randn(1, 128, 8, 8, generator=gen).to(torch.bfloat16)
    v_un = torch.randn(1, 128, 8, 8, generator=gen).to(torch.bfloat16)
    spy = _SpyTransformer(velocity=lambda skip: v_un if skip else v_cond)
    sampler, spy = _sampler(spy=spy)
    embeds = {"cond": _text_output(1, 5), "uncond": _text_output(1, 2)}
    embeds["cond"].embeddings.normal_(generator=gen)
    embeds["uncond"].embeddings.normal_(generator=gen)

    noise = sampler._create_initial_noise(128, 128, torch.Generator().manual_seed(0))
    assert noise.dtype == torch.float32 and tuple(noise.shape) == (1, 128, 8, 8), (
        f"{MARK}: initial noise is {noise.dtype} {tuple(noise.shape)}, expected fp32 (1, 128, 8, 8)")
    out = sampler.denoise(noise, embeds, 1, 3.0, 0)

    assert len(spy.calls) == 2, f"{MARK}: CFG needs a cond and an uncond forward per step, got {len(spy.calls)}"
    assert not [c for c in spy.calls if c["autocast"]], (
        f"{MARK}: an autocast region is active around the DiT (autocast-sampler-collapse)")
    cond_calls = [c for c in spy.calls if c["skip_layers"] is None]
    skip_calls = [c for c in spy.calls if c["skip_layers"] is not None]
    assert len(cond_calls) == 1 and len(skip_calls) == 1, f"{MARK}: block skipping must hit exactly one branch"
    assert skip_calls[0]["skip_layers"] == (22,), (
        f"{MARK}: the negative branch skips block 22, got {skip_calls[0]['skip_layers']}")
    assert torch.equal(skip_calls[0]["context"], embeds["uncond"].embeddings), (
        f"{MARK}: block 22 must be skipped on the NEGATIVE (uncond) branch only")
    assert torch.equal(cond_calls[0]["context"], embeds["cond"].embeddings)

    from app.engine.models.families.flux2.utils import bn_denormalize, unpatchify_latents

    combined = v_un.float() + 3.0 * (v_cond.float() - v_un.float())  # fp32 combine of the bf16 outputs
    expected = unpatchify_latents(bn_denormalize(noise - combined, sampler.pipeline.vae))
    assert out.dtype == torch.float32, f"{MARK}: the Euler trajectory must stay fp32, got {out.dtype}"
    assert torch.allclose(out, expected, rtol=0, atol=1e-6), (
        f"{MARK}: CFG combine / Euler update is not the fp32 computation "
        f"(max diff {float((out - expected).abs().max()):.3e})")


def test_moss_image_sampler_grid_parity():
    """The preview grid is the vendored scheduler's (shifted, then snapped to the training grid), step for step."""
    sched_cls = _load("vendor.scheduler_moss_image", "MOSSImageScheduler")
    scheduler = sched_cls(shift=5.0)
    scheduler.set_timesteps(50)
    expected = (scheduler.sigmas[:-1] * 1000.0).float()

    sampler, spy = _sampler()
    embeds = {"cond": _text_output(1, 5), "uncond": _text_output(1, 2)}
    noise = sampler._create_initial_noise(128, 128, torch.Generator().manual_seed(0))
    sampler.denoise(noise, embeds, 50, 3.0, 0)

    seen = torch.stack([c["timestep"].reshape(-1)[0] for c in spy.calls if c["skip_layers"] is None])
    assert seen.shape == expected.shape, (
        f"{MARK}: the sampler ran {seen.shape[0]} steps, the vendored scheduler has {expected.shape[0]}")
    assert torch.allclose(seen, expected, rtol=0, atol=1e-3), (
        f"{MARK}: sampler timesteps differ from the vendored scheduler's (last {seen[-1]:.3f} vs {expected[-1]:.3f})")


def test_moss_image_gradient_checkpointing():
    """The default config (gradient_checkpointing on) must set up and backprop into the LoRA params."""
    from peft import LoraConfig, get_peft_model

    model = _tiny_transformer(num_layers=2)
    model.enable_gradient_checkpointing()  # the call pipeline_optimization._configure_gradient_checkpointing makes
    forwards = {"n": 0}
    layer0 = model.layers[0]
    original = layer0.forward

    def counting(*args, **kwargs):
        forwards["n"] += 1
        return original(*args, **kwargs)

    layer0.forward = counting
    peft_model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=4, target_modules=r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)"))
    peft_model.train()
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    x = torch.randn(1, 8, 4, 4)
    ctx = torch.randn(1, 3, 8)
    out = peft_model(hidden_states=x, timestep=torch.tensor([500.0]), encoder_hidden_states=ctx,
                     return_dict=False)[0]
    out.float().pow(2).mean().backward()
    assert forwards["n"] == 2, (
        f"{MARK}: block 0 ran {forwards['n']}x for forward+backward, checkpointing recomputes it (expected 2)")
    grads = [p.grad for n, p in peft_model.named_parameters() if "lora_" in n]
    assert grads and all(g is not None for g in grads) and any(float(g.abs().sum()) > 0 for g in grads), (
        f"{MARK}: LoRA params received no gradient under checkpointing")


# ── scenario 6: the 252 LoRA targets ──────────────────────────────────────────────────────────────


def test_moss_image_lora_targets():
    model = _tiny_transformer()
    driver, _te, _proc, _spy = _driver(spy=model)
    targets = list(driver.get_lora_targets())
    executed = {name for name, _ in model.named_modules() if TARGET_NAME.fullmatch(name)}

    assert len(targets) == len(set(targets)) == 252, (
        f"{MARK}: {len(targets)} targets ({len(set(targets))} unique), expected exactly 252")
    assert set(targets) == executed, (
        f"{MARK}: targets differ from the model's block projections: missing {sorted(executed - set(targets))[:4]}, "
        f"extra {sorted(set(targets) - executed)[:4]}")
    for block in range(NUM_BLOCKS):
        mine = {t for t in targets if t.startswith(f"layers.{block}.proj.")}
        assert mine == {f"layers.{block}.proj.{p}" for p in PROJECTIONS}, (
            f"{MARK}: block {block} must carry exactly {PROJECTIONS}, got {sorted(mine)}")
    assert list(_definition().lora_targetable_modules or []) == targets or set(
        _definition().lora_targetable_modules or []) == set(targets), (
        f"{MARK}: the YAML's lora_targetable_modules must equal the resolved list")


# ── scenario 7: the saved bytes ───────────────────────────────────────────────────────────────────


def _peft_tiny(rank: int, alpha: float, use_rslora: bool):
    from peft import LoraConfig, get_peft_model

    model = _tiny_transformer()
    peft_model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha, use_rslora=use_rslora,
        target_modules=r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)"))
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_" in name:
                param.copy_(torch.randn(param.shape, generator=gen) * 0.05)
    return peft_model


def _save(model, path: Path, extra_config: dict | None = None):
    saver_cls = _load("saver", "MossImageSaver")
    config = {"save_precision": "bf16", "global_triggerword": "ohwx", "optimizer_type": "adamw",
              "learning_rate": 1e-4, "max_train_steps": 10, "lora_name": "moss-contract",
              **(extra_config or {})}
    saver_cls().save({"unet": model, "config": config}, path)
    assert path.is_file(), f"{MARK}: the saver wrote no file at {path}"
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}, dict(f.metadata() or {})


def _effective_delta(peft_model, module_name: str, scale: float) -> torch.Tensor:
    layer = dict(peft_model.named_modules())[f"base_model.model.{module_name}"]
    return scale * (layer.lora_B["default"].weight.detach().float()
                    @ layer.lora_A["default"].weight.detach().float())


_SCALING_CASES = (
    (16, 8.0, False, 8.0 / 16),      # classic LoRA: alpha / rank
    (16, 16.0, True, 16.0 / 4.0),    # rsLoRA: alpha / sqrt(rank) = 4.0
)


def test_moss_image_saver_keys(tmp_path):
    for case, (rank, alpha, use_rslora, scale) in enumerate(_SCALING_CASES):
        _check_saved_bytes(tmp_path / f"case{case}.safetensors", rank, alpha, use_rslora, scale)


def _check_saved_bytes(path, rank, alpha, use_rslora, scale):
    model = _peft_tiny(rank, alpha, use_rslora)
    tensors, metadata = _save(model, path)

    assert len(tensors) == 504, f"{MARK}: {len(tensors)} keys saved, expected 504 (one A and one B per target)"
    bad = [k for k in tensors if not KEY_PATTERN.fullmatch(k)]
    assert not bad, f"{MARK}: keys must be diffusion_model.layers.N.proj.<p>.lora_A/B.weight, got {bad[:4]}"
    for projection in ("q", "k", "v"):
        assert any(f".proj.{projection}.lora_A.weight" in k for k in tensors), (
            f"{MARK}: {projection} must be its own key (no fused qkv)")
    assert not [k for k in tensors if "alpha" in k], f"{MARK}: the file must carry no alpha tensor"

    driver, _te, _proc, _spy = _driver(spy=_tiny_transformer())
    target_set = set(driver.get_lora_targets())
    unmatched = [k for k in tensors
                 if re.sub(r"\.lora_[AB]\.weight$", "", k.removeprefix("diffusion_model.")) not in target_set]
    assert not unmatched, f"{MARK}: {len(unmatched)} keys name no Scenario-6 target, e.g. {unmatched[:3]}"

    assert metadata.get("modelspec.trigger_phrase") == "ohwx", f"{MARK}: trigger-phrase metadata missing"
    assert float(metadata.get("ss_network_alpha", "nan")) == 16.0, (
        f"{MARK}: ss_network_alpha must equal the rank 16 (scale folded into lora_B), "
        f"got {metadata.get('ss_network_alpha')}")

    for key, lora_a in tensors.items():
        if not key.endswith(".lora_A.weight"):
            continue
        module = key.removeprefix("diffusion_model.").removesuffix(".lora_A.weight")
        lora_b = tensors[key.replace(".lora_A.weight", ".lora_B.weight")]
        exported = lora_b.float() @ lora_a.float()
        expected = _effective_delta(model, module, scale)
        rel = float((exported - expected).norm() / expected.norm())
        assert rel <= 1e-2, (
            f"{MARK}: {module}: exported B@A differs from the trained delta by {rel:.3e} relative "
            f"(scale {scale}); an unscaled lora_B fails here")


# ── scenario 8: the text-embedding cache key ──────────────────────────────────────────────────────


def _precache_run(tmp_path, captions, **architecture_overrides):
    """A cold trainer (empty memory) over a shared disk cache; returns (TE forwards, hint count, hints)."""
    driver, te, _proc, _spy = _driver(_definition(**architecture_overrides))
    inventory = [{"path": f"/datasets/probe/img_{i}.png", "caption": c} for i, c in enumerate(captions)]
    trainer = _bare_trainer(driver, inventory=inventory, tmp_path=tmp_path)
    hints = set(trainer._build_caption_hints())
    trainer._pre_cache_text_embeddings()
    return len(te.calls), hints


def test_moss_image_te_cache_key(tmp_path):
    captions = ["a red fox", "two cats on a sofa", "a lighthouse at dusk"]
    cold, hints_a = _precache_run(tmp_path, captions)
    assert cold == len(hints_a) > 0, f"{MARK}: a cold run must encode every hint ({cold} vs {len(hints_a)})"

    warm, _ = _precache_run(tmp_path, captions)
    assert warm == 0, f"{MARK}: an unchanged run must hit the cache fully, it ran {warm} text forwards"

    changed = list(captions)
    changed[1] = "two dogs on a sofa"
    partial, hints_c = _precache_run(tmp_path, changed)
    assert partial == len(hints_c - hints_a) > 0, (
        f"{MARK}: one changed caption must miss only its own entries: {partial} forwards, "
        f"expected {len(hints_c - hints_a)}")

    capped, _ = _precache_run(tmp_path, captions, **{"te.max_length": 2048})
    assert capped == len(hints_a), f"{MARK}: a different token cap must miss every entry ({capped} forwards)"

    unthought, _ = _precache_run(tmp_path, captions, **{"te.enable_thinking": False})
    assert unthought == len(hints_a), (
        f"{MARK}: a different chat-template flag must miss every entry ({unthought} forwards)")


def test_moss_image_te_cache_dedup(tmp_path):
    """Verify 1.02: the disk key is the caption's identity, never the first image that carried it."""
    cold, hints_a = _precache_run(tmp_path, ["fox", "fox"])
    assert cold == len(hints_a) > 0, f"{MARK}: two images captioned 'fox' must encode 'fox' once ({cold} forwards)"
    sha = hashlib.sha256(b"fox").hexdigest()
    assert len(list(tmp_path.rglob(f"*{sha}*"))) == 1, f"{MARK}: one 'fox' entry on disk, not one per image"

    # The first image's caption changes: the second image's 'fox' must still hit.
    partial, hints_c = _precache_run(tmp_path, ["a hen", "fox"])
    assert partial == len(hints_c - hints_a) == 1, (
        f"{MARK}: only the changed caption may miss; {partial} forwards (a filename keyed on the "
        f"source image re-encodes the unchanged 'fox')")


def _dataset_items(root: Path, name: str, captions: list[str]) -> list[dict]:
    """Inventory items whose latent ``cache_dir`` places them in dataset ``name`` (the real layout)."""
    cache_dir = root / name / ".cache" / BASE_ID / "v1" / "latents" / "original" / "1024"
    return [{"path": str(root / name / f"img_{i}.png"), "caption": c, "cache_dir": str(cache_dir)}
            for i, c in enumerate(captions)]


def _multi_dataset_run(items: list[dict]) -> tuple[int, set[str]]:
    driver, te, _proc, _spy = _driver()
    trainer = _bare_trainer(driver, inventory=items)  # the real per-item cache-dir resolution
    trainer._pre_cache_text_embeddings()
    return len(te.calls), set(trainer._build_caption_hints())


def test_moss_image_te_cache_multi_dataset(tmp_path):
    """Verify 1.02: each entry persists under its owning dataset, so a dataset trained alone hits."""
    ds_a = _dataset_items(tmp_path, "ds_a", ["a red fox", "two cats on a sofa"])
    ds_b = _dataset_items(tmp_path, "ds_b", ["an owl at night"])
    together, hints = _multi_dataset_run(ds_a + ds_b)
    assert together == len(hints) > 0, f"{MARK}: a cold two-dataset run encodes every hint once ({together})"

    owl = hashlib.sha256(b"an owl at night").hexdigest()
    assert list((tmp_path / "ds_b").rglob(f"*{owl}*")), f"{MARK}: ds_b's caption must persist under ds_b"
    assert not list((tmp_path / "ds_a").rglob(f"*{owl}*")), f"{MARK}: ds_b's caption landed under ds_a"

    b_alone, _ = _multi_dataset_run(ds_b)
    assert b_alone == 0, f"{MARK}: ds_b trained alone must hit its own entries, it ran {b_alone} forwards"
    a_alone, _ = _multi_dataset_run(ds_a)
    assert a_alone == 0, f"{MARK}: ds_a trained alone must hit its own entries, it ran {a_alone} forwards"


# ── scenario 11: the definition's preview defaults reach a normal sample prompt ──────────────────


def test_moss_image_preview_defaults():
    """Verify 3.01: an UNSET field resolves to the definition's 50 / 3.0; an explicit value, 20 / 3.5
    included, is kept; the broadcast report names the executed steps and guidance."""
    from structlog.testing import capture_logs

    from app.engine.models.base import SamplePromptConfig

    assert SamplePromptConfig().num_inference_steps is None and SamplePromptConfig().guidance_scale is None, (
        f"{MARK}: the schema must carry 'unset' as None, not a default that collides with an explicit value")
    sampler, _spy = _sampler()
    sampler.config["sample_prompts"] = [
        SamplePromptConfig(prompt="unset"),
        SamplePromptConfig(prompt="collides", num_inference_steps=20, guidance_scale=3.5, width=1024),
        SamplePromptConfig(prompt="other", num_inference_steps=30, guidance_scale=2.0, width=512),
        {"prompt": "legacy dict", "num_inference_steps": None},
    ]
    unset, collides, other, legacy = sampler._get_sample_prompts()

    def got(c):
        return (c["num_inference_steps"], c["guidance_scale"], c["width"], c["height"])

    assert got(unset) == (50, 3.0, 1024, 1024), f"{MARK}: unset resolved to {got(unset)}, not 50 / 3.0"
    assert got(collides) == (20, 3.5, 1024, 1024), f"{MARK}: explicit 20 / 3.5 was overridden: {got(collides)}"
    assert got(other) == (30, 2.0, 512, 1024), f"{MARK}: explicit values must win, got {got(other)}"
    assert legacy["num_inference_steps"] == 50 and legacy["guidance_scale"] == 3.0

    # The report is built from the same resolved config the sampler executes.
    with capture_logs() as logs:
        sampler._broadcast_sample(Path("s.png"), 0, 1, collides, raw_prompt="collides")
    event = next(e for e in logs if e["event"] == "sample_generated")
    assert (event["num_inference_steps"], event["guidance_scale"]) == (20, 3.5), (
        f"{MARK}: the report names {event['num_inference_steps']} / {event['guidance_scale']}, not the executed 20 / 3.5")


# ── scenario 9: the vendored package's imports ────────────────────────────────────────────────────

_ALLOWED_ROOTS = {"torch", "diffusers", "transformers", "app", "__future__"} | set(sys.stdlib_module_names)


def _import_violations(source: str) -> list[str]:
    """Every import outside torch/diffusers/transformers/stdlib/project, and every trust_remote_code=True."""
    problems: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            problems += [f"import {a.name}" for a in node.names if a.name.split(".")[0] not in _ALLOWED_ROOTS]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if (node.module or "").split(".")[0] not in _ALLOWED_ROOTS:
                problems.append(f"from {node.module} import ...")
        elif isinstance(node, ast.keyword) and node.arg == "trust_remote_code":
            if not (isinstance(node.value, ast.Constant) and node.value.value is False):
                problems.append("trust_remote_code passed non-False")
    return problems


def test_moss_image_vendor_imports(tmp_path):
    # Positive control (CONVENTIONS Tests 11): the scanner flags what it claims to flag.
    assert _import_violations("import requests\n"), f"{MARK}: scanner missed a third-party import"
    assert _import_violations("from numpy import array\n"), f"{MARK}: scanner missed a from-import"
    assert _import_violations("m = X.from_pretrained('p', trust_remote_code=True)\n"), (
        f"{MARK}: scanner missed trust_remote_code=True")
    assert not _import_violations("import torch\nfrom diffusers import ConfigMixin\nfrom . import x\n"
                                  "import math\nfrom app.engine import y\n"), f"{MARK}: scanner false positive"

    vendor = FAMILY_DIR / "vendor"
    files = sorted(vendor.glob("*.py")) if vendor.is_dir() else []
    names = {p.name for p in files}
    assert {"transformer_moss_image.py", "scheduler_moss_image.py"} <= names, (
        f"{MARK}: vendor/ must hold the transformer and the scheduler, found {sorted(names)}")
    for family_file in sorted(FAMILY_DIR.rglob("*.py")):
        found = _import_violations(family_file.read_text(encoding="utf-8")) if family_file.parent == vendor else [
            p for p in _import_violations(family_file.read_text(encoding="utf-8")) if "trust_remote_code" in p]
        assert not found, f"{MARK}: {family_file.relative_to(FAMILY_DIR)} breaks the import rule: {found}"


# ── scenario 11: the two definitions ──────────────────────────────────────────────────────────────

_REPO_PIN = re.compile(r"^huggingface:OpenMOSS-Team/MOSS-Image-(Base|SFT)@[0-9a-f]{40}$")


def test_moss_image_definition_defaults():
    base, sft = _definition_data(BASE_ID), _definition_data(SFT_ID)
    for data in (base, sft):
        assert data["family"] == "moss_image"
        assert int(data["defaults"]["num_inference_steps"]) == 50, f"{MARK}: {data['id']} steps"
        assert float(data["defaults"]["guidance_scale"]) == 3.0, f"{MARK}: {data['id']} CFG"
        params = data["architecture_params"]
        assert float(params["scheduler.shift"]) == 5.0, f"{MARK}: {data['id']} scheduler.shift"
        assert "scheduler.shift" in data["enrich_pinned_keys"], f"{MARK}: {data['id']} scheduler.shift pin"
        assert "scheduler.training_shift" not in params, f"{MARK}: {data['id']} still has training_shift"
        assert int(params["te.max_length"]) == TEXT_CAP and params["te.enable_thinking"] is True, (
            f"{MARK}: {data['id']} text recipe keys")
        assert params["sampling.seed_prompt_defaults"] is True, (
            f"{MARK}: {data['id']} must opt in to row seeding (sampling.seed_prompt_defaults)")
        assert _REPO_PIN.match(data["components"]["repo"]["path"]), (
            f"{MARK}: {data['id']} must pin a 40-hex revision: {data['components']['repo']['path']}")
    assert base["components"]["repo"]["path"] != sft["components"]["repo"]["path"]

    def stripped(data):
        copy = yaml.safe_load(yaml.safe_dump(data))
        for key in ("id", "name"):
            copy.pop(key, None)
        copy["components"]["repo"].pop("path", None)
        return copy

    assert stripped(base) == stripped(sft), f"{MARK}: the definitions may differ only in repo + revision (and id/name)"

    from app.engine.models.registry import ModelRegistry

    ModelRegistry.initialize()
    family_cls = _load("family", "MossImageFamily")
    assert ModelRegistry.get_family_class("moss_image") is family_cls
    trainer_cls = _load("trainer", "MossImageTrainer")
    assert family_cls(_definition(), {}).get_trainer_class() is trainer_cls
    for definition_id in (BASE_ID, SFT_ID):
        driver, _te, _proc, _spy = _driver(_definition(definition_id))
        assert type(driver.get_saver()) is _load("saver", "MossImageSaver")

    flat = _definition()
    sampler, _spy2 = _sampler(flat)
    assert sampler._flow_shift() == 5.0, f"{MARK}: the sampler must still use scheduler.shift 5.0"

    # the opt-in reaches the capabilities the form reads; a definition without the key stays false
    from app.engine.core.archetypes import resolve_capabilities

    for definition_id in (BASE_ID, SFT_ID):
        assert resolve_capabilities(ModelRegistry.get_definition(definition_id))["capabilities"][
            "seed_prompt_defaults"] is True, f"{MARK}: {definition_id} capabilities must expose the opt-in"
    others = [d for d in ModelRegistry._definitions.values() if d.family != "moss_image"]
    assert others and not [d.id for d in others
                           if resolve_capabilities(d)["capabilities"]["seed_prompt_defaults"]], (
        f"{MARK}: no other definition may opt in")


# ── scenario 12: MOSS's own text recipe ───────────────────────────────────────────────────────────


def test_moss_image_text_encoding_contract(tmp_path):
    captions = ["a red fox", _words("w", 40), "", _words("x", 5000)]
    expected_tokens = [4, 41, 1, TEXT_CAP]  # the template adds one token; the last is truncated at 4096

    definition = _definition()
    driver, te, proc, _spy = _driver(definition)
    uncached: dict[str, object] = {}
    for caption, n_tokens in zip(captions, expected_tokens, strict=True):
        before = len(te.calls)
        out = driver.encode_text([caption], torch.float32)
        assert len(te.calls) == before + 1, f"{MARK}: each caption is its own forward"
        call = te.calls[-1]
        assert call["input_ids"].shape == (1, n_tokens), (
            f"{MARK}: caption {caption[:12]!r} entered the TE as {tuple(call['input_ids'].shape)}, "
            f"expected (1, {n_tokens})")
        assert 0 not in call["input_ids"], f"{MARK}: a padding token reached the text encoder"
        assert call["attention_mask"] is None or bool(call["attention_mask"].all()), (
            f"{MARK}: the forward must be unpadded")
        emb, mask = out.embeddings, out.attention_mask
        assert emb.shape == (1, n_tokens, TEXT_DIM), f"{MARK}: embedding {tuple(emb.shape)}"
        assert torch.allclose(emb[0], _expected_hidden(call["input_ids"][0]), atol=1e-6), (
            f"{MARK}: the embedding is not the final-layer hidden state (a stacked layer or a wrong layer)")
        assert mask is not None and mask.dtype == torch.bool and mask.shape == (1, n_tokens) and bool(mask.all()), (
            f"{MARK}: the valid-token mask must carry the caption's own {n_tokens} tokens")
        uncached[caption] = out
    assert proc.template_calls and all(c.get("enable_thinking") is True and c["add_generation_prompt"]
                                       for c in proc.template_calls), (
        f"{MARK}: the chat template must be applied with enable_thinking=True and the generation prompt")

    batch = driver.encode_text(captions[:3], torch.float32)
    assert batch.embeddings.shape == (3, 41, TEXT_DIM) and batch.attention_mask.sum(dim=1).tolist() == [4, 41, 1], (
        f"{MARK}: a batch is right-padded to its longest with a mask of the true lengths")
    assert len(te.calls) == 4 + 3, f"{MARK}: a batch of 3 captions must still be 3 batch-size-1 forwards"

    # the cached path: pre-cache, offload, then serve every caption from the cache
    inventory = [{"path": f"/datasets/probe/img_{i}.png", "caption": c} for i, c in enumerate(captions) if c]
    trainer = _bare_trainer(driver, inventory=inventory, tmp_path=tmp_path)
    trainer._pre_cache_text_embeddings()
    te.offloaded = True
    for caption in captions:
        cached = trainer.encode_text([caption], torch.float32)
        ref = uncached[caption]
        assert torch.equal(cached.embeddings.cpu(), ref.embeddings.cpu()), (
            f"{MARK}: cached != uncached embedding for {caption[:12]!r}")
        assert torch.equal(cached.attention_mask.cpu(), ref.attention_mask.cpu()), (
            f"{MARK}: cached != uncached mask for {caption[:12]!r}")


# ── verify 1.01: the preview encode path with the REAL TextEncoderOutput ──────────────────────────


def test_moss_image_encode_prompt_real_types():
    """``encode_prompt`` through the trainer's real encoding path (real ``TextEncoderOutput``) reaches the DiT."""
    cls = _load("sampler", "MossImageSampler")
    driver, _te, _proc, spy = _driver()
    trainer = _bare_trainer(driver, config={"mixed_precision": "fp32", "te_quantization": "none",
                                            "cache_text_embeddings": False, "sample_prompts": []})
    sampler = cls(trainer)
    sampler.device = torch.device("cpu")

    embeds = sampler.encode_prompt("a red fox")
    assert embeds["cond"].embeddings.shape[-1] == TEXT_DIM and embeds["uncond"].embeddings.shape[0] == 1

    noise = sampler._create_initial_noise(128, 128, torch.Generator().manual_seed(0))
    sampler.denoise(noise, embeds, 1, 3.0, 0)
    assert len(spy.calls) == 2, f"{MARK}: the first denoising call must be reached ({len(spy.calls)} forwards)"
