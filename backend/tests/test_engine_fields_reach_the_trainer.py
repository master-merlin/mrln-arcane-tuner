"""LANE-168: every engine field the training form SHOWS for a family is observed on that family's
training path, or is hidden by a declared capability.

Before this lane the form showed every ENGINE field to every family while some were dropped on the
way to the trainer: ``bernini_r`` and ``hidream_o1`` drew their own timesteps and never read
``timestep_sampling``; ``hidream_o1`` drew fresh noise and never applied ``noise_offset``;
``min_snr_gamma`` was applied by the SDXL trainer alone. Nothing in the suite compared "shown in
``field_visibility``" with "read by the trainer", so each of those stayed invisible.

CPU only, no weights, no network. Every family trainer is the real class built with
``object.__new__`` plus its real ``_setup_family`` (real driver, no model). The registry-wide
contract drives the REAL training step (``forward_loss_step``: noise, pack, timestep draw, noise-add,
the family's forward, target, loss) through the family's own driver into a model DOUBLE that records
what the denoiser receives -- its timestep and its noisy input -- and the step's target and loss are
read back from what the step returned. Only the denoiser (and a VAE's normalisation statistics) is
replaced; every family hook between the loop and the model runs for real. The positive controls at
the end break one link of that path each and prove the contract then reports the family.
"""

from __future__ import annotations

import random
from contextlib import ExitStack
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest
import structlog
import torch
from torch import nn

from app.engine.core.archetypes import resolve_capabilities
from app.engine.core.config_allowlist import apply_capability_allowlist
from app.engine.core.pipeline.pipeline_train import PipelineTrainMixin
from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.factories.optimizer import LRSchedulerFactory, OptimizerFactory
from app.engine.models.registry import registry
from app.engine.strategies.noise_interpolation import NoiseInterpolation
from app.engine.strategies.timestep_sampling import TimestepSampler

BASE_CONFIG = {
    "mixed_precision": "fp32",
    "te_quantization": "none",
    "cache_text_embeddings": True,
    "sample_prompts": [],
}
ENGINE_FIELDS = ("timestep_sampling", "noise_offset", "min_snr_gamma", "optimizer_type", "lr_scheduler")
# Sentinels no family default equals, so a call that carries one proves the config value arrived.
TIMESTEP_SENTINEL = "radc"
OPTIMIZER_SENTINEL = "Lion"
SCHEDULER_SENTINEL = "polynomial"


@pytest.fixture(scope="module", autouse=True)
def _registry():
    registry.initialize()
    return registry


# ── Shared helpers ───────────────────────────────────────────────────────────


def _definitions():
    """Every definition of every family (unavailable ones included -- a hidden family is the defect)."""
    registry.initialize()
    return list(registry._definitions.values())


def _trainer(defn, **config):
    """The real trainer class of ``defn``'s family around its real driver (no weights loaded)."""
    family = object.__new__(registry.get_family_class(defn.family))
    family.definition = defn
    family.config = {}
    trainer = object.__new__(family.get_trainer_class())
    trainer.definition = defn
    # minimax_h3 refuses any precision but bf16 (its own explicit validation), whatever the field under test
    base = {**BASE_CONFIG, "mixed_precision": "bf16"} if defn.family == "minimax_h3" else BASE_CONFIG
    trainer.config = {**base, **config}
    trainer.device = torch.device("cpu")
    trainer.logger = structlog.get_logger("lane-168-test")
    trainer.components = {}
    trainer.epoch = 0
    trainer.global_step = 30
    trainer.max_train_steps = 60
    trainer.optimizer = None
    trainer._setup_family()
    try:  # SDXL's discrete draw and Min-SNR read the driver's DDPM scheduler
        trainer.scheduler = trainer.driver.init_scheduler()
    except Exception:  # noqa: BLE001 -- families without a driver scheduler never read self.scheduler
        trainer.scheduler = None
    return trainer


class _SamplerSpy:
    """Records every ``TimestepSampler.sample`` / ``sample_scaled`` call and still draws for real.

    ``force`` (a float in [0, 1]) replaces the VALUES of each real draw after it was drawn, so the
    random stream is consumed exactly as without it: two steps that differ only in ``force`` differ
    only in what the factory returned, and a model that receives the same timestep under both
    trains on a draw of its own.
    """

    def __init__(self, monkeypatch):
        self.calls: list[SimpleNamespace] = []
        self.force: float | None = None
        real_sample = TimestepSampler.sample
        real_scaled = TimestepSampler.sample_scaled

        def sample(mode, bs, device, config, *, latents=None, progress=0.0):
            self.calls.append(SimpleNamespace(
                kind="sample", mode=mode, bs=bs, device=device, config=config, progress=progress))
            drawn = real_sample(mode, bs, device, config, latents=latents, progress=progress)
            return drawn if self.force is None else torch.full_like(drawn, self.force)

        def sample_scaled(mode, bs, device, config, *, scale=1000.0, latents=None, progress=0.0):
            self.calls.append(SimpleNamespace(
                kind="sample_scaled", mode=mode, bs=bs, device=device, config=config, progress=progress))
            return real_scaled(mode, bs, device, config, scale=scale, latents=latents, progress=progress)

        monkeypatch.setattr(TimestepSampler, "sample", staticmethod(sample))
        monkeypatch.setattr(TimestepSampler, "sample_scaled", staticmethod(sample_scaled))

    def draws(self, mode=None):
        """The ``sample`` calls (``sample_scaled`` delegates to ``sample``, so this counts each draw once)."""
        return [c for c in self.calls if c.kind == "sample" and (mode is None or c.mode == mode)]


@pytest.fixture
def spy(monkeypatch):
    return _SamplerSpy(monkeypatch)


# ── 1. bernini_r ─────────────────────────────────────────────────────────────


def test_bernini_r_timestep_mode_reaches_the_factory(spy):
    defn = registry.get_definition("bernini-r-1.3b")
    assert defn is not None, "definition bernini-r-1.3b is not in the registry"
    trainer = _trainer(defn, timestep_sampling="radc", radc_start=0.8)
    trainer.global_step, trainer.max_train_steps = 30, 60

    timesteps = trainer.sample_timesteps(4)

    draws = spy.draws("radc")
    assert draws, "TimestepSampler.sample was not called for mode 'radc' (bernini_r draws its own torch.rand)"
    assert [(c.mode, c.bs, c.progress) for c in spy.draws()] == [("radc", 4, 0.5)], (
        f"expected exactly one factory draw (radc, batch 4, progress 0.5), got "
        f"{[(c.mode, c.bs, c.progress) for c in spy.draws()]}")
    assert draws[0].device == trainer.device, f"the factory got device {draws[0].device}, not the trainer's"
    assert draws[0].config.get("radc_start") == 0.8, "the factory did not receive the trainer's config"
    assert timesteps.shape == (4,) and bool((timesteps >= 0).all() and (timesteps <= 1000).all()), (
        f"timesteps must be 4 values in [0, 1000], got {timesteps.tolist()}")


# ── 2. hidream_o1 ────────────────────────────────────────────────────────────


class _SigmaCapturingModel(nn.Module):
    """Stands in for the Qwen3-VL pixel model: returns a grad-carrying x_pred and keeps the timestep."""

    def __init__(self, patch_dim: int):
        super().__init__()
        self.config = MagicMock()
        self.patch_dim = patch_dim
        self.timesteps: list[torch.Tensor] = []

    def forward(self, **kwargs):
        self.timesteps.append(kwargs["timestep"].detach().clone())
        out = MagicMock()
        out.x_pred = torch.randn(1, kwargs["vinputs"].shape[1], self.patch_dim, requires_grad=True)
        return out


def _hidream_trainer_and_batch(**config):
    from app.engine.models.families.hidream_o1.trainer import PATCH_SIZE
    from tests.test_hidream_o1_integration import _make_mock_text_sample

    defn = registry.get_definition("hidream_o1_image")
    assert defn is not None, "definition hidream_o1_image is not in the registry"
    trainer = _trainer(defn, **config)
    model = _SigmaCapturingModel(3 * PATCH_SIZE * PATCH_SIZE)
    trainer.driver.assign_components({"unet": model})
    trainer.components["unet"] = model
    trainer.processor = MagicMock()
    trainer.tokenizer = MagicMock()
    trainer.autocast_dtype = torch.float32
    side = PATCH_SIZE * 2
    n_patches = (side // PATCH_SIZE) ** 2
    batch = {"images": torch.randn(8, 3, side, side), "captions": ["a caption"] * 8}
    text_sample = _make_mock_text_sample(seq_len=n_patches, n_visual=n_patches)
    return trainer, model, batch, text_sample


def test_hidream_o1_timestep_mode_reaches_the_factory(spy):
    from app.engine.models.families.hidream_o1.trainer import T_EPS

    target = "app.engine.models.families.hidream_o1.trainer.build_t2i_text_sample"
    configs = (
        ("logit_normal", dict(timestep_sampling="logit_normal", logit_normal_mu=0.0, logit_normal_sigma=1.0)),
        ("radc", dict(timestep_sampling="radc", radc_start=0.8)),
    )
    for mode, config in configs:
        trainer, model, batch, text_sample = _hidream_trainer_and_batch(**config)
        trainer.global_step, trainer.max_train_steps = 30, 60
        drivers = (("forward_pass", lambda: trainer.forward_pass(batch["images"], None, None, batch)),
                   ("compute_loss", lambda: trainer.compute_loss(batch)))
        for name, run in drivers:
            spy.calls.clear()
            model.timesteps.clear()
            torch.manual_seed(0)
            with patch(target, return_value=text_sample):
                run()
            assert spy.draws(mode), (
                f"TimestepSampler.sample was not called for mode '{mode}' (hidream_o1 draws its own _sample_sigma)")
            assert [(c.mode, c.bs) for c in spy.draws()] == [(mode, 8)], (
                f"{name}: expected one factory draw ({mode}, batch 8), got "
                f"{[(c.mode, c.bs) for c in spy.draws()]}")
            if mode == "radc":
                assert spy.draws()[0].progress == 0.5, (
                    f"{name}: the factory got progress {spy.draws()[0].progress}, expected 0.5 (step 30 of 60)")
            sigma = 1.0 - model.timesteps[-1]
            assert sigma.shape == (8,), f"{name}: expected 8 sigmas, got shape {tuple(sigma.shape)}"
            assert bool((sigma >= T_EPS - 1e-6).all() and (sigma <= 0.9999 + 1e-6).all()), (
                f"{name}: sigmas leave [{T_EPS}, 0.9999]: {sigma.tolist()}")


# ── 3. min_snr_gamma capability rule ─────────────────────────────────────────


def test_min_snr_gamma_is_hidden_for_every_family_but_sdxl():
    failures: list[str] = []
    checked = 0
    for defn in _definitions():
        checked += 1
        rule = resolve_capabilities(defn)["field_visibility"].get("min_snr_gamma")
        is_sdxl = defn.family == "sdxl"
        where = f"{defn.family}/{defn.id} min_snr_gamma"
        if rule is None:
            failures.append(f"{where}: no field_visibility rule at all (a missing rule counts as supported)")
        elif bool(rule.get("supported")) != is_sdxl:
            failures.append(f"{where}: supported={rule.get('supported')} but must be {is_sdxl}")
        elif not is_sdxl and "sdxl" not in str(rule.get("reason", "")).lower():
            failures.append(f"{where}: hidden without a reason naming SDXL (reason={rule.get('reason')!r})")
        config = {"min_snr_gamma": 7.0}
        dropped = apply_capability_allowlist(config, defn)
        if is_sdxl and (dropped or config.get("min_snr_gamma") != 7.0):
            failures.append(f"{where}: the allowlist dropped min_snr_gamma for an sdxl definition")
        if not is_sdxl and ("min_snr_gamma" in config or dropped != ["min_snr_gamma"]):
            failures.append(f"{where}: the allowlist kept min_snr_gamma (dropped={dropped}, config={config})")
    assert checked >= 32, f"only {checked} definitions in the registry -- the registry did not load"
    assert not failures, f"{len(failures)} min_snr_gamma visibility failure(s):\n" + "\n".join(failures)


# ── 4. the registry-wide contract: the real step, observed at the model and loss boundaries ──


def _rand(*shape: int, seed: int = 1) -> torch.Tensor:
    """A fixed tensor: the same clean latents / text rows in every run of a comparison."""
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _snap(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, (list, tuple)):
        return [_snap(v) for v in value]
    return None


def _same(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.shape == b.shape and torch.equal(a.float(), b.float())
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return a is None and b is None


class _Out(tuple):
    """A ``return_dict=False`` tuple that also answers ``.sample`` (the diffusers output attribute)."""

    @property
    def sample(self):
        return self[0]


def _zeros_like(hidden: Any, keep: Callable[[torch.Tensor], torch.Tensor] | None = None) -> Any:
    if isinstance(hidden, list):
        return [_zeros_like(h, keep) for h in hidden]
    pred = torch.zeros_like(hidden)
    return keep(pred) if keep is not None else pred


class _ModelDouble(nn.Module):
    """Stands in for a family's denoiser: records the timestep and the noisy input of every call and
    returns zeros shaped like the prediction its driver unpacks.

    ``config`` carries only the keys a profile names; a key the driver reads and the profile omits
    raises ``AttributeError`` and the family is reported UNVERIFIED, never silently passed.
    """

    def __init__(self, profile: "_Profile"):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(**profile.model_config)
        self.profile = profile
        self.calls: list[SimpleNamespace] = []

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def record(self, timestep: Any, hidden: Any) -> None:
        self.calls.append(SimpleNamespace(timestep=_snap(timestep), hidden=_snap(hidden)))

    def forward(self, *args, **kwargs):
        hidden = kwargs.get(self.profile.hidden_key, args[0] if args else None)
        self.record(kwargs.get(self.profile.timestep_key), hidden)
        if self.profile.respond is not None:
            return self.profile.respond(hidden, kwargs)
        pred = _zeros_like(hidden, self.profile.keep)
        return pred if self.profile.returns == "tensor" else _Out((pred,))


def _vae_double(channels: int) -> MagicMock:
    """A VAE whose batch-norm statistics are identity (flux2-style families normalise latents with them)."""
    vae = MagicMock()
    vae.bn.running_mean = torch.zeros(channels * 4)
    vae.bn.running_var = torch.ones(channels * 4)
    vae.config.batch_norm_eps = 1e-5
    vae.config.scaling_factor = 1.0
    vae.config.shift_factor = 0.0
    vae.config.latents_mean = [0.0] * channels  # qwen-style per-channel latent normalisation
    vae.config.latents_std = [1.0] * channels
    return vae


def _pair() -> tuple[torch.Tensor, torch.Tensor]:
    """The ``(embeddings [B, L, D], attention_mask [B, L])`` text form most drivers take."""
    return _rand(2, 4, 32, seed=2), torch.ones(2, 4, dtype=torch.long)


@dataclass
class _Profile:
    """What one family's real step needs around it: latent layout, text form, model-double shape."""

    latents: tuple[int, ...] = (2, 16, 8, 8)
    text: Callable[[], Any] = _pair
    model_config: dict[str, Any] = field(default_factory=dict)
    hidden_key: str = "hidden_states"
    timestep_key: str = "timestep"
    keep: Callable[[torch.Tensor], torch.Tensor] | None = None  # the prediction's slice of the input
    returns: str = "tuple"
    respond: Callable[[Any, dict], Any] | None = None
    setup: Callable[[Any, _ModelDouble, dict, ExitStack], None] | None = None
    components: Callable[[], dict[str, Any]] = dict  # extra loader components the driver stashes from
    config: dict[str, Any] = field(default_factory=dict)  # run-config keys the step needs to be drivable


def _setup_sdxl(trainer, model, batch, stack):
    # what encode_text / build_batch_extra leave for the forward (pooled CLIP + SDXL size ids)
    trainer.driver._pooled_embeds = _rand(2, 1280, seed=3)
    batch["time_ids"] = torch.tensor([[64.0, 64.0, 0.0, 0.0, 64.0, 64.0]] * 2)


def _setup_bernini(trainer, model, batch, stack):
    # the vendored packed forward IS the transformer call: route it into the double
    stack.enter_context(patch(
        "app.engine.models.families.bernini_r.driver.bernini_packed_forward",
        lambda m, *, target_latent, timestep, **_: m(hidden_states=target_latent, timestep=timestep)))


def _setup_minimax(trainer, model, batch, stack):
    def packed_forward(m, layout, video_rows, audio_rows, text_rows, distinct):
        m.record(distinct, video_rows)
        return torch.zeros_like(video_rows), torch.zeros_like(audio_rows)

    stack.enter_context(patch("app.engine.models.families.minimax_h3.packing.packed_forward", packed_forward))


def _setup_hidream(trainer, model, batch, stack):
    from app.engine.models.families.hidream_o1.trainer import PATCH_SIZE
    from tests.test_hidream_o1_integration import _make_mock_text_sample

    trainer.processor = MagicMock()
    trainer.tokenizer = MagicMock()
    side = PATCH_SIZE * 2
    n_patches = (side // PATCH_SIZE) ** 2
    batch["images"] = _rand(2, 3, side, side, seed=4)
    stack.enter_context(patch(
        "app.engine.models.families.hidream_o1.trainer.build_t2i_text_sample",
        return_value=_make_mock_text_sample(seq_len=n_patches, n_visual=n_patches)))


def _hidream_respond(hidden, kwargs):
    from app.engine.models.families.hidream_o1.trainer import PATCH_SIZE

    # hidream_o1 computes its loss inside forward_pass and refuses one without a grad_fn
    return SimpleNamespace(x_pred=torch.zeros(1, hidden.shape[1], 3 * PATCH_SIZE * PATCH_SIZE, requires_grad=True))


def _ace_condition_encoder():
    # the two buffers the driver stashes at assign time (the text2music context + the genre-drop null)
    return {"condition_encoder": SimpleNamespace(silence_latent=torch.zeros(1, 4, 64),
                                                 null_condition_emb=torch.zeros(1, 1, 32))}


def _k5_text():
    emb, _ = _pair()
    return TextEncoderOutput(embeddings=emb.reshape(1, 8, 32), attention_mask=torch.tensor([0, 4, 8], dtype=torch.int32),
                             pooled=_rand(2, 32, seed=5))


_PROFILES: dict[str, _Profile] = {
    "ace_step15": _Profile(latents=(2, 64, 16), components=_ace_condition_encoder),
    "bernini_r": _Profile(latents=(2, 16, 3, 8, 8), setup=_setup_bernini),
    "boogu_image": _Profile(model_config={"axes_dim_rope": (8, 4, 4), "axes_lens": (64, 64, 64)},
                            returns="tensor"),
    "dreamlite": _Profile(latents=(2, 4, 8, 8)),
    "ernie_image": _Profile(latents=(2, 32, 8, 8)),
    "flux1": _Profile(text=lambda: _pair()[0], model_config={"pooled_projection_dim": 8}),
    "flux2": _Profile(latents=(2, 32, 8, 8), text=lambda: _pair()[0]),
    "flux3": _Profile(latents=(2, 32, 8, 8), text=lambda: _pair()[0]),
    "hidream_o1": _Profile(latents=(2, 3, 64, 64), hidden_key="vinputs", respond=_hidream_respond,
                           setup=_setup_hidream),
    "hunyuan_video15": _Profile(latents=(2, 32, 3, 8, 8), text=lambda: (*_pair(), *_pair()),
                                keep=lambda p: p[:, :32]),
    "ideogram4": _Profile(latents=(2, 32, 8, 8)),
    "kandinsky5": _Profile(latents=(2, 16, 3, 8, 8), text=_k5_text, keep=lambda p: p[..., :16]),
    "krea2": _Profile(model_config={"patch_size": 2}),
    "ltx2": _Profile(latents=(2, 128, 3, 8, 8), text=lambda: _pair()[0]),
    "microsoft_lens": _Profile(latents=(2, 32, 8, 8), text=lambda: _rand(2, 3, 4, 32, seed=2), returns="tensor"),
    # a video-only step (no audio latents in the batch): the audio stream is off, as for a silent clip
    "minimax_h3": _Profile(latents=(2, 16, 3, 8, 8), setup=_setup_minimax, config={"train_audio": False}),
    "moss_image": _Profile(latents=(2, 32, 8, 8), text=lambda: _pair()[0]),
    "nucleus_image": _Profile(model_config={"patch_size": 2}),
    "omnigen2": _Profile(model_config={"axes_dim_rope": (8, 4, 4), "axes_lens": (64, 64, 64)},
                         returns="tensor"),
    "prx_pixel": _Profile(latents=(2, 3, 64, 64)),
    "qwen_image": _Profile(model_config={"patch_size": 2}),
    "sdxl": _Profile(latents=(2, 4, 8, 8), setup=_setup_sdxl),
    "wan21": _Profile(latents=(2, 16, 3, 8, 8), keep=lambda p: p[:, :16]),
    "wan22": _Profile(latents=(2, 16, 3, 8, 8), keep=lambda p: p[:, :16]),
    "wan22_ti2v_5b": _Profile(latents=(2, 48, 3, 8, 8), keep=lambda p: p[:, :48]),
    "zimage": _Profile(text=lambda: [_rand(4, 32, seed=2), _rand(4, 32, seed=3)], hidden_key="x", timestep_key="t"),
}

# Families whose loss target is, by design, independent of the noise: the noise_offset check at the
# LOSS boundary is off for them (the model-input check still runs). Each row carries its evidence.
_TARGET_WITHOUT_NOISE = {
    # prx_pixel/driver.py compute_target: "x0 objective: the target IS the clean image" -- ``noise``
    # deliberately does not appear in it (the pipeline converts x0_pred to a velocity per step)
    "prx_pixel": "x0 objective",
}


def _drive(defn, spy: _SamplerSpy, *, force: float | None = 0.5, **config) -> SimpleNamespace:
    """ONE real training step (``forward_loss_step``) of ``defn``'s trainer, seeded, into the model double.

    Returns the model double's calls (what the denoiser received) and the step's own tensors (its
    target and loss). A step that never reaches the double raises: the family is UNVERIFIED.
    """
    profile = _PROFILES.get(defn.family, _Profile())
    trainer = _trainer(defn, **{**profile.config, **config})
    # what setup() builds after init_scheduler (pipeline_loading.py, "3b. Noise interpolation")
    trainer.noise_interpolation = NoiseInterpolation(
        mode=trainer.config.get("noise_interpolation", "linear"), scheduler=trainer.scheduler)
    model = _ModelDouble(profile)
    components = {"unet": model, "unet_low": model, "transformer": model,
                  "vae": _vae_double(profile.latents[1]), **profile.components()}
    trainer.driver.assign_components(components)
    trainer.components = components
    trainer.autocast_dtype = torch.float32
    latents = _rand(*profile.latents)
    batch: dict[str, Any] = {"captions": ["a caption", "another caption"]}
    with ExitStack() as stack:
        if profile.setup is not None:
            profile.setup(trainer, model, batch, stack)
        trainer._attach_conditioning(batch, latents)
        spy.force = force
        try:
            random.seed(0)
            torch.manual_seed(0)
            step = trainer.forward_loss_step(latents, profile.text(), batch, 1)
        finally:
            spy.force = None
    if not model.calls:
        raise AssertionError("the step never called the model double (the denoiser boundary was not reached)")
    return SimpleNamespace(calls=model.calls, step=step)


# Pairs of forced factory values, one pair per side of a wan22-style expert boundary (the band
# sampler rejects draws outside the active expert's range, so only the in-band pair can move the
# model's timestep). A family that redraws its own timestep sees no difference under ANY pair.
_FORCED_PAIRS = ((0.2, 0.4), (0.92, 0.97))


def _observe_timestep(defn, spy) -> str | None:
    spy.calls.clear()
    seen = {t: _drive(defn, spy, force=t, timestep_sampling=TIMESTEP_SENTINEL, radc_start=0.8).calls[0].timestep
            for pair in _FORCED_PAIRS for t in pair}
    if not spy.draws(TIMESTEP_SENTINEL):
        return (f"the step never asked TimestepSampler for mode {TIMESTEP_SENTINEL!r} "
                f"(modes seen: {[c.mode for c in spy.draws()]})")
    if any(t is None for t in seen.values()):
        return "the model received no timestep tensor"
    if all(_same(seen[a], seen[b]) for a, b in _FORCED_PAIRS):
        return (f"the model received the same timestep whatever the factory drew ({_FORCED_PAIRS}) -- "
                "the step trains on a draw of its own, not the factory's")
    return None


def _observe_noise_offset(defn, spy) -> str | None:
    if defn.family == "minimax_h3":
        quiet = _trainer(defn, noise_offset=0.0)
        quiet._refuse_unproven_settings()  # the negative: no offset, no refusal
        try:  # the refusal may fire while the trainer is built or when it validates; either is explicit
            _trainer(defn, noise_offset=0.1)._refuse_unproven_settings()
        except ValueError as exc:
            if "noise_offset" in str(exc):
                return None
            return f"the validation error does not name noise_offset: {exc}"
        return "noise_offset=0.1 raised no validation error"
    # Same seed, same forced timesteps; 0.1 and 0.2 consume the random stream identically, so only
    # the offset term's scale differs between them.
    runs = {s: _drive(defn, spy, noise_offset=s) for s in (0.0, 0.1, 0.2)}
    hidden = {s: r.calls[0].hidden for s, r in runs.items()}
    if hidden[0.1] is None:
        return "the model received no noisy input tensor"
    if _same(hidden[0.0], hidden[0.1]):
        return "the model's noisy input is the same with noise_offset 0.1 as with 0.0"
    if _same(hidden[0.1], hidden[0.2]):
        return ("the model's noisy input is the same with noise_offset 0.2 as with 0.1 (same draws, "
                "only the offset scale differs) -- the configured offset never reaches the model")
    targets = {s: _snap(r.step.target) for s, r in runs.items()}
    if defn.family in _TARGET_WITHOUT_NOISE:
        return None
    if targets[0.1] is not None and _same(targets[0.1], targets[0.2]):
        return "the loss target is the same with noise_offset 0.2 as with 0.1 -- the offset noise never reaches the loss"
    return None


def _observe_min_snr(defn, spy) -> str | None:
    # gammas far apart: min(snr, gamma) / snr differs at every DDPM timestep the step can draw
    losses = {g: _drive(defn, spy, force=None, min_snr_gamma=g).step.loss for g in (1e-3, 1e3)}
    if torch.equal(losses[1e-3].detach(), losses[1e3].detach()):
        return "the step's loss is identical for min_snr_gamma 0.001 and 1000 -- the gamma never weights the loss"
    return None


def _configure_optimization(defn, **config):
    trainer = _trainer(defn, **config)
    model = nn.Linear(2, 2)
    trainer._get_primary_model = lambda: model
    trainer._name_optimizer_params = lambda params: [f"p{i}" for i in range(len(params))]
    trainer._get_text_encoders = lambda: {}
    try:
        trainer.driver.assign_components({"unet": model})
    except Exception:  # noqa: BLE001 -- drivers that need no model for the optimizer step ignore it
        pass
    # bernini_r (14B) and wan22 wrap the shared method around their expert pair and call super():
    # the family's OWN entry point is driven (an override that drops a field must fail), with only
    # the expert plumbing stubbed -- the expert params collapse onto the one stub model and the
    # start placement is a no-op (no weights to place).
    trainer._collect_expert_params = lambda: list(model.parameters())
    if hasattr(trainer.driver, "place_experts_for_start"):
        trainer.driver.place_experts_for_start = lambda *a, **k: None
    trainer._configure_optimization(10)


def _observe_factory(defn, spy, field) -> str | None:
    seen: dict[str, object] = {}

    def optimizer_create(optimizer_type, params, lr, weight_decay, **kwargs):
        seen["optimizer_type"] = optimizer_type
        return torch.optim.SGD(params, lr=lr)

    def scheduler_create(sched_type, optimizer, warmup, total):
        seen["lr_scheduler"] = sched_type
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    sentinel = {"optimizer_type": OPTIMIZER_SENTINEL, "lr_scheduler": SCHEDULER_SENTINEL}[field]
    with patch.object(OptimizerFactory, "create", staticmethod(optimizer_create)), \
            patch.object(LRSchedulerFactory, "create", staticmethod(scheduler_create)):
        _configure_optimization(defn, **{field: sentinel})
    if seen.get(field) == sentinel:
        return None
    return f"{'OptimizerFactory' if field == 'optimizer_type' else 'LRSchedulerFactory'}.create saw {seen.get(field)!r}, not {sentinel!r}"


_OBSERVERS: dict[str, Callable[[Any, _SamplerSpy], str | None]] = {
    "timestep_sampling": _observe_timestep,
    "noise_offset": _observe_noise_offset,
    "min_snr_gamma": _observe_min_snr,
    "optimizer_type": lambda d, s: _observe_factory(d, s, "optimizer_type"),
    "lr_scheduler": lambda d, s: _observe_factory(d, s, "lr_scheduler"),
}


def _contract(spy, definitions, fields=ENGINE_FIELDS) -> SimpleNamespace:
    """Every shown field of every definition through its observer: failures, unverified, hidden."""
    failures: list[str] = []
    hidden: list[str] = []
    unverified: list[str] = []
    observed = 0
    for defn in definitions:
        visibility = resolve_capabilities(defn)["field_visibility"]
        for name in fields:
            rule = visibility.get(name)
            if rule is not None and not rule.get("supported", True):
                hidden.append(f"{defn.family}/{defn.id} {name}: hidden ({rule.get('reason')})")
                continue
            try:
                problem = _OBSERVERS[name](defn, spy)
            except Exception as exc:  # noqa: BLE001 -- a family that cannot be driven is reported by name
                unverified.append(f"{defn.family}/{defn.id} {name}: UNVERIFIED, driving raised {type(exc).__name__}: {exc}")
                continue
            if problem is None:
                observed += 1
            else:
                failures.append(f"{defn.family}/{defn.id} {name}: shown but never reached its sink -- {problem}")
    return SimpleNamespace(failures=failures, unverified=unverified, hidden=hidden, observed=observed)


def test_every_shown_engine_field_is_observed(spy):
    definitions = _definitions()
    result = _contract(spy, definitions)
    report = [f"{len(result.failures)} shown-but-unobserved, {len(result.unverified)} unverified, "
              f"{len(result.hidden)} hidden by a declared capability, {result.observed} observed over "
              f"{len(definitions)} definitions"]
    report += result.failures + result.unverified + [f"HIDDEN {h}" for h in result.hidden]
    print("\n".join(report))
    assert len(definitions) >= 32, f"only {len(definitions)} definitions in the registry -- it did not load"
    assert not result.failures and not result.unverified, "\n".join(report)


# ── 5. positive controls: break one link of the path, the contract must name the family ──


def _one(definition_id: str):
    defn = registry.get_definition(definition_id)
    assert defn is not None, f"definition {definition_id} is not in the registry"
    return defn


def _trainer_class(defn):
    return type(_trainer(defn))


def test_positive_control_a_redrawn_timestep_is_reported(spy, monkeypatch):
    defn = _one("chroma1-base")
    assert _contract(spy, [defn], ("timestep_sampling",)).observed == 1, "the control's family must pass unbroken"
    cls = _trainer_class(defn)
    original = cls.forward_pass

    def redraw(self, noisy_input, timesteps, text_embeddings, batch):
        return original(self, noisy_input, torch.rand_like(timesteps.float()) * 1000, text_embeddings, batch)

    monkeypatch.setattr(cls, "forward_pass", redraw)
    result = _contract(spy, [defn], ("timestep_sampling",))
    assert any("chroma1-base timestep_sampling" in f and "draw of its own" in f for f in result.failures), (
        f"a forward_pass that discards the sampled timesteps and redraws was not reported: {result}")


def test_positive_control_a_dropped_offset_term_is_reported(spy, monkeypatch):
    defn = _one("zimage-base")
    assert _contract(spy, [defn], ("noise_offset",)).observed == 1, "the control's family must pass unbroken"

    def no_offset_term(self, latents):
        noise = torch.randn_like(latents)
        strength = float(self.config.get("noise_offset", 0.0))
        if strength > 0:  # the offset is still drawn (same random stream) but never added
            torch.randn(latents.shape[0], latents.shape[1], *([1] * (latents.ndim - 2)), dtype=latents.dtype)
        return noise

    monkeypatch.setattr(PipelineTrainMixin, "make_training_noise", no_offset_term)
    result = _contract(spy, [defn], ("noise_offset",))
    assert any("zimage-base noise_offset" in f and "same with noise_offset 0.1 as with 0.0" in f
               for f in result.failures), f"a dropped offset term was not reported: {result}"


def test_positive_control_a_family_that_redraws_its_noise_is_reported(spy, monkeypatch):
    defn = _one("qwen-image-2512")
    assert _contract(spy, [defn], ("noise_offset",)).observed == 1, "the control's family must pass unbroken"
    cls = _trainer_class(defn)
    monkeypatch.setattr(cls, "prepare_noise_for_training",
                        lambda self, noise: self.driver.prepare_noise(torch.randn_like(noise)))
    result = _contract(spy, [defn], ("noise_offset",))
    assert any("qwen-image-2512 noise_offset" in f and "never reaches the model" in f for f in result.failures), (
        f"a family hook that discards the loop's noise and redraws was not reported: {result}")


def test_positive_control_a_loop_disconnected_from_the_config_is_reported(spy, monkeypatch):
    defn = _one("flux2-dev")
    fields = ("timestep_sampling", "noise_offset")
    assert _contract(spy, [defn], fields).observed == 2, "the control's family must pass unbroken"
    real_noise = PipelineTrainMixin.make_training_noise
    real_sample = _trainer_class(defn).sample_timesteps

    def unread_noise(self, latents):  # reads a key the form never writes: the configured value is lost
        saved = self.config
        self.config = {k: v for k, v in saved.items() if k != "noise_offset"}
        try:
            return real_noise(self, latents)
        finally:
            self.config = saved

    def unread_mode(self, batch_size, latents=None):  # the step draws with the default mode, not the configured one
        saved = self.config
        self.config = {k: v for k, v in saved.items() if k != "timestep_sampling"}
        try:
            return real_sample(self, batch_size, latents)
        finally:
            self.config = saved

    monkeypatch.setattr(PipelineTrainMixin, "make_training_noise", unread_noise)
    monkeypatch.setattr(_trainer_class(defn), "sample_timesteps", unread_mode)
    result = _contract(spy, [defn], fields)
    reported = {f.split(":")[0] for f in result.failures}
    assert reported == {"flux2/flux2-dev timestep_sampling", "flux2/flux2-dev noise_offset"}, (
        f"a step disconnected from the configured mode and offset was not reported on both fields: {result}")


def test_an_undrivable_family_is_unverified_never_skipped(spy, monkeypatch):
    defn = _one("chroma1-base")
    monkeypatch.setitem(_PROFILES, "chroma", _Profile(latents=(2, 16, 8)))  # a layout its driver refuses
    result = _contract(spy, [defn], ("noise_offset",))
    assert result.observed == 0 and not result.failures and len(result.unverified) == 1, result
    assert "UNVERIFIED" in result.unverified[0], result.unverified
