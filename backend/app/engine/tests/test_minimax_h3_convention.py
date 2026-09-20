"""minimax_h3 convention-contract suite (concept §5.2; plan rows 1.1–1.5, 1.7).

The failure this suite guards is silent at training time and visible only at
inference: a sign, scale or schedule error produces a LoRA of pure noise, and
this project has shipped that defect once (memory `flowmatch-timestep-scale-
gotcha`). Each test names the wrong answer it rejects.

Row 1.1 — tests 6 and 7 (the dual-shift schedule):

* 6 ``test_dual_shift_comes_from_the_definition_not_a_literal`` — the drawn
  σ_v CHANGES when the definition's ``video.sigma_shift`` changes (the exact
  PR0 Task-7 defect: a literal nothing reads), matches the closed form at
  ``12.0`` to 1e-9, and a definition without the key RAISES.
* 7 ``test_one_u_drives_both_curves`` — for ONE ``u``, ``σ_v == shift(u, 12)``
  AND ``σ_a == shift(u, 3)`` to 1e-9, and ``σ_a`` is a deterministic function
  of ``σ_v`` (the reference identity ``remap_sigma(σ_v, 12→3) == σ_a``,
  research §3.2). Rejects two independent draws, which a naive ``σ_a != σ_v``
  assertion would pass.

Closed form: ``s·u / (1 + (s−1)·u)`` — diffusers ``scheduling_minimax_h3.py:157``
(0.40.0); ai-toolkit ``src/packing.py`` ``shift_sigma`` / ``remap_sigma``
(method re-derived, no line copied).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.minimax_h3.schedule import H3SigmaSchedule
from app.engine.models.families.minimax_h3.settings import resolve_h3_settings

_TESTS_DIR = Path(__file__).resolve().parent

_ARCH: dict[str, Any] = {
    "audio.loss_weight": 0.1,
    "video.sigma_shift": 12.0,
    "audio.sigma_shift": 3.0,
}


def _definition(arch: dict[str, Any]) -> ModelDefinition:
    return ModelDefinition(
        id="minimax-h3-t2va",
        family="minimax_h3",
        name="fixture",
        defaults={"train_audio": True},
        architecture_params=arch,
    )


def _schedule(arch: dict[str, Any]) -> H3SigmaSchedule:
    return H3SigmaSchedule.from_settings(
        resolve_h3_settings(_definition(arch), {"train_audio": True})
    )


def _closed_form(u: torch.Tensor, s: float) -> torch.Tensor:
    # Written independently of schedule.py so the test is an oracle, not an echo.
    return (s * u) / (1.0 + (s - 1.0) * u)


_U = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)


# ── 6. The shifts come from the definition, never a literal ─────────────


def test_dual_shift_comes_from_the_definition_not_a_literal():
    at_12 = _schedule(dict(_ARCH))
    at_7 = _schedule(dict(_ARCH, **{"video.sigma_shift": 7.0}))

    sigma_v_12, sigma_a_12 = at_12.draw(_U)
    sigma_v_7, _ = at_7.draw(_U)

    assert not torch.allclose(sigma_v_12[1:-1], sigma_v_7[1:-1]), (
        "σ_v unchanged under video.sigma_shift: 7.0 — the schedule reads a "
        "literal, not the definition"
    )
    assert torch.allclose(sigma_v_12, _closed_form(_U, 12.0), atol=1e-9, rtol=0)
    assert torch.allclose(sigma_a_12, _closed_form(_U, 3.0), atol=1e-9, rtol=0)
    assert torch.allclose(sigma_v_7, _closed_form(_U, 7.0), atol=1e-9, rtol=0)

    arch_without = dict(_ARCH)
    del arch_without["video.sigma_shift"]
    with pytest.raises(ValueError, match="sigma_shift_video"):
        _schedule(arch_without)


# ── 7. ONE u drives BOTH curves ──────────────────────────────────────────


def test_one_u_drives_both_curves():
    schedule = _schedule(dict(_ARCH))
    sigma_v, sigma_a = schedule.draw(_U)

    assert torch.allclose(sigma_v, _closed_form(_U, 12.0), atol=1e-9, rtol=0)
    assert torch.allclose(sigma_a, _closed_form(_U, 3.0), atol=1e-9, rtol=0), (
        "σ_a not a function of σ_v — audio drew its own u"
    )

    # Reference identity (research §3.2): un-shift σ_v by 12, re-shift by 3.
    # Inverse of s·u/(1+(s−1)·u):  u = σ / (s − (s−1)·σ).
    u_recovered = sigma_v / (12.0 - 11.0 * sigma_v)
    remapped = _closed_form(u_recovered, 3.0)
    assert torch.allclose(remapped, sigma_a, atol=1e-9, rtol=0), (
        "σ_a not a function of σ_v — remap_sigma(σ_v, 12→3) != σ_a"
    )

    # Endpoints are fixed points of the shift: u=0 → 0, u=1 → 1 on both curves.
    assert sigma_v[0] == 0.0 and sigma_a[0] == 0.0
    assert sigma_v[-1] == 1.0 and sigma_a[-1] == 1.0


# ── Row 1.2: the training convention on the driver ───────────────────────
#
# Oracle (concept §5.1, closed, no free parameters), against the INSTALLED
# diffusers 0.40.0 `MiniMaxH3Scheduler` (`scheduling_minimax_h3.py:170-171`
# t = 1 − σ; `:225` scale_noise x_t = t·x₀ + (1−t)·noise; `:273` step
# x̂₀ = x_t + σ·v):  x₀ = x_t + σ·v  ⇒  v = x₀ − noise, unique, no scale.
# Research §3.1: ai-toolkit `t_v = 1.0 − sigma_v`, `return -noise_pred`;
# diffusion-pipe `t_v = 1.0 − sigma_v`, `-video_out` — the same contract.

_CONTROLS = _TESTS_DIR / "fixtures" / "h3_convention_controls.py"


def _driver(arch: dict[str, Any] | None = None):
    from app.engine.models.families.minimax_h3.driver import MiniMaxH3Driver

    return MiniMaxH3Driver(_definition(dict(_ARCH) if arch is None else arch), torch.device("cpu"))


def _scheduler(shift: float = 12.0):
    from diffusers import MiniMaxH3Scheduler

    return MiniMaxH3Scheduler(shift=shift)


def _family_sources() -> list[Path]:
    family_dir = _TESTS_DIR.parents[0] / "models" / "families" / "minimax_h3"
    return sorted(
        p for p in family_dir.rglob("*.py") if "vendor" not in p.relative_to(family_dir).parts
    )


# ── 1. The target is the unique velocity the scheduler inverts ───────────


def test_target_is_the_unique_velocity_the_scheduler_inverts():
    drv = _driver()
    sched = _scheduler()
    torch.manual_seed(1)
    x0 = torch.randn(2, 24, 3, 4, 4)
    noise = torch.randn(2, 24, 3, 4, 4)
    for sigma in (0.999, 0.9, 0.5, 0.1, 0.01):
        t = torch.full((2,), 1.0 - sigma)
        x_t = drv.add_noise(x0, noise, t)
        # The driver's forward process IS the reference's forward process.
        assert torch.allclose(x_t, sched.scale_noise(x0, t, noise), atol=1e-6), (
            f"add_noise != MiniMaxH3Scheduler.scale_noise at sigma={sigma}"
        )
        v = drv.compute_target(x0, noise, t)
        assert torch.allclose(x_t + sigma * v, x0, atol=1e-5), (
            f"x_t + sigma*v != x0 at sigma={sigma}: the target is not x0 - noise"
        )


# ── 2. A perfect-velocity walk recovers x₀ ───────────────────────────────


def test_perfect_velocity_round_trip_recovers_x0():
    drv = _driver()
    sched = _scheduler(shift=12.0)
    sched.set_timesteps(20, device="cpu")
    torch.manual_seed(2)
    x0 = torch.randn(1, 24, 2, 4, 4)
    noise = torch.randn(1, 24, 2, 4, 4)
    v = drv.compute_target(x0, noise, torch.tensor([0.0]))
    sample = noise.clone()
    for t in sched.timesteps:
        sample = sched.step(v, t, sample, return_dict=False)[0]
    assert torch.allclose(sample, x0, atol=1e-4), (
        "the oracle velocity walked through the reference scheduler does not "
        "land on x0 — a direction/sign error, not a precision one"
    )


# ── 3. add_noise runs on H3's clock: t=1 clean, t=0 noise ────────────────


def test_add_noise_endpoints_are_h3_clockwise():
    drv = _driver()
    torch.manual_seed(3)
    x0 = torch.randn(2, 24, 2, 4, 4)
    noise = torch.randn(2, 24, 2, 4, 4)
    assert torch.equal(drv.add_noise(x0, noise, torch.tensor([1.0, 1.0])), x0), (
        "t=1 is not x0 — the standard clock (t=0 clean) is the wrong one here"
    )
    assert torch.equal(drv.add_noise(x0, noise, torch.tensor([0.0, 0.0])), noise), (
        "t=0 is not pure noise"
    )
    # Strictly monotone: the distance to x0 falls as t rises.
    dist = [
        (drv.add_noise(x0, noise, torch.full((2,), t)) - x0).norm().item()
        for t in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]
    assert all(b < a for a, b in zip(dist, dist[1:])), f"not monotone: {dist}"


# ── sample_timesteps: the drawn u, through the VIDEO shift, as t ─────────


def test_sample_timesteps_are_video_clock_t_of_the_drawn_u():
    drv = _driver()
    config = {"train_audio": True, "timestep_sampling": "uniform"}
    torch.manual_seed(4)
    t = drv.sample_timesteps(8, torch.device("cpu"), config)
    torch.manual_seed(4)
    u = torch.rand((8,))
    expected = 1.0 - _closed_form(u, 12.0)
    assert t.shape == (8,) and t.dtype.is_floating_point
    assert torch.all((t >= 0) & (t <= 1))
    assert torch.allclose(t, expected.to(t.dtype), atol=1e-6), (
        "sample_timesteps is not 1 - shift(u, video.sigma_shift)"
    )
    # The shift is the definition's, not a literal.
    torch.manual_seed(4)
    t7 = _driver(dict(_ARCH, **{"video.sigma_shift": 7.0})).sample_timesteps(
        8, torch.device("cpu"), config
    )
    assert not torch.allclose(t, t7)


# ── 5. No ×1000 anywhere in the family ───────────────────────────────────

TRIPLE = '"' * 3

_THOUSAND_RE = re.compile(r"[*/]\s*1000(?:\.0)?\b|num_train_timesteps")


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """``(lineno, code)`` with ``#`` comments and triple-quoted docstrings
    stripped — the guards scan what EXECUTES; a docstring that names the
    contract (``t = 1 - sigma``) is documentation, not a conversion site."""
    out: list[tuple[int, str]] = []
    in_doc = False
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        code = raw.split("#", 1)[0]
        quotes = code.count(TRIPLE)
        if quotes % 2 == 1:
            in_doc = not in_doc
            continue
        if in_doc or quotes:
            continue
        out.append((n, code))
    return out


def _thousand_offenders(paths: list[Path]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for p in paths:
        hits = [n for n, code in _code_lines(p) if _THOUSAND_RE.search(code)]
        if hits:
            out[p.name] = hits
    return out


def test_no_thousand_scaling_in_family_source():
    assert _thousand_offenders([_CONTROLS]), f"control not flagged: {_CONTROLS}"
    offenders = _thousand_offenders(_family_sources())
    assert not offenders, f"timestep rescale in family source: {offenders}"


# ── 8. ONE σ↔t conversion module ─────────────────────────────────────────

_CONVERSION_RE = re.compile(r"\b1(?:\.0)?\s*-\s*(?:sigma|timesteps?|t)\b")


def _conversion_sites(paths: list[Path]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for p in paths:
        hits = [n for n, code in _code_lines(p) if _CONVERSION_RE.search(code)]
        if hits:
            out[p.name] = hits
    return out


def test_trainer_and_sampler_share_one_conversion_module():
    assert _conversion_sites([_CONTROLS]), f"control not flagged: {_CONTROLS}"
    sites = _conversion_sites(_family_sources())
    assert set(sites) == {"schedule.py"}, (
        "the sigma<->t conversion must live in schedule.py and nowhere else in "
        f"the family (training and sampling would diverge): {sites}"
    )


# ── Row 1.3: the packed layout (`packing.py`) ────────────────────────────
#
# Row order [text | keyframe conds | audio | video]; patch (1,2,2)
# frame-major then row-major, features [c,pt,ph,pw]; rotary grids float64
# from numpy `linspace(endpoint=False)` on the shared 40-units/s clock;
# `media_origin = num_text + media_advance`. Reference: ai-toolkit@561a0236
# `src/packing.py` (MIT, read-and-reimplement; no line copied), research
# §3.3. The transformer checks STRUCTURE only (`transformer_minimax_h3.py`
# `:585-591`) and embeds whatever `timestep` arrives (`:613`), so the
# cardinality guard sits at the DRIVER boundary.


def _geometry(**overrides):
    from app.engine.models.families.minimax_h3.packing import H3Geometry

    base = dict(
        num_text=8,
        latent_frames=2,
        latent_height=6,
        latent_width=4,
        audio_latents=3,
    )
    base.update(overrides)
    return H3Geometry(**base)


class _RecordingTransformer:
    """Stands in for the DiT: records the forward kwargs and returns its
    video / audio rows unchanged (an identity transformer)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any):
        self.calls.append(kwargs)
        return kwargs["hidden_states"], kwargs["audio_hidden_states"]


def _ramp_inputs(layout, geometry):
    from app.engine.models.families.minimax_h3.packing import (
        pack_audio,
        patchify_video,
    )

    c = 24
    t, h, w = geometry.latent_frames, geometry.latent_height, geometry.latent_width
    ramp = torch.arange(c * t * h * w, dtype=torch.float32).reshape(1, c, t, h, w)
    video_rows = patchify_video(ramp, layout.patch_size)
    audio = torch.arange(2 * 32 * geometry.audio_latents, dtype=torch.float32).reshape(
        1, 2, 32, geometry.audio_latents
    )
    audio_rows = pack_audio(audio)
    text = torch.randn(1, geometry.num_text, 16)
    return ramp, video_rows, audio, audio_rows, text


# ── 4. Timesteps at the transformer boundary are the distinct, unscaled set


def test_timesteps_at_the_transformer_boundary_are_unscaled():
    from app.engine.models.families.minimax_h3.packing import build_layout, packed_forward

    geometry = _geometry()
    layout = build_layout(geometry)
    _, video_rows, _, audio_rows, text = _ramp_inputs(layout, geometry)
    stub = _RecordingTransformer()
    packed_forward(stub, layout, video_rows, audio_rows, text, torch.tensor([0.5, 0.3]))

    (call,) = stub.calls
    ts = call["timestep"]
    assert ts.numel() <= 3, f"numel > 3: seq_len timesteps ({ts.numel()}) reached the transformer"
    assert ts.dtype.is_floating_point
    assert torch.all((ts >= 0) & (ts <= 1)), f"timestep outside [0, 1]: {ts}"
    assert abs(float(ts[0]) - 0.5) < 1e-6, f"video timestep scaled: {float(ts[0])} (not 500 either way)"
    assert call["timestep_indices"].shape == (layout.seq_len,)
    assert int(call["timestep_indices"].max()) < ts.numel(), "a row points past the timestep set"
    # Every target video row is at slot 0 (t_v), every target audio row at slot 1 (t_a).
    assert torch.all(call["timestep_indices"][layout.video_indices] == 0)
    assert torch.all(call["timestep_indices"][layout.audio_indices] == 1)


def test_driver_rejects_seq_len_timesteps():
    drv = _driver()
    layout_len = 8 + 3 * 2 + 2 * (6 // 2) * (4 // 2)
    with pytest.raises(ValueError, match="timestep"):
        drv.assert_timestep_cardinality(torch.rand(layout_len))
    with pytest.raises(ValueError, match="timestep"):
        drv.assert_timestep_cardinality(torch.tensor([0.1, 0.2, 0.3, 0.4]))  # 4 on t2v
    drv.assert_timestep_cardinality(torch.tensor([0.5, 0.3]))  # (t_v, t_a)
    drv.assert_timestep_cardinality(torch.tensor([0.5, 0.3, 0.999]))  # + t_c
    with pytest.raises(ValueError, match="timestep"):
        drv.assert_timestep_cardinality(torch.tensor([0.5, 1.5]))  # outside [0, 1]
    with pytest.raises(ValueError, match="timestep"):
        drv.assert_timestep_cardinality(torch.tensor([1, 0]))  # not floating
    # The reference mode packs a reference soundtrack at a fourth slot.
    ref = _driver(dict(_ARCH, mode="reference"))
    ref.assert_timestep_cardinality(torch.tensor([0.5, 0.3, 0.999, 1.0]))


# ── E4. The ramp round-trips through pack → forward → unpatchify ─────────


def test_ramp_round_trips_through_the_packed_forward():
    from app.engine.models.families.minimax_h3.packing import (
        build_layout,
        packed_forward,
        unpack_audio,
        unpatchify_video,
    )

    geometry = _geometry()
    layout = build_layout(geometry)
    ramp, video_rows, audio, audio_rows, text = _ramp_inputs(layout, geometry)
    video_out, audio_out = packed_forward(
        _RecordingTransformer(), layout, video_rows, audio_rows, text, torch.tensor([0.5, 0.3])
    )
    assert torch.equal(unpatchify_video(video_out, layout), ramp)
    assert torch.equal(unpack_audio(audio_out, geometry.audio_latents), audio)

    # Row k of the video rows is the k-th patch frame-major then row-major, and
    # its rotary coordinate says so: t ascends slowest, then h, then w.
    pos = layout.position_ids[layout.video_indices]
    t, h, w = pos.unbind(-1)
    rows_per_frame = (geometry.latent_height // 2) * (geometry.latent_width // 2)
    assert torch.all(t[:rows_per_frame] == t[0]) and torch.all(t[rows_per_frame:] > t[0])
    key = (t * 1e6 + h * 1e3 + w).tolist()
    assert key == sorted(key), "video rows are not in frame-major, row-major order"


# ── The rotary grid is float64 numpy `linspace(endpoint=False)` ──────────


def test_rotary_grid_is_float64_numpy():
    from app.engine.models.families.minimax_h3.packing import build_layout

    # 4 x 6 latents: on the WIDTH axis (dim 6, 3 patches) a torch.linspace
    # build differs from numpy's endpoint=False build in the last ulp
    # (probed over every even grid 4..64: 1538 such axes exist; this is one).
    geometry = _geometry(latent_height=4, latent_width=6)
    layout = build_layout(geometry)
    assert layout.position_ids.dtype == torch.float64

    # Independent numpy reference (research §3.3): per axis
    #   ratio = dim / sqrt(H*W); left = (1 - ratio) / 2;
    #   grid = linspace(left, left + ratio, dim // patch, endpoint=False) * 32
    sqrt_area = np.sqrt(4 * 6)

    def ref(dim: int, patch: int) -> np.ndarray:
        ratio = dim / sqrt_area
        left = (1.0 - ratio) / 2.0
        return np.linspace(left, left + ratio, dim // patch, endpoint=False) * 32

    h_ref, w_ref = ref(4, 2), ref(6, 2)
    first_frame = layout.position_ids[layout.video_indices[: (4 // 2) * (6 // 2)]]
    h_got = first_frame[:, 1].reshape(2, 3)[:, 0].numpy()
    w_got = first_frame[:, 2].reshape(2, 3)[0, :].numpy()
    assert np.array_equal(h_got, h_ref), f"h grid not bit-equal to numpy: {h_got - h_ref}"
    assert np.array_equal(w_got, w_ref), f"w grid not bit-equal to numpy: {w_got - w_ref}"

    # The torch.linspace build IS off in the last ulp here — the difference
    # the released checkpoint's audio/video alignment depends on.
    ratio = 6 / sqrt_area
    left = (1.0 - ratio) / 2.0
    torch_build = torch.linspace(left, left + ratio, 4, dtype=torch.float64)[:3] * 32
    assert not np.array_equal(torch_build.numpy(), w_ref), "geometry no longer discriminates"


# ── Prompt length shifts the whole media clock ───────────────────────────


def test_prompt_length_shifts_media_origin():
    from app.engine.models.families.minimax_h3.packing import build_layout

    short = build_layout(_geometry(num_text=8))
    long = build_layout(_geometry(num_text=12))
    assert long.media_origin - short.media_origin == 4.0
    assert short.media_origin == 8.0, "media_origin must be num_text + media_advance"
    # Text rows sit at their own index on the time axis; media starts after.
    assert torch.equal(short.position_ids[short.text_indices, 0], torch.arange(8, dtype=torch.float64))
    t_video_short = short.position_ids[short.video_indices, 0]
    t_video_long = long.position_ids[long.video_indices, 0]
    assert torch.equal(t_video_long - t_video_short, torch.full_like(t_video_short, 4.0))
    t_audio_short = short.position_ids[short.audio_indices, 0]
    t_audio_long = long.position_ids[long.audio_indices, 0]
    assert torch.equal(t_audio_long - t_audio_short, torch.full_like(t_audio_short, 4.0))
    assert float(t_video_short[0]) == 8.0 and float(t_audio_short[0]) == 8.0


# ── 9. Gradients reach only the targeted modules (row 1.4, E2) ───────────


def _t2va_targets() -> list[str]:
    import yaml

    path = (
        _TESTS_DIR.parents[0] / "models" / "families" / "minimax_h3" / "definitions" / "minimax_h3_t2va.yaml"
    )
    with open(path, encoding="utf-8") as fh:
        return list(yaml.safe_load(fh)["lora_targetable_modules"])


def _random_inputs(layout, geometry):
    from app.engine.models.families.minimax_h3.packing import pack_audio, patchify_video

    g = torch.Generator().manual_seed(0)
    t, h, w = geometry.latent_frames, geometry.latent_height, geometry.latent_width
    video = torch.randn(1, 24, t, h, w, generator=g)
    audio = torch.randn(1, 2, 32, geometry.audio_latents, generator=g)
    text = torch.randn(1, geometry.num_text, 16, generator=g)
    return patchify_video(video, layout.patch_size), pack_audio(audio), text


def test_gradients_reach_only_the_targeted_modules(build_tiny_transformer):
    """Tiny-arch forward + backward with a REAL PEFT wrap on the curated
    ``t2va.yaml`` targets, driven through ``packed_forward``:

    * every module the YAML names got an adapter and its ``lora_B`` receives a
      finite, non-zero gradient — the adapter is on the path the loss sees;
    * no base weight receives a gradient (``grad is None``);
    * NO adapter sits on any ``adaln_proj.linear`` — the AdaLN table differs
      between the full and the pruned checkpoint, so a LoRA touching it loads
      into only one of them (research §3.6; diffusion-pipe's stated reason);
    * no ``CacheMixin`` config is enabled on the transformer (A-4): a cache
      skips blocks at inference and would silently skip them in training too.
    """
    from peft import LoraConfig, get_peft_model

    from app.engine.models.families.minimax_h3.packing import build_layout, packed_forward

    from app.engine.models.families.minimax_h3.driver import MiniMaxH3Driver

    targets = _t2va_targets()
    definition = _definition(dict(_ARCH)).model_copy(update={"lora_targetable_modules": targets})
    driver = MiniMaxH3Driver(definition, torch.device("cpu"))
    assert driver.get_lora_targets() == targets, "the driver must hand PEFT the YAML list verbatim"

    base = build_tiny_transformer()
    base.train()
    assert base.is_cache_enabled is False, "a CacheMixin config is enabled on the transformer (A-4)"
    matching_linears = {
        name
        for name, mod in base.named_modules()
        if isinstance(mod, torch.nn.Linear) and any(name.endswith("." + t) for t in targets)
    }
    assert matching_linears, "no Linear matches the curated targets"

    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=2, target_modules=targets))
    for name, mod in model.named_modules():
        assert not ("adaln_proj" in name and hasattr(mod, "lora_A")), f"adapter on adaln_proj: {name}"

    geometry = _geometry()
    layout = build_layout(geometry)
    video_rows, audio_rows, text = _random_inputs(layout, geometry)
    video_out, audio_out = packed_forward(
        model, layout, video_rows, audio_rows, text, torch.tensor([0.5, 0.3])
    )
    loss = video_out.float().pow(2).mean() + audio_out.float().pow(2).mean()
    assert torch.isfinite(loss)
    loss.backward()

    adapted = set()
    for name, param in model.named_parameters():
        if ".lora_B." in name:
            assert param.grad is not None, f"lora_B grad is None: {name}"
            assert torch.isfinite(param.grad).all(), f"lora_B grad not finite: {name}"
            assert param.grad.abs().sum() > 0, f"lora_B grad is all-zero: {name}"
            adapted.add(name.split(".lora_B.")[0].removeprefix("base_model.model."))
        elif ".lora_A." not in name:
            assert param.grad is None, f"base weight received a gradient: {name}"
    assert adapted == matching_linears, (
        "adapter coverage drifted from the curated targets.\n"
        f"  targeted but no adapter: {sorted(matching_linears - adapted)}\n"
        f"  adapter but not targeted: {sorted(adapted - matching_linears)}"
    )
    assert base.is_cache_enabled is False


# ── 10. Gradient checkpointing arms BOTH sites (row 1.5, E3) ─────────────


def _lora_wrapped(build_tiny_transformer):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        build_tiny_transformer(), LoraConfig(r=2, lora_alpha=2, target_modules=_t2va_targets())
    )
    model.train()
    return model


def _lora_grads(model) -> dict[str, torch.Tensor]:
    return {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if ".lora_" in n and p.grad is not None
    }


def test_checkpointing_arms_both_sites(build_tiny_transformer):
    """ONE ``enable_gradient_checkpointing()`` must reach the token refiner
    (``transformer_minimax_h3.py:312-313``) AND the block stack (``:648-649``):
    diffusers walks ``named_modules()`` for every ``gradient_checkpointing``
    attribute (``modeling_utils.py:2052-2057``, 0.40.0), and the PR0 scaffold
    carried two such sites neither of which had ever executed. Counted per
    site through the public ``gradient_checkpointing_func`` seam, and the
    recomputed gradients must equal eager within ``atol=1e-5, rtol=1e-4``
    (fp32) — a site that checkpoints the wrong callable is silent otherwise.
    """
    from app.engine.models.families.minimax_h3.packing import build_layout, packed_forward

    geometry = _geometry()
    layout = build_layout(geometry)
    video_rows, audio_rows, text = _random_inputs(layout, geometry)
    ts = torch.tensor([0.5, 0.3])

    def run(model):
        video_out, audio_out = packed_forward(model, layout, video_rows, audio_rows, text, ts)
        (video_out.float().pow(2).mean() + audio_out.float().pow(2).mean()).backward()
        return _lora_grads(model)

    eager = _lora_wrapped(build_tiny_transformer)
    eager_grads = run(eager)
    assert eager_grads, "eager run produced no LoRA gradients"

    ckpt = _lora_wrapped(build_tiny_transformer)
    ckpt.load_state_dict(eager.state_dict())
    calls: dict[str, int] = {}

    def counting_checkpoint(module, *args):
        calls[type(module).__name__] = calls.get(type(module).__name__, 0) + 1
        return torch.utils.checkpoint.checkpoint(module.__call__, *args, use_reentrant=False)

    base = ckpt.base_model.model
    base.enable_gradient_checkpointing(counting_checkpoint)
    assert base.token_refiner.gradient_checkpointing is True
    assert base.gradient_checkpointing is True
    ckpt_grads = run(ckpt)

    assert calls.get("MiniMaxH3TokenRefinerBlock", 0) >= 1, f"refiner site never executed: {calls}"
    assert calls.get("MiniMaxH3TransformerBlock", 0) >= 1, f"block-stack site never executed: {calls}"
    assert calls["MiniMaxH3TokenRefinerBlock"] == len(base.token_refiner.refiner_blocks)
    assert calls["MiniMaxH3TransformerBlock"] == len(base.transformer_blocks)

    assert ckpt_grads.keys() == eager_grads.keys()
    for name, g in eager_grads.items():
        torch.testing.assert_close(ckpt_grads[name], g, atol=1e-5, rtol=1e-4, msg=lambda m: f"{name}: {m}")


# ── Research §6's free assertions (row 1.7) ─────────────────────────────

_DEFINITIONS_DIR = _TESTS_DIR.parents[0] / "models" / "families" / "minimax_h3" / "definitions"

# The ONLY architecture_params the three definitions may differ in: t2va vs
# fl2va is `mode` (t2v / both); ref2va reads the second checkpoint.
_PARAMS_ALLOWED_TO_DIFFER = frozenset({"mode", "transformer.subfolder"})


def _definition_yamls() -> dict[str, dict[str, Any]]:
    import yaml

    out = {}
    for path in sorted(_DEFINITIONS_DIR.glob("*.yaml")):
        with open(path, encoding="utf-8") as fh:
            out[path.name] = yaml.safe_load(fh)
    assert len(out) >= 3, f"expected the three minimax_h3 definitions, found {sorted(out)}"
    return out


def test_conditioning_rows_need_no_new_module(build_tiny_transformer):
    """(a) fl2v conditioning is a LAYOUT feature: N clean keyframe rows at a
    third timestep slot (`t_c`), tagged video, addressed through the SAME
    AdaLN table (`adaln_indices = slot * MODALITY_NUM + tag`, so every tag
    must stay below `MINIMAX_H3_MODALITY_NUM`); the tiny forward runs on it
    unchanged — no new module (research §3.4 / §6)."""
    from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM

    from app.engine.models.families.minimax_h3.packing import (
        SLOT_CONDITION_VIDEO,
        VIDEO_TAG,
        build_layout,
        num_distinct_timesteps,
        packed_forward,
        patchify_video,
    )

    geometry = _geometry(keyframe_anchors=("first", "last"))
    layout = build_layout(geometry)
    rows_per_frame = (geometry.latent_height // 2) * (geometry.latent_width // 2)
    assert layout.num_condition_video_rows == 2 * rows_per_frame
    cond = layout.video_indices[: layout.num_condition_video_rows]
    # The AdaLN table has `num_timesteps * MODALITY_NUM` rows; a tag outside
    # 0..2 or a fourth slot indexes past it — checked FIRST, it is the failure
    # the other assertions would only describe indirectly.
    assert int(layout.token_tags.max()) < MINIMAX_H3_MODALITY_NUM, (
        f"adaln_indices out of range: token tag {int(layout.token_tags.max())} "
        f">= MINIMAX_H3_MODALITY_NUM ({MINIMAX_H3_MODALITY_NUM})"
    )
    adaln_indices = layout.timestep_slot * MINIMAX_H3_MODALITY_NUM + layout.token_tags
    assert int(adaln_indices.max()) < 3 * MINIMAX_H3_MODALITY_NUM, "adaln_indices out of range"
    assert torch.all(layout.timestep_slot[cond] == SLOT_CONDITION_VIDEO)
    assert torch.all(layout.token_tags[cond] == VIDEO_TAG)
    assert num_distinct_timesteps(layout) == 3

    target_rows, audio_rows, text = _random_inputs(layout, geometry)
    g = torch.Generator().manual_seed(1)
    keyframes = torch.randn(1, 24, 2, geometry.latent_height, geometry.latent_width, generator=g)
    video_rows = torch.cat([patchify_video(keyframes, layout.patch_size), target_rows], dim=1)
    assert video_rows.shape[1] == layout.video_indices.numel()
    model = build_tiny_transformer()
    model.eval()
    with torch.no_grad():
        video_out, audio_out = packed_forward(
            model, layout, video_rows, audio_rows, text, torch.tensor([0.5, 0.3, 0.0])
        )
    assert video_out.shape == video_rows.shape and torch.isfinite(video_out).all()
    assert audio_out.shape == audio_rows.shape and torch.isfinite(audio_out).all()


def test_adaln_excluded_in_all_definitions():
    """(b) no definition targets `adaln_proj`: the AdaLN table differs between
    the full and the pruned checkpoint, so a LoRA touching it is portable to
    neither (research §3.6)."""
    for name, data in _definition_yamls().items():
        targets = data["lora_targetable_modules"]
        assert targets, f"{name}: empty target list"
        offenders = [t for t in targets if "adaln" in t]
        assert not offenders, f"{name} targets adaln modules: {offenders}"


def test_params_differ_only_in_allowlist():
    """(c) the three definitions describe ONE architecture: their
    `architecture_params` differ only in the named allowlist. A stray
    per-definition value (a second `audio.loss_weight`, a private shift) is
    exactly how "one graph, three fine-tunes" would silently stop being true."""
    yamls = _definition_yamls()
    names = sorted(yamls)
    base_name, base = names[0], yamls[names[0]]["architecture_params"]
    for name in names[1:]:
        params = yamls[name]["architecture_params"]
        differing = sorted(k for k in set(base) | set(params) if base.get(k) != params.get(k))
        stray = [k for k in differing if k not in _PARAMS_ALLOWED_TO_DIFFER]
        assert not stray, (
            f"{name} vs {base_name}: architecture_params differ outside the allowlist "
            f"{sorted(_PARAMS_ALLOWED_TO_DIFFER)}: {stray}"
        )


# ── Row 2.2: stereo audio latents — mono VAE twice, channel-major along time
#
# A-5 CONFIRMED by ai-toolkit `pack_audio_latents` (research §3.3):
# `(B, 2, C, T) → (B, 2·T, C)`, all T latents of L then all of R. The reference
# below is written as explicit loops, INDEPENDENT of `packing.py` (ASTRA
# MAJOR-9), so the module and the layout are checked against arithmetic, not
# against each other.

_STEREO_CLICK_WAV = _TESTS_DIR / "fixtures" / "h3_stereo_click.wav"


def test_stereo_packing_reference():
    from app.engine.models.families.minimax_h3 import audio_latents as al
    from app.engine.models.families.minimax_h3.packing import build_layout

    t2va = _definition_yamls()["minimax_h3_t2va.yaml"]["architecture_params"]
    latent_channels = int(t2va["audio_vae.latent_channels"])
    audio_in = int(t2va["transformer.audio_in_channels"])
    T = 5
    latents = torch.arange(2 * latent_channels * T, dtype=torch.float32).reshape(
        1, 2, latent_channels, T
    )

    rows = al.stereo_to_rows(latents)
    assert rows.shape == (1, 2 * T, audio_in), (
        f"audio rows must be {audio_in}-wide (transformer.audio_in_channels), got {tuple(rows.shape)}"
    )
    # Independent reference: row k = channel k // T, time k % T.
    for k in range(2 * T):
        ch, t = k // T, k % T
        expected = latents[0, ch, :, t]
        assert torch.equal(rows[0, k], expected), (
            f"channel order: row {k} should be channel {ch} ({'L' if ch == 0 else 'R'}) "
            f"time {t}, got channel {int(rows[0, k, 0].item()) // (latent_channels * T)}"
        )
    assert torch.equal(al.rows_to_stereo(rows, T), latents)

    # The layout agrees with the reference: 2·T audio rows, [L 0..T−1 | R 0..T−1],
    # the clock strictly monotone inside each block and identical across them.
    layout = build_layout(_geometry(audio_latents=T))
    assert layout.audio_indices.numel() == 2 * T
    times = layout.position_ids[layout.audio_indices, 0]
    left, right = times[:T], times[T:]
    assert torch.all(left[1:] > left[:-1]) and torch.all(right[1:] > right[:-1]), (
        "audio position_ids are not monotone in time inside a channel block"
    )
    assert torch.equal(left, right), "L and R blocks must share one clock"
    assert torch.all(left[1:] - left[:-1] == 1.0), "one rotary unit per audio latent"


def _local_audio_vae_dir() -> str | None:
    """The `audio_vae/` folder of the MiniMax-H3 snapshot in the LOCAL HF
    cache (`huggingface_hub.constants.HF_HUB_CACHE`, the one facade) — never
    a download."""
    try:
        from huggingface_hub import snapshot_download

        root = snapshot_download(
            "MiniMaxAI/MiniMax-H3", allow_patterns=["audio_vae/*"], local_files_only=True
        )
    except Exception:  # noqa: BLE001 — absent / offline == skip, reported with -rs
        return None
    path = Path(root) / "audio_vae"
    return str(path) if (path / "config.json").exists() else None


def test_stereo_round_trip_real_vae():
    """Encode → pack → unpack → decode through the INSTALLED diffusers audio
    VAE with the real weights: an asymmetric fixture (L = 440 Hz sine,
    R = one burst at 0.5 s) proves the channels never cross."""
    vae_dir = _local_audio_vae_dir()
    if vae_dir is None:
        pytest.skip("MiniMax-H3 audio_vae weights are not in the local HF cache")
    import soundfile as sf
    from diffusers import AutoencoderKLMiniMaxH3Audio

    from app.engine.models.families.minimax_h3 import audio_latents as al

    t2va = _definition_yamls()["minimax_h3_t2va.yaml"]["architecture_params"]
    sr = int(t2va["audio.sampling_rate"])
    latent_rate = int(t2va["audio.latent_rate"])

    data, file_sr = sf.read(str(_STEREO_CLICK_WAV), dtype="float32", always_2d=True)
    assert file_sr == sr and data.shape[1] == 2
    wave = torch.from_numpy(data.T).unsqueeze(0)  # (1, 2, N)
    n = wave.shape[-1]

    vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(vae_dir, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        latents = al.encode_stereo(vae, wave)
        assert latents.shape == (1, 2, int(t2va["audio_vae.latent_channels"]), n * latent_rate // sr)
        rows = al.stereo_to_rows(latents)
        assert rows.shape[-1] == int(t2va["transformer.audio_in_channels"])
        back = al.rows_to_stereo(rows, latents.shape[-1])
        assert torch.equal(back, latents)
        out = al.decode_stereo(vae, back)[..., :n]  # (1, 2, N)

    left_in, right_in = wave[0, 0], wave[0, 1]
    left_out, right_out = out[0, 0], out[0, 1]
    one_frame = sr // latent_rate  # 800 samples = 25 ms
    click = n // 2

    # The burst is in R only, at 0.5 s ± one latent frame.
    peak = int(torch.argmax(right_out.abs()))
    assert abs(peak - click) <= one_frame, (
        f"channel order: the burst decoded at {peak / sr:.3f} s in R, expected 0.5 s ± 25 ms"
    )
    window = slice(click - one_frame, click + one_frame)
    # L inside the window is still just the 0.5-amplitude sine: a 0.9 burst
    # leaking across would lift its peak well above 1.5x the sine's.
    assert left_out[window].abs().max() <= 1.5 * left_in.abs().max(), "the burst leaked into L"
    # L-energy in R (outside the burst window) < -30 dB relative to L.
    outside = torch.ones(n, dtype=torch.bool)
    outside[window] = False
    leak_db = 10 * torch.log10(
        right_out[outside].pow(2).mean() / left_out[outside].pow(2).mean()
    )
    assert leak_db < -30, f"L energy leaked into R: {leak_db:.1f} dB"
    # Per-channel correlation with the input.
    def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))

    assert _corr(left_in, left_out) >= 0.95, f"L correlation {_corr(left_in, left_out):.3f}"
    assert _corr(right_in[window], right_out[window]) >= 0.95, (
        f"R burst correlation {_corr(right_in[window], right_out[window]):.3f}"
    )


# ── Row 2.7: the frame rule's give-ups through the SHARED object ─────────────
#
# Research §3.3, confirmed verbatim: the visual VAE chunks 17 pixel frames
# into 5 latent frames with a 5-frame head, so a legal clip length is
# ``17n+5`` (5, 22, 39, 56, 73, 90, 107, ...) and ``97 → 90``. DIVERGENCE
# recorded: diffusion-pipe rounds to ``17n`` (no +5 head), which would call
# 102 legal and snap 97 to 85 — the reference implementations (ai-toolkit,
# musubi, DiffSynth) and the VAE's own arithmetic say ``17n+5``.
# RULE-21: the family reads ``arch["video.frame_rule"]`` — never a literal —
# and snaps through ``video_contract.snap_frames``, the object every video
# family shares, so the ladder here and the temporal bucketing agree.

_H3_RULE_ARCH: dict[str, Any] = {**_ARCH, "video.frame_rule": "17n+5"}


def test_frame_ladder_snaps_down():
    d = _driver(dict(_H3_RULE_ARCH))
    assert d.frame_ladder(107) == [5, 22, 39, 56, 73, 90, 107]
    assert d.snap_num_frames(97) == 90, f"97 → 90 (17·5+5), got {d.snap_num_frames(97)}"
    assert d.snap_num_frames(107) == 107
    assert d.snap_num_frames(120) == 107
    assert d.snap_num_frames(22) == 22 and d.snap_num_frames(21) == 5
    # Definition mutation: a copy with `19n+5` must move the whole ladder —
    # a family literal `"17n+5"` would keep answering 90.
    m = _driver({**_ARCH, "video.frame_rule": "19n+5"})
    assert m.snap_num_frames(97) == 81, f"the rule is not read from the definition (got {m.snap_num_frames(97)})"
    assert m.frame_ladder(107) == [5, 24, 43, 62, 81, 100]
    # No rule in the definition: refused, never a silent default.
    with pytest.raises((KeyError, ValueError)):
        _driver(dict(_ARCH)).snap_num_frames(97)


def test_too_short_clip_refused():
    d = _driver(dict(_H3_RULE_ARCH))
    with pytest.raises(ValueError, match=r"clip has 3 frames; the smallest legal H3 length is 5 \(17n\+5\)"):
        d.assert_clip_frames(3)
    with pytest.raises(ValueError, match=r"clip has 4 frames"):
        d.assert_clip_frames(4)
    assert d.assert_clip_frames(5) is None
    # Long enough but off the ladder: not this check's business (snapping is).
    assert d.assert_clip_frames(6) is None
