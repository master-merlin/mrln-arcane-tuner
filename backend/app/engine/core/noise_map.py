"""The noise-coordinate map: which noise levels each timestep sampler actually trains.

Every registered definition's own trainer class is built without weights, the way the shared
``setup()`` prepares it, and asked for its training draws through its own ``sample_timesteps``.
Each draw is mapped to its noise fraction through the family's own ``add_noise``
(:func:`noise_bands.noise_fraction_of`), and the shares are counted over the validation noise
bands (:func:`noise_bands.band_edges`). No shift, clock, curriculum or band formula is copied
here: a family joins with no edit to this module. CPU only, no weights, no network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
import torch

from app.engine.core.archetypes import resolve_capabilities
from app.engine.core.pipeline.noise_bands import band_edges, band_shares, noise_fraction_of
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.registry import registry
from app.engine.strategies.noise_interpolation import NoiseInterpolation
from app.engine.strategies.timestep_sampling import (
    TIMESTEP_PARAMETERS,
    TIMESTEP_SCHEMA_DEFAULTS,
    TimestepSampler,
    apply_family_timestep_defaults,
)

_logger = structlog.get_logger(__name__)

# What a job's config carries before ``setup()`` that the sampling path may read; bf16 is the one
# precision every family accepts.
_BASE_CONFIG: dict[str, Any] = {
    "mixed_precision": "bf16",
    "te_quantization": "none",
    "cache_text_embeddings": True,
    "sample_prompts": [],
}

PIXEL_SPACE = "has_vae false: pixel space"
GIVEN_DOWNSCALE = "--latent-downscale"
NEEDS_DOWNSCALE = (
    "this sampler depends on resolution and the definition declares no latent downscale: pass --latent-downscale"
)
NO_FLOW_SAMPLING = "declares no flow-matching timestep sampling: only its own draw (default) applies"

# The schema's own default band count: the count a validation pass uses when the job names none.
DEFAULT_BANDS: int = BaseTrainingConfig.model_fields["validation_timestep_bands"].default

# Two latent grids, far apart in token count: a sampler whose whole run draws the same on both is
# taken as independent of resolution (only for modes other than radc, whose term is declared).
_PROBE_CELLS = (16, 256)


def capabilities(definition) -> dict:
    """The definition's resolved capability flags."""
    return resolve_capabilities(definition)["capabilities"]


class _FailedScheduler:
    """Stands in for a scheduler whose construction raised: every attribute read re-raises it."""

    def __init__(self, error: BaseException):
        object.__setattr__(self, "_error", error)

    def __getattr__(self, name: str):
        raise object.__getattribute__(self, "_error")


def sampling_shell(definition, config: dict, steps: int):
    """``definition``'s own trainer class, built without weights, ready to draw its timesteps.

    The config is the base config, then ``config``, then the family's timestep defaults (what the
    shared ``setup()`` does first), then the schema default of each timestep key still missing.
    A scheduler that raises is kept as a stand-in that re-raises on every read, and its error is
    on ``scheduler_error``; a returned ``None`` stays ``None``.
    """
    family = object.__new__(registry.get_family_class(definition.family))
    family.definition = definition
    family.config = {}
    trainer = object.__new__(family.get_trainer_class())
    trainer.definition = definition
    trainer.config = {**_BASE_CONFIG, **config}
    apply_family_timestep_defaults(definition, trainer.config)
    for key, value in TIMESTEP_SCHEMA_DEFAULTS.items():
        trainer.config.setdefault(key, value)
    trainer.device = torch.device("cpu")
    trainer.logger = _logger
    trainer.components = {}
    trainer.epoch = 0
    trainer.global_step = 0
    trainer.max_train_steps = steps
    trainer.optimizer = None
    trainer._setup_family()
    trainer.scheduler_error = None
    try:
        trainer.scheduler = trainer.driver.init_scheduler()
    except Exception as exc:  # noqa: BLE001 -- kept, re-raised on every read of the scheduler
        trainer.scheduler = _FailedScheduler(exc)
        trainer.scheduler_error = f"{type(exc).__name__}: {exc}"
    trainer.noise_interpolation = NoiseInterpolation(
        mode=trainer.config.get("noise_interpolation", "linear"), scheduler=trainer.scheduler)
    return trainer


def _block_out_downscale(value: Any) -> int:
    return 2 ** (len(value) - 1)


# The declared keys that name a definition's spatial VAE downscale, in the order they are read.
_DOWNSCALE_KEYS = (
    ("vae_downsample_factor", int),
    ("vae.spatial_compression_ratio", int),
    ("vae.scale_factor_spatial", int),
    ("vae.vae_scale_factor", int),
    ("vae.block_out_channels", _block_out_downscale),
)


def latent_downscale(definition) -> tuple[int, str] | None:
    """The spatial downscale from pixels to latent cells and the key that declares it.

    ``(1, "has_vae false: pixel space")`` only for a definition without a VAE; ``None`` when a
    model with a VAE declares none of the keys (a missing key never means pixel space).
    """
    arch = getattr(definition, "architecture_params", None) or {}
    for key, read in _DOWNSCALE_KEYS:
        value = arch.get(key)
        if value not in (None, "", []):
            return int(read(value)), key
    if not capabilities(definition).get("has_vae", True):
        return 1, PIXEL_SPACE
    return None


def latents_for(definition, cells: int, frames: int, batch: int) -> torch.Tensor:
    """Zero latents of the family's rank: ``[B, C, frames]`` for audio, ``[B, C, frames, cells,
    cells]`` for video, ``[B, C, cells, cells]`` else; ``C`` from the declared channels."""
    arch = getattr(definition, "architecture_params", None) or {}
    channels = int(arch.get("vae.latent_channels") or arch.get("in_channels") or 16)
    caps = capabilities(definition)
    if caps.get("is_audio_family"):
        return torch.zeros(batch, channels, frames)
    if caps.get("is_video"):
        return torch.zeros(batch, channels, frames, cells, cells)
    return torch.zeros(batch, channels, cells, cells)


# The settings a sampler text may carry: every timestep parameter whose value is one number (the
# ``fixed_sigmas`` list has no form in it).
_NUMBER_SETTINGS = tuple(key for key in TIMESTEP_PARAMETERS if key != "fixed_sigmas")


@dataclass(frozen=True)
class SamplerSpec:
    """A sampler as the run config names it: ``label`` for the report, ``config`` the keys."""

    label: str
    config: dict


def parse_sampler(text: str) -> SamplerSpec:
    """``default`` (the definition's own draw) or ``<mode>[:key=value,...]``.

    ``mode`` is one of the sampler's modes and each key one of its run-config parameters, with a
    number for a value; anything else raises ``ValueError`` naming the bad part.
    """
    label = text.strip()
    if label == "default":
        return SamplerSpec(label, {})
    mode, _, settings = label.partition(":")
    if mode not in TimestepSampler.SUPPORTED_MODES:
        raise ValueError(f"{mode!r} is no timestep mode (one of {', '.join(TimestepSampler.SUPPORTED_MODES)})")
    config: dict[str, Any] = {"timestep_sampling": mode}
    for item in filter(None, (part.strip() for part in settings.split(","))):
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep:
            raise ValueError(f"{item!r} is no key=value setting")
        if key not in _NUMBER_SETTINGS:
            raise ValueError(f"{key!r} is no numeric timestep setting (one of {', '.join(_NUMBER_SETTINGS)})")
        try:
            config[key] = float(value)
        except ValueError:
            raise ValueError(f"{key}={value!r}: {value!r} is no number") from None
    return SamplerSpec(label, config)


class _NotMapped(Exception):
    """A row that can't be mapped, with the reason the row carries."""


def _draw_run(trainer, steps: int, draws_per_step: int, latents: torch.Tensor) -> torch.Tensor:
    """Every step's draw of a training of ``steps`` steps, in step order, as the loop makes them.

    The loop calls the driver's ``on_optimizer_step`` with -1 before step 0 and with ``s`` after
    step ``s``, so step ``s`` follows the call with ``s - 1``. Draws keep the trainer's dtype.
    """
    hook = getattr(trainer.driver, "on_optimizer_step", None)
    draws = []
    for step in range(steps):
        trainer.global_step = step
        if hook is not None:
            hook(step - 1)
        draws.append(trainer.sample_timesteps(draws_per_step, latents).reshape(-1))
    return torch.cat(draws)


def _seeded_shell(definition, sampler: SamplerSpec, steps: int, seed: int):
    """A fresh shell on a freshly seeded default generator (never ``torch.manual_seed``: it also
    reseeds every CUDA generator, which the caller's ``fork_rng`` does not restore)."""
    torch.default_generator.manual_seed(seed)
    return sampling_shell(definition, sampler.config, steps)


def _sampler_config(config: dict) -> dict:
    out = {"timestep_sampling": config.get("timestep_sampling")}
    out.update({key: config[key] for key in TIMESTEP_PARAMETERS if key in config})
    return out


def _part(fractions: torch.Tensor, first: int, end: int, count: int) -> dict:
    return {"steps": [first, end], "shares": band_shares(fractions, count), "median": float(fractions.median())}


def _fill(row: dict, definition, sampler: SamplerSpec, resolution: int, steps: int, phases: int,
          draws_per_step: int, count: int, seed: int, given_downscale: int | None, frames: int) -> None:
    caps = capabilities(definition)
    trainer = _seeded_shell(definition, sampler, steps, seed)
    row["sampler_config"] = _sampler_config(trainer.config)
    row["scheduler_error"] = trainer.scheduler_error
    if sampler.config and not caps.get("supports_flow_timestep_sampling", True):
        raise _NotMapped(NO_FLOW_SAMPLING)

    downscale = (given_downscale, GIVEN_DOWNSCALE) if given_downscale is not None else latent_downscale(definition)
    if downscale is not None:
        cells = resolution // downscale[0]
        latents = latents_for(definition, cells, frames, draws_per_step)
        row.update(latent_grid=[cells, cells], latent_grid_source=downscale[1], latent_shape=list(latents.shape))
        draws = _draw_run(trainer, steps, draws_per_step, latents)
    elif trainer.config.get("timestep_sampling") == "radc":
        # radc's resolution term is declared: never read its absence off the draws.
        if caps.get("supports_radc_resolution", True):
            raise _NotMapped(NEEDS_DOWNSCALE)
        draws = _draw_run(trainer, steps, draws_per_step, latents_for(definition, _PROBE_CELLS[0], frames, draws_per_step))
    else:
        # The complete run on both grids: a term that grows with progress shows only late.
        draws = _draw_run(trainer, steps, draws_per_step, latents_for(definition, _PROBE_CELLS[0], frames, draws_per_step))
        other = _draw_run(_seeded_shell(definition, sampler, steps, seed), steps, draws_per_step,
                          latents_for(definition, _PROBE_CELLS[1], frames, draws_per_step))
        if draws.shape != other.shape or not torch.equal(draws, other):
            raise _NotMapped(NEEDS_DOWNSCALE)

    like = latents_for(definition, 2, 4 if caps.get("is_audio_family") else 3, 1)
    fractions = noise_fraction_of(trainer, draws, like)
    if fractions.numel() % steps:
        raise _NotMapped(f"the trainer drew {fractions.numel()} timesteps over {steps} steps, not the same number each step")
    per_step = fractions.numel() // steps
    bounds = [(i * steps // phases, (i + 1) * steps // phases) for i in range(phases)]
    row["phases"] = [_part(fractions[first * per_step:end * per_step], first, end, count) for first, end in bounds]
    row["all"] = _part(fractions, 0, steps, count)
    row["status"] = "mapped"


def map_definition(
    definition,
    sampler: SamplerSpec,
    resolution: int,
    *,
    steps: int = 2000,
    phases: int = 4,
    draws_per_step: int = 4,
    bands: int | None = None,
    seed: int = 0,
    latent_downscale: int | None = None,
    latent_frames: int = 1,
) -> dict:
    """The share of ``definition``'s training draws per validation noise band, per phase of a run.

    The definition's own trainer draws every step of a simulated training of ``steps`` steps; each
    draw's noise fraction is measured through the family's own ``add_noise``. A row that can't be
    mapped says so (``status`` ``not_mapped``) with its ``reason``; the caller's random streams are
    left as found.
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if phases < 1 or phases > steps:
        raise ValueError(f"phases must lie in [1, steps={steps}], got {phases}")
    if resolution < 1:
        raise ValueError(f"resolution must be at least 1, got {resolution}")
    if draws_per_step < 1:
        raise ValueError(f"draws_per_step must be at least 1, got {draws_per_step}")
    if latent_downscale is not None and latent_downscale < 1:
        raise ValueError(f"latent_downscale must be at least 1, got {latent_downscale}")
    count = DEFAULT_BANDS if bands is None else int(bands)
    row: dict[str, Any] = {
        "definition": definition.id,
        "family": definition.family,
        "sampler": sampler.label,
        "sampler_config": dict(sampler.config),
        "resolution": resolution,
        "latent_grid": None,
        "latent_grid_source": None,
        "latent_frames": latent_frames,
        "latent_shape": None,
        "band_edges": band_edges(count).tolist(),
        "scheduler_error": None,
        "steps": steps,
        "draws_per_step": draws_per_step,
        "seed": seed,
        "status": "not_mapped",
        "reason": None,
        "phases": [],
        "all": None,
    }
    with torch.random.fork_rng(devices=[]):
        try:
            _fill(row, definition, sampler, resolution, steps, phases, draws_per_step, count, seed,
                  latent_downscale, latent_frames)
        except _NotMapped as refused:
            row.update(status="not_mapped", reason=str(refused), phases=[], all=None)
        except Exception as exc:  # noqa: BLE001 -- every failure is the row's reason, never skipped
            row.update(status="not_mapped", reason=f"{type(exc).__name__}: {exc}", phases=[], all=None)
    return row
