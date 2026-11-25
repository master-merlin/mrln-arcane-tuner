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
