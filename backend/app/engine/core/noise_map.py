"""The noise-coordinate map: which noise levels each timestep sampler actually trains.

Every registered definition's own trainer class is built without weights, through the same
constructors a training job uses, and asked for its training draws through its own
``sample_timesteps``. Each draw is mapped to its noise fraction through the family's own
``add_noise`` (:func:`noise_bands.noise_fraction_of`), and the shares are counted over the
validation noise bands (:func:`noise_bands.band_edges`). No shift, clock, curriculum or band
formula is copied here: a family joins with no edit to this module. CPU only, no weights, no network.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import torch

from app.engine.core.archetypes import resolve_capabilities
from app.engine.core.pipeline.noise_bands import band_centres, band_edges, band_shares, noise_fraction_of
from app.engine.core.pipeline.validation_split import setting_bounds
from app.engine.core.video_contract import resolve_video_profile
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.registry import registry
from app.engine.strategies.timestep_sampling import (
    TIMESTEP_PARAMETERS,
    TimestepSampler,
    apply_family_timestep_defaults,
    timestep_setting,
)

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
_BAND_BOUNDS = setting_bounds("validation_timestep_bands")

# Two latent grids, far apart in token count: a sampler whose whole run draws the same on both is
# taken as independent of resolution (only for modes other than radc, whose term is declared).
_PROBE_CELLS = (16, 256)


@dataclass(frozen=True)
class MapOptions:
    """The settings of a map, with their defaults: the one source for rows, the report and the CLI.

    ``bands`` ``None`` is the schema's default count; a count validation refuses is refused here.
    """

    steps: int = 2000
    phases: int = 4
    draws_per_step: int = 4
    bands: int | None = None
    seed: int = 0
    latent_downscale: int | None = None
    latent_frames: int = 1

    def __post_init__(self):
        object.__setattr__(self, "bands", DEFAULT_BANDS if self.bands is None else int(self.bands))
        if self.steps < 1:
            raise ValueError(f"steps must be at least 1, got {self.steps}")
        if self.phases < 1 or self.phases > self.steps:
            raise ValueError(f"phases must lie in [1, steps={self.steps}], got {self.phases}")
        if self.draws_per_step < 1:
            raise ValueError(f"draws_per_step must be at least 1, got {self.draws_per_step}")
        low, high = _BAND_BOUNDS
        if not low <= self.bands <= high:
            raise ValueError(f"bands must lie in [{low:g}, {high:g}] as validation's do, got {self.bands}")
        if self.latent_downscale is not None and self.latent_downscale < 1:
            raise ValueError(f"latent_downscale must be at least 1, got {self.latent_downscale}")


def capabilities(definition) -> dict:
    """The definition's resolved capability flags."""
    return resolve_capabilities(definition)["capabilities"]


def _reason(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class _FailedScheduler:
    """Stands in for a scheduler whose construction raised: every attribute read re-raises it."""

    def __init__(self, error: BaseException):
        object.__setattr__(self, "_error", error)

    def __getattr__(self, name: str):
        raise object.__getattribute__(self, "_error")


def _init_scheduler(trainer):
    """The trainer's own ``init_scheduler`` hook, as ``load_model`` runs it. A hook that reads what
    loading assigns (SDXL's reads the betas ``_assign_components`` caches) raises without weights:
    then its driver's hook, which builds the same scheduler from the driver's own defaults."""
    try:
        return trainer.init_scheduler()
    except Exception:  # noqa: BLE001 -- the driver's hook decides; its error is the one kept
        return trainer.driver.init_scheduler()


def sampling_shell(definition, config: dict, steps: int):
    """``definition``'s own trainer, built as a training job builds it, without weights.

    The run config is the base config, then ``config``, then the family's timestep defaults (what
    the shared ``setup()`` does first), with ``max_train_steps`` ``steps``. The family and the
    trainer come from their own constructors; the device is the CPU. The scheduler comes from
    :func:`_init_scheduler`: one that raises is kept as a stand-in that re-raises on every read,
    its error on ``scheduler_error``; a returned ``None`` stays ``None``.
    """
    run_config = {**_BASE_CONFIG, **config, "max_train_steps": steps}
    apply_family_timestep_defaults(definition, run_config)
    family = registry.get_family_class(definition.family)(definition, run_config)
    trainer = family.get_trainer_class()(definition=definition, run_config=run_config)
    trainer.device = torch.device("cpu")
    trainer.max_train_steps = steps
    trainer._setup_family()
    try:
        scheduler, trainer.scheduler_error = _init_scheduler(trainer), None
    except Exception as exc:  # noqa: BLE001 -- kept, re-raised on every read of the scheduler
        scheduler, trainer.scheduler_error = _FailedScheduler(exc), _reason(exc)
    trainer._init_noise_path(scheduler)
    return trainer


def _block_out_downscale(value: Any) -> int:
    return 2 ** (len(value) - 1)


# The declared keys that name a definition's spatial VAE downscale, in the order they are read;
# a video definition's ``video.vae_spatial`` (its video profile) after them.
_DOWNSCALE_KEYS = (
    ("vae_downsample_factor", int),
    ("vae.spatial_compression_ratio", int),
    ("vae.scale_factor_spatial", int),
    ("vae.vae_scale_factor", int),
    ("vae.block_out_channels", _block_out_downscale),
)
_VIDEO_DOWNSCALE = "video.vae_spatial"


def latent_downscale(definition, caps: dict | None = None) -> tuple[int, str] | None:
    """The spatial downscale from pixels to latent cells and the key that declares it.

    ``(1, "has_vae false: pixel space")`` only for a definition without a VAE; ``None`` when a
    model with a VAE declares none of the keys (a missing key never means pixel space).
    """
    arch = getattr(definition, "architecture_params", None) or {}
    for key, read in _DOWNSCALE_KEYS:
        value = arch.get(key)
        if value not in (None, "", []):
            return int(read(value)), key
    spatial = resolve_video_profile(definition).vae_spatial
    if spatial:
        return spatial, _VIDEO_DOWNSCALE
    if not (caps if caps is not None else capabilities(definition)).get("has_vae", True):
        return 1, PIXEL_SPACE
    return None


def latents_for(definition, cells: int, frames: int, batch: int, caps: dict | None = None) -> torch.Tensor:
    """Zero latents of the family's rank: ``[B, C, frames]`` for audio, ``[B, C, frames, cells,
    cells]`` for video, ``[B, C, cells, cells]`` else; ``C`` from the declared channels."""
    arch = getattr(definition, "architecture_params", None) or {}
    channels = int(arch.get("vae.latent_channels") or arch.get("in_channels") or 16)
    caps = caps if caps is not None else capabilities(definition)
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


def _draw_run(trainer, opts: MapOptions, latents: torch.Tensor) -> torch.Tensor:
    """Every step's draw of a training of ``opts.steps`` steps, in step order, as the loop makes them.

    The loop calls the driver's ``on_optimizer_step`` with -1 before step 0 and with ``s`` after
    step ``s``, so step ``s`` follows the call with ``s - 1``. Draws keep the trainer's dtype.
    """
    hook = getattr(trainer.driver, "on_optimizer_step", None)
    draws = []
    for step in range(opts.steps):
        trainer.global_step = step
        if hook is not None:
            hook(step - 1)
        draws.append(trainer.sample_timesteps(opts.draws_per_step, latents).reshape(-1))
    return torch.cat(draws)


def _seeded_shell(definition, sampler: SamplerSpec, opts: MapOptions):
    """A fresh shell on a freshly seeded default generator (never ``torch.manual_seed``: it also
    reseeds every CUDA generator, which the caller's ``fork_rng`` does not restore)."""
    torch.default_generator.manual_seed(opts.seed)
    return sampling_shell(definition, sampler.config, opts.steps)


def _sampler_config(config: dict, definition) -> dict:
    """The value training reads for the mode and each timestep parameter; ``fixed_sigmas`` only
    when the config carries it (the family default fills it only for the ``fixed_sigmas`` mode)."""
    out = {}
    for key in ("timestep_sampling", *TIMESTEP_PARAMETERS):
        if key == "fixed_sigmas":
            if config.get(key) not in (None, []):
                out[key] = config[key]
            continue
        out[key] = timestep_setting(config, key, definition)
    return out


def _grid_free_draws(trainer, definition, sampler: SamplerSpec, opts: MapOptions, caps: dict,
                     mode: str) -> torch.Tensor:
    """The draws of a definition with no latent grid, refused when they depend on resolution."""
    probe = latents_for(definition, _PROBE_CELLS[0], opts.latent_frames, opts.draws_per_step, caps)
    if mode == "radc":
        # radc's resolution term is declared: never read its absence off the draws.
        if caps.get("supports_radc_resolution", True):
            raise _NotMapped(NEEDS_DOWNSCALE)
        return _draw_run(trainer, opts, probe)
    # The complete run on both grids: a term that grows with progress shows only late.
    draws = _draw_run(trainer, opts, probe)
    other = _draw_run(_seeded_shell(definition, sampler, opts), opts,
                      latents_for(definition, _PROBE_CELLS[1], opts.latent_frames, opts.draws_per_step, caps))
    if draws.shape != other.shape or not torch.equal(draws, other):
        raise _NotMapped(NEEDS_DOWNSCALE)
    return draws


def _shares(trainer, definition, draws: torch.Tensor, opts: MapOptions, caps: dict) -> dict:
    like = latents_for(definition, 2, 4 if caps.get("is_audio_family") else 3, 1, caps)
    fractions = noise_fraction_of(trainer, draws, like)
    steps = opts.steps
    if fractions.numel() % steps:
        raise _NotMapped(f"the trainer drew {fractions.numel()} timesteps over {steps} steps, not the same number each step")
    per_step = fractions.numel() // steps

    def part(fractions, first, end):
        return {"steps": [first, end], "shares": band_shares(fractions, opts.bands), "median": float(fractions.median())}

    bounds = [(i * steps // opts.phases, (i + 1) * steps // opts.phases) for i in range(opts.phases)]
    return {
        "phases": [part(fractions[first * per_step:end * per_step], first, end) for first, end in bounds],
        "all": part(fractions, 0, steps),
    }


def _downscale(definition, opts: MapOptions, caps: dict) -> tuple[int, str] | None:
    if opts.latent_downscale is not None:
        return opts.latent_downscale, GIVEN_DOWNSCALE
    return latent_downscale(definition, caps)


def _map(definition, sampler: SamplerSpec, resolution: int, opts: MapOptions, caps: dict) -> dict:
    if resolution < 1:
        raise ValueError(f"resolution must be at least 1, got {resolution}")
    row: dict[str, Any] = {
        "definition": definition.id,
        "family": definition.family,
        "sampler": sampler.label,
        "sampler_config": dict(sampler.config),
        "resolution": resolution,
        "latent_grid": None,
        "latent_grid_source": None,
        "latent_frames": opts.latent_frames,
        "latent_shape": None,
        "band_edges": band_edges(opts.bands).tolist(),
        "scheduler_error": None,
        "steps": opts.steps,
        "draws_per_step": opts.draws_per_step,
        "seed": opts.seed,
        "status": "not_mapped",
        "reason": None,
        "phases": [],
        "all": None,
    }
    with torch.random.fork_rng(devices=[]):
        try:
            trainer = _seeded_shell(definition, sampler, opts)
            row.update(sampler_config=_sampler_config(trainer.config, definition),
                       scheduler_error=trainer.scheduler_error)
            if sampler.config and not caps.get("supports_flow_timestep_sampling", True):
                raise _NotMapped(NO_FLOW_SAMPLING)
            downscale = _downscale(definition, opts, caps)
            if downscale is None:
                draws = _grid_free_draws(trainer, definition, sampler, opts, caps,
                                         row["sampler_config"]["timestep_sampling"])
            else:
                cells = resolution // downscale[0]
                latents = latents_for(definition, cells, opts.latent_frames, opts.draws_per_step, caps)
                row.update(latent_grid=[cells, cells], latent_grid_source=downscale[1], latent_shape=list(latents.shape))
                draws = _draw_run(trainer, opts, latents)
            row.update(_shares(trainer, definition, draws, opts, caps), status="mapped")
        except Exception as exc:  # noqa: BLE001 -- every failure is the row's reason, never skipped
            reason = str(exc) if isinstance(exc, _NotMapped) else _reason(exc)
            row.update(status="not_mapped", reason=reason, phases=[], all=None)
    return row


def map_definition(definition, sampler: SamplerSpec, resolution: int, **options) -> dict:
    """The share of ``definition``'s training draws per validation noise band, per phase of a run.

    ``options`` are the fields of :class:`MapOptions`. The definition's own trainer draws every
    step of a simulated training of ``steps`` steps; each draw's noise fraction is measured through
    the family's own ``add_noise``. A row that can't be mapped says so (``status`` ``not_mapped``)
    with its ``reason``; the caller's random streams are left as found.
    """
    return _map(definition, sampler, resolution, MapOptions(**options), capabilities(definition))


# ── The report over the registry ──


class Refusal(ValueError):
    """A request the report can't serve (an unknown model name), with the reason."""


def _ensure_registry() -> None:
    if not registry.list_models():
        registry.initialize()


def select_definitions(names: list[str] | None) -> list[str]:
    """The definition ids ``names`` select, each once, in order.

    ``None`` selects every registered definition; a name is a definition id or a family id (all of
    its definitions). Definitions are ordered by ``(family, id)`` within each selection. An unknown
    name raises :class:`Refusal` naming it.
    """
    _ensure_registry()
    by_family: dict[str, list[str]] = {}
    for definition_id in registry.list_models():
        by_family.setdefault(registry.get_definition(definition_id).family, []).append(definition_id)

    def ordered(ids):
        return sorted(ids, key=lambda i: (registry.get_definition(i).family, i))

    if names is None:
        return ordered(registry.list_models())
    unknown = [n for n in names if registry.get_definition(n) is None and n not in by_family]
    if unknown:
        raise Refusal(f"Unknown model name(s): {', '.join(unknown)} (a definition id or a family id)")
    out: list[str] = []
    for name in names:
        for definition_id in [name] if registry.get_definition(name) is not None else ordered(by_family[name]):
            if definition_id not in out:
                out.append(definition_id)
    return out


def build_report(names: list[str] | None, samplers: list[SamplerSpec], resolutions: list[int], **options) -> dict:
    """One :func:`map_definition` row per selected definition, sampler and resolution, in that order,
    with the settings every row shares and whether any row mapped (``any_mapped``).

    ``options`` are the fields of :class:`MapOptions`. A row drawn without a latent grid is the same
    at every resolution, so it is drawn once and copied for the others.
    """
    opts = MapOptions(**options)
    for resolution in resolutions:
        if resolution < 1:
            raise ValueError(f"resolution must be at least 1, got {resolution}")
    rows = []
    for definition_id in select_definitions(names):
        definition = registry.get_definition(definition_id)
        caps = capabilities(definition)
        grid_free = _downscale(definition, opts, caps) is None
        for sampler in samplers:
            first = None
            for resolution in resolutions:
                if grid_free and first is not None:
                    rows.append({**copy.deepcopy(first), "resolution": resolution})
                    continue
                first = _map(definition, sampler, resolution, opts, caps)
                rows.append(first)
    settings = asdict(opts)
    del settings["bands"]
    return {
        "band_edges": band_edges(opts.bands).tolist(),
        "band_centres": band_centres(opts.bands).tolist(),
        **settings,
        "any_mapped": any(row["status"] == "mapped" for row in rows),
        "rows": rows,
    }


def _band_labels(edges: list[float]) -> list[str]:
    """``lo-hi`` per band, with two decimals where every edge has them exactly, else three."""
    digits = 2 if all(abs(e * 100 - round(e * 100)) < 1e-9 for e in edges) else 3
    return [f"{lo:.{digits}f}-{hi:.{digits}f}" for lo, hi in zip(edges, edges[1:])]


def _share_cells(part: dict, width: int) -> str:
    shares = "".join(f"{share * 100:{width - 1}.1f}%" for share in part["shares"])
    return f"{shares}   median {part['median']:.3f}"


def format_report(report: dict) -> str:
    """The report as text: per definition a block, one line per sampler, resolution and phase,
    and a closing ``Not mapped:`` section with each such row's reason (none when every row maps)."""
    labels = _band_labels(report["band_edges"])
    width = max(len(label) for label in labels) + 1
    lines = [
        f"Noise-coordinate map: share of training draws per validation noise band "
        f"({report['steps']} steps, {report['phases']} phases, {report['draws_per_step']} draws per step, "
        f"seed {report['seed']}, latent frames {report['latent_frames']})",
        "Band (noise fraction): " + " | ".join(labels),
    ]
    header = " " * 26 + "".join(f"{label:>{width}}" for label in labels)
    not_mapped = []
    current = None
    for row in report["rows"]:
        if row["definition"] != current:
            current = row["definition"]
            lines += ["", f"== {row['definition']} ({row['family']}) ==", header]
        name = f"{row['sampler']} @ {row['resolution']}"
        if row["status"] != "mapped":
            lines.append(f"  {name}: not mapped")
            not_mapped.append(f"  {row['family']}/{row['definition']} {name}: {row['reason']}")
            continue
        grid = row["latent_grid"]
        where = f"grid {grid[0]}x{grid[1]} ({row['latent_grid_source']})" if grid else "grid-free"
        mode = row["sampler_config"].get("timestep_sampling")
        lines.append(f"  {name}  [{mode}, {where}]")
        if row["scheduler_error"]:
            lines.append(f"    (scheduler not built: {row['scheduler_error']})")
        for part in row["phases"]:
            first, end = part["steps"]
            lines.append(f"    {f'steps {first}-{end}':<22}{_share_cells(part, width)}")
        lines.append(f"    {'all':<22}{_share_cells(row['all'], width)}")
    if not_mapped:
        lines += ["", "Not mapped:", *not_mapped]
    return "\n".join(lines) + "\n"
