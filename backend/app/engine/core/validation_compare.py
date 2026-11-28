"""Compare the held-out validation of training runs: is a gap between two settings more
than noise?

Pure functions over the two files a run with validation writes, ``validation_metrics.jsonl``
and ``validation_split.json``. The module imports only the standard library and never loads
torch, PIL or numpy, so a comparison needs no GPU or model stack, only the backend's
environment (its package, ``app``, imports structlog).

Two kinds of noise stand between a gap and a verdict:

- the **training seed**: two runs of one setting with different seeds differ. The largest
  such same-setting gap is the noise floor, and a gap has to clear it;
- the **measurement**: each held-out image is measured under fixed noise draws. With two or
  more draws, the spread of the gap over the draws says how much of it the draws alone make.

A gap also gets a 95% percentile bootstrap interval over the held-out images. A pair
"beats the noise floor" when that interval excludes 0 and the gap is larger than the floor.
The floor is a minimum bar from a few seeds, not a significance test.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

METRICS_FILE = "validation_metrics.jsonl"
SPLIT_FILE = "validation_split.json"
MIN_IMAGES = 5
DEFAULT_RESAMPLES = 10000
FRESH_FOLDER = "resume into a fresh output folder when you mean to compare runs"
FLOOR_CAVEAT = "The noise floor is a minimum bar from the seeds given, not a significance test."
ONE_DRAW = "one noise draw: measurement noise not measured"

BEATS = "beats the noise floor"
WITHIN = "within the noise floor"


class Refusal(Exception):
    """The runs don't measure the same thing: no report (exit code 2)."""


class Unreadable(Exception):
    """A file can't be read, or isn't what a validation pass writes (exit code 1)."""


# The band count a run measures when its config doesn't set one; the schema's default
# (a test pins the two together, as this module can't import the schema).
DEFAULT_TIMESTEP_BANDS = 8

# The keys of a split file that define what its numbers measure, in the order a refusal
# names the first one that differs: the settings before the holdout they produce, so a
# changed split seed is named as itself rather than as the images it moved.
COMPARED: tuple[str, ...] = ("split_seed", "global_triggerword", "timestep_bands", "holdout", "representatives")

# For each ``COMPARED`` key, what a resume restores, and runs to compare keep the same, when it differs.
RESTORE: dict[str, str] = {
    "split_seed": "the dataset and the split seed",
    "global_triggerword": "the trigger word",
    "timestep_bands": "validation_timestep_bands",
    "holdout": "the dataset and the split seed",
    "representatives": "the held-out images, their captions and their caption prefixes",
}

_CHOOSES_THE_HOLDOUT = "It only chooses the holdout, and the holdout and each held-out image's digest are compared."

# Every other key a split file records, and why two runs may differ in it.
LEFT_OUT: dict[str, str] = {
    "groups": _CHOOSES_THE_HOLDOUT,
    "holdout_fraction": _CHOOSES_THE_HOLDOUT,
    "validation_dataset": _CHOOSES_THE_HOLDOUT,
    "hash_threshold_bits": _CHOOSES_THE_HOLDOUT,
    "every_n_steps": "It sets when a pass runs, not what a pass measures, and each record names its own step.",
    "noise_draws": (
        "Draw 0 is the same noise whatever the count, each pass record names its own count, "
        "and a comparison of runs pairs only the draws every run has."
    ),
    "model": (
        "Older split files don't record it, so a resume of such a run must still be accepted; "
        "a comparison of runs refuses two recorded models on its own."
    ),
}

# The fields of a held-out image's record that two runs may differ in, and why.
REPRESENTATIVE_LEFT_OUT: dict[str, str] = {
    "path": (
        "A checkpoint can move to a machine where the dataset lies in another folder; "
        "the file's content digest is compared, so a name can't change its picture."
    ),
}


def _bands(split: dict[str, Any]) -> Any:
    return split.get("timestep_bands", DEFAULT_TIMESTEP_BANDS)


def recorded_bands(split_text: str) -> Any:
    """The band count a split file records, or the default for a record without one."""
    return _bands(json.loads(split_text))


def _compared_parts(split_text: str) -> dict[str, Any]:
    """Each ``COMPARED`` key's value as the comparison reads it."""
    split = json.loads(split_text)
    return {
        "representatives": [
            {field: value for field, value in item.items() if field not in REPRESENTATIVE_LEFT_OUT}
            for item in split.get("representatives") or []
        ],
        **{key: split.get(key) for key in ("holdout", "split_seed", "global_triggerword")},
        "timestep_bands": _bands(split),
    }


def measurement_identity(split_text: str) -> tuple:
    """``(representatives, holdout, split_seed, global_triggerword)``: what a split file's
    numbers measured before the band count was compared, for a digest recorded then."""
    parts = _compared_parts(split_text)
    return tuple(parts[key] for key in ("representatives", "holdout", "split_seed", "global_triggerword"))


def measurement_difference(earlier: str, content: str) -> str | None:
    """The first ``COMPARED`` key in which two split files differ, or None when they agree."""
    one, other = _compared_parts(earlier), _compared_parts(content)
    return next((key for key in COMPARED if one[key] != other[key]), None)


def same_measurement(earlier: str, content: str) -> bool:
    """Whether two split files measure the same images the same way.

    The keys compared are ``COMPARED``; every other key, and why it may differ, is in
    ``LEFT_OUT`` (a held-out image's own fields in ``REPRESENTATIVE_LEFT_OUT``). A changed
    interval, the draw count (``noise_draws``), or a new image in training keeps the numbers
    comparable. A different holdout doesn't: an image that trained could now be measured;
    nor does another band count, whose means lie on another grid.
    """
    return measurement_difference(earlier, content) is None


# ── Loading ──────────────────────────────────────────────────────────────────


@dataclass
class Run:
    arm: str
    folder: Path
    split_text: str
    records: dict[int, dict[str, Any]] = field(default_factory=dict)

    @property
    def metrics_path(self) -> Path:
        return self.folder / METRICS_FILE

    @property
    def name(self) -> str:
        return f"{self.arm} ({self.folder})"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise Unreadable(f"{path}: {exc}") from None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _loss_or_null(value: Any) -> bool:
    """A pass writes a non-finite loss as null: a valid record, unusable only at its own step."""
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))


def _check_record(path: Path, line: int, record: Any) -> dict[str, Any]:
    def invalid(why: str) -> Unreadable:
        return Unreadable(f"{path}: line {line} is not a validation record: {why}")

    if not isinstance(record, dict) or not _integer(record.get("step")):
        raise invalid("no integer step")
    version = record.get("version", 1)
    if not _integer(version) or version not in (1, 2):
        raise invalid(f"unknown record version {version!r}")
    centres = record.get("band_centres")
    if not isinstance(centres, list) or not centres or not all(_number(c) for c in centres):
        raise invalid("no band centres")
    bands = len(centres)
    if version == 2:
        keys = record.get("image_keys")
        draws = record.get("noise_draws")
        losses = record.get("image_band_loss")
        if not isinstance(keys, list) or not keys or not _integer(draws) or draws < 1:
            raise invalid("no image keys or draw count")
        if (
            not isinstance(losses, list)
            or len(losses) != len(keys)
            or any(not isinstance(image, list) or len(image) != draws for image in losses)
            or any(not isinstance(draw, list) or len(draw) != bands for image in losses for draw in image)
            or not all(_loss_or_null(loss) for image in losses for draw in image for loss in draw)
        ):
            raise invalid("image_band_loss is not [images][draws][bands] of losses")
        seed = record.get("train_seed")
        if seed is not None and not _integer(seed):
            raise invalid("train_seed is neither an integer nor null")
    else:
        band_loss = record.get("band_loss")
        if not isinstance(band_loss, list) or len(band_loss) != bands or not all(_loss_or_null(x) for x in band_loss):
            raise invalid("band_loss doesn't match the band centres")
    return record


def _check_finite(run: Run, step: int, draws: int) -> None:
    """Every loss the comparison uses at ``step`` (its first ``draws`` draws) is a finite number."""
    record = run.records[step]
    if _version(record) == 2:
        losses = [loss for image in record["image_band_loss"] for draw in image[:draws] for loss in draw]
    else:
        losses = record["band_loss"]
    if not all(_number(loss) for loss in losses):
        raise Unreadable(f"{run.metrics_path}: step {step} has a non-finite loss; compare at another --step")


def _not_a_split(split: Any) -> str | None:
    """Why a parsed split file isn't one, or None: the fields every writer has put in it."""
    if not isinstance(split, dict):
        return "not a JSON object"
    holdout = split.get("holdout")
    if not isinstance(holdout, list) or not holdout or not all(isinstance(key, str) for key in holdout):
        return "no list of held-out keys"
    representatives = split.get("representatives")
    if not isinstance(representatives, list) or not all(
        isinstance(item, dict)
        and isinstance(item.get("key"), str)
        and "caption" in item
        and all(_integer(item.get(size)) for size in ("target_w", "target_h"))
        for item in representatives
    ):
        return "the held-out images lack their keys, captions or sizes"
    if sorted(item["key"] for item in representatives) != sorted(holdout):
        return "its held-out images aren't its held-out keys"
    if not _integer(split.get("split_seed")):
        return "no integer split seed"
    model = split.get("model")
    if model is not None and not isinstance(model, str):
        return "model is neither a name nor null"
    return None


def load_run(arm: str, folder: str | Path) -> Run:
    """Read one run's two files; refuse a history whose steps repeat or go back."""
    folder = Path(folder).resolve()
    run = Run(arm=arm, folder=folder, split_text=_read(folder / SPLIT_FILE))
    try:
        split = json.loads(run.split_text)
    except json.JSONDecodeError as exc:
        raise Unreadable(f"{folder / SPLIT_FILE}: {exc}") from None
    if why := _not_a_split(split):
        raise Unreadable(f"{folder / SPLIT_FILE} is not a validation split file: {why}")

    path = run.metrics_path
    last: int | None = None
    grid: list[float] | None = None
    for line, text in enumerate(_read(path).splitlines(), start=1):
        if not text.strip():
            continue
        try:
            record = _check_record(path, line, json.loads(text))
        except json.JSONDecodeError as exc:
            raise Unreadable(f"{path}: line {line}: {exc}") from None
        step = record["step"]
        if last is not None and step <= last:
            what = f"repeats step {step}" if step in run.records else f"goes back to step {step} after step {last}"
            raise Refusal(
                f"{path}: line {line} {what}. A resume into the same folder appends to its history, "
                f"and no record says which training it came from; {FRESH_FOLDER}."
            )
        if grid is not None and record["band_centres"] != grid:
            raise Refusal(f"{path}: step {step} has other band centres than the steps before it (a mixed band grid).")
        grid = record["band_centres"]
        run.records[step] = record
        last = step
    if not run.records:
        raise Unreadable(f"{path}: no validation records")
    return run


# ── The numbers ──────────────────────────────────────────────────────────────


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _version(record: dict[str, Any]) -> int:
    return 2 if record.get("version", 1) == 2 else 1


def _images(record: dict[str, Any], draws: list[int]) -> list[list[float]]:
    """``[image][band]``: each image's mean over the given draws."""
    return [
        [_mean([image[d][b] for d in draws]) for b in range(len(image[0]))]
        for image in record["image_band_loss"]
    ]


def _aggregate(record: dict[str, Any]) -> list[float]:
    """``[band]`` from draw 0 only, the one observation every record version has."""
    if _version(record) == 1:
        return list(record["band_loss"])
    return [_mean(column) for column in zip(*_images(record, [0]), strict=True)]


def _arm_images(runs: list[Run], step: int, draws: list[int]) -> list[list[float]]:
    per_run = [_images(run.records[step], draws) for run in runs]
    return [
        [_mean([values[i][b] for values in per_run]) for b in range(len(per_run[0][0]))]
        for i in range(len(per_run[0]))
    ]


def _gap(a: list[list[float]], b: list[list[float]]) -> tuple[float, list[float], list[list[float]]]:
    """Overall gap, per-band gaps, and the per-image differences ``[image][band]``."""
    differences = [[x - y for x, y in zip(ia, ib, strict=True)] for ia, ib in zip(a, b, strict=True)]
    bands = [_mean(column) for column in zip(*differences, strict=True)]
    return _mean(bands), bands, differences


def _percentile_interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    count = len(ordered)
    low = ordered[int(0.025 * count)]
    high = ordered[max(0, math.ceil(0.975 * count) - 1)]
    return [low, high]


def _bootstrap(differences: list[list[float]], resamples: list[list[int]]) -> tuple[list[float], list[list[float]]]:
    """Intervals of the overall and each band's gap, resampling images."""
    bands = len(differences[0])
    overall_samples: list[float] = []
    band_samples: list[list[float]] = [[] for _ in range(bands)]
    for picks in resamples:
        rows = [differences[i] for i in picks]
        per_band = [_mean([row[b] for row in rows]) for b in range(bands)]
        overall_samples.append(_mean(per_band))
        for b in range(bands):
            band_samples[b].append(per_band[b])
    return _percentile_interval(overall_samples), [_percentile_interval(s) for s in band_samples]


def _verdict(gap: float, interval: list[float] | None, floor: float | None, reasons: list[str]) -> str:
    if interval is None or floor is None:
        return "no verdict: " + "; ".join(reasons)
    excludes_zero = interval[0] > 0 or interval[1] < 0
    return BEATS if excludes_zero and abs(gap) > floor else WITHIN


def _check_models(runs: list[Run]) -> list[str]:
    """Refuse runs of two recorded models: their losses are on other scales, or of other targets.
    -> a note for each run whose split file names no model (written before models were recorded)."""
    named = [(run, json.loads(run.split_text).get("model")) for run in runs]
    known = [(run, model) for run, model in named if model is not None]
    for run, model in known[1:]:
        if model != known[0][1]:
            raise Refusal(
                f"{known[0][0].folder / SPLIT_FILE} measures model {known[0][1]} and {run.folder / SPLIT_FILE} "
                f"model {model}: different models' losses are not on one scale."
            )
    return [
        f"{run.folder / SPLIT_FILE} doesn't record its model (written before models were recorded): "
        "make sure every run trains the same model."
        for run, model in named
        if model is None
    ]


def _digest_reasons(runs: list[Run]) -> list[str]:
    """Why no verdict can rest on each run whose split file lacks a held-out image's content digest
    (written before digests were recorded): without them, one file name may hold different pictures."""
    return [
        f"no interval: {run.folder / SPLIT_FILE} records no content digest for some held-out image "
        "(written before digests were recorded), so the same names may hold different pictures"
        for run in runs
        if not all(
            isinstance(item.get("sha256"), str) and item["sha256"]
            for item in json.loads(run.split_text)["representatives"]
        )
    ]


def _check_comparable(runs: list[Run], step: int) -> None:
    first = runs[0]
    for run in runs[1:]:
        key = measurement_difference(first.split_text, run.split_text)
        if key is not None:
            message = (
                f"{first.folder / SPLIT_FILE} and {run.folder / SPLIT_FILE} don't hold out the same "
                f"images the same way: their numbers measure different things. They differ in {key}."
            )
            if key == "timestep_bands":
                message += (
                    f" {first.folder / SPLIT_FILE} records {recorded_bands(first.split_text)} noise bands and "
                    f"{run.folder / SPLIT_FILE} {recorded_bands(run.split_text)}: compare runs trained with one "
                    f"{RESTORE[key]}."
                )
            else:
                message += f" Keep {RESTORE[key]} the same in the runs you compare."
            raise Refusal(message)
    centres = first.records[step]["band_centres"]
    for run in runs[1:]:
        if run.records[step]["band_centres"] != centres:
            raise Refusal(f"{run.metrics_path}: step {step} has other band centres than {first.metrics_path}.")
    measured = [run for run in runs if _version(run.records[step]) == 2]
    for run in measured[1:]:
        if run.records[step]["image_keys"] != measured[0].records[step]["image_keys"]:
            raise Refusal(
                f"{run.metrics_path}: step {step} measured other images than {measured[0].metrics_path}."
            )
    seen: dict[tuple[str, int], Run] = {}
    for run in measured:
        seed = run.records[step].get("train_seed")
        if seed is None:
            continue
        twin = seen.setdefault((run.arm, seed), run)
        if twin is not run:
            raise Refusal(
                f"{twin.metrics_path} and {run.metrics_path}: arm {run.arm} has two runs with training "
                f"seed {seed}, which would make a noise floor of one training."
            )


def _floors(arms: dict[str, list[Run]], step: int, values) -> tuple[dict[str, dict[str, Any] | None], list[str]]:
    """Each arm's largest same-arm gap between seeds, and why an arm has none."""
    per_arm: dict[str, dict[str, Any] | None] = {}
    reasons: list[str] = []
    for arm, runs in arms.items():
        why = None
        if len(runs) < 2:
            why = f"no noise floor: arm {arm} has one seed run"
        else:
            for run in runs:
                record = run.records[step]
                if _version(record) == 1 or record.get("train_seed") is None:
                    why = f"no noise floor: {run.name} has no recorded training seed"
                    break
        if why:
            per_arm[arm] = None
            reasons.append(why)
            continue
        overall, bands = 0.0, None
        for a, b in combinations(runs, 2):
            gap, band_gaps, _ = _gap(values([a]), values([b]))
            overall = max(overall, abs(gap))
            band_abs = [abs(g) for g in band_gaps]
            bands = band_abs if bands is None else [max(x, y) for x, y in zip(bands, band_abs, strict=True)]
        per_arm[arm] = {"overall": overall, "bands": bands}
    return per_arm, reasons


def _default_step(runs: list[Run]) -> int:
    common = set(runs[0].records)
    for run in runs[1:]:
        common &= set(run.records)
    if not common:
        raise Refusal("No step is in every run's " + METRICS_FILE + ": " + ", ".join(str(r.metrics_path) for r in runs))
    return max(common)


def compare(
    runs: list[tuple[str, str | Path]],
    step: int | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> dict[str, Any]:
    """The report over ``(arm, folder)`` runs at one step, as a dict.

    Raises Refusal when the runs don't measure the same thing, and Unreadable when a file
    can't be read.
    """
    folders: dict[Path, str] = {}
    for arm, folder in runs:
        resolved = Path(folder).resolve()
        if resolved in folders:
            raise Refusal(f"{resolved} is given twice (as {folders[resolved]} and {arm}).")
        folders[resolved] = arm
    if len({arm for arm, _ in runs}) < 2:
        raise Refusal(
            "A comparison needs at least two arms; got only "
            + ", ".join(f"{arm}={Path(folder).resolve()}" for arm, folder in runs)
        )
    if resamples < 1:
        raise Refusal("resamples must be at least 1")

    loaded = [load_run(arm, folder) for arm, folder in runs]
    if step is None:
        step = _default_step(loaded)
    for run in loaded:
        if step not in run.records:
            raise Refusal(f"{run.metrics_path}: no record at step {step}.")
    _check_comparable(loaded, step)
    notes = _check_models(loaded)

    arms: dict[str, list[Run]] = {}
    for run in loaded:
        arms.setdefault(run.arm, []).append(run)
    records = [run.records[step] for run in loaded]
    centres = records[0]["band_centres"]
    bands = len(centres)
    legacy_runs = [run for run in loaded if _version(run.records[step]) == 1]
    legacy = bool(legacy_runs)
    shared_draws = 1 if legacy else min(record["noise_draws"] for record in records)
    for run in loaded:
        _check_finite(run, step, shared_draws)
    interval_reasons: list[str] = []

    if legacy:
        images = None
        interval_reasons = [
            f"no interval: {run.name} has no per-image losses (recorded before per-image losses were kept)"
            for run in legacy_runs
        ]

        def values(group: list[Run]) -> list[list[float]]:
            # One aggregate observation per run, as a single "image" row.
            return [[_mean(list(column)) for column in zip(*(_aggregate(r.records[step]) for r in group), strict=True)]]
    else:
        images = len(records[0]["image_keys"])
        draws = list(range(shared_draws))
        if images < MIN_IMAGES:
            interval_reasons = [f"too few held-out images ({images}) for an interval; at least {MIN_IMAGES}"]

        def values(group: list[Run]) -> list[list[float]]:
            return _arm_images(group, step, draws)

    interval_reasons += _digest_reasons(loaded)
    per_arm_floor, floor_reasons = _floors(arms, step, values)
    if floor_reasons:
        floor = None
    else:
        floor = {
            "overall": max(f["overall"] for f in per_arm_floor.values()),
            "bands": [max(f["bands"][b] for f in per_arm_floor.values()) for b in range(bands)],
        }

    rng = random.Random(seed)
    picks = (
        [[rng.randrange(images) for _ in range(images)] for _ in range(resamples)]
        if images is not None and not interval_reasons
        else None
    )
    if shared_draws < 2:
        notes.append(ONE_DRAW)
    notes.append(FLOOR_CAVEAT)
    reasons = interval_reasons + floor_reasons

    pairs = []
    for a, b in combinations(arms, 2):
        gap, band_gaps, differences = _gap(values(arms[a]), values(arms[b]))
        if picks is not None:
            interval, band_intervals = _bootstrap(differences, picks)
        else:
            interval, band_intervals = None, [None] * bands
        pair: dict[str, Any] = {
            "a": a,
            "b": b,
            "overall": {
                "gap": gap,
                "interval": interval,
                "verdict": _verdict(gap, interval, floor and floor["overall"], reasons),
            },
            "bands": [
                {
                    "centre": centres[k],
                    "gap": band_gaps[k],
                    "interval": band_intervals[k],
                    "verdict": _verdict(band_gaps[k], band_intervals[k], floor and floor["bands"][k], reasons),
                }
                for k in range(bands)
            ],
            "measurement_noise": None,
        }
        if not legacy and shared_draws >= 2:
            per_draw = [_gap(_arm_images(arms[a], step, [d]), _arm_images(arms[b], step, [d]))[:2] for d in range(shared_draws)]

            def noise(gaps: list[float], whole: float) -> dict[str, Any]:
                deviation = statistics.stdev(gaps)
                return {
                    "per_draw": gaps,
                    "sd": deviation,
                    "min": min(gaps),
                    "max": max(gaps),
                    "as_large_as_gap": deviation >= abs(whole),
                }

            pair["measurement_noise"] = {
                "overall": noise([g for g, _ in per_draw], gap),
                "bands": [noise([bg[k] for _, bg in per_draw], band_gaps[k]) for k in range(bands)],
            }
        pairs.append(pair)

    return {
        "step": step,
        "runs": [
            {
                "arm": run.arm,
                "folder": str(run.folder),
                "version": _version(run.records[step]),
                "train_seed": run.records[step].get("train_seed"),
                "noise_draws": run.records[step].get("noise_draws", 1),
            }
            for run in loaded
        ],
        "images": images,
        "shared_draws": shared_draws,
        "band_centres": centres,
        "resamples": resamples,
        "seed": seed,
        "floor": floor,
        "arm_floors": per_arm_floor,
        "reasons": reasons,
        "notes": notes,
        "pairs": pairs,
    }


# ── Text ─────────────────────────────────────────────────────────────────────


def _f(value: float | None) -> str:
    return "-" if value is None else f"{value:+.5f}"


def _plain(value: float | None) -> str:
    return "none" if value is None else f"{value:.5f}"


def _interval(interval: list[float] | None) -> str:
    return "no interval" if interval is None else f"[{interval[0]:+.5f}, {interval[1]:+.5f}]"


def format_report(report: dict[str, Any]) -> str:
    """The report as a plain text table."""
    lines = [f"Validation comparison at step {report['step']}"]
    for run in report["runs"]:
        lines.append(
            f"  {run['arm']}: {run['folder']} (record version {run['version']}, "
            f"training seed {'unknown' if run['train_seed'] is None else run['train_seed']}, "
            f"{run['noise_draws']} draw(s))"
        )
    images = report["images"]
    lines.append(
        f"Held-out images: {'unknown (older records)' if images is None else images}; "
        f"shared noise draws: {report['shared_draws']}; bootstrap: {report['resamples']} resamples, seed {report['seed']}"
    )
    floor = report["floor"]
    lines.append(f"Noise floor: {_plain(floor and floor['overall'])}")
    for arm, arm_floor in report["arm_floors"].items():
        lines.append(f"  arm {arm}: {_plain(arm_floor and arm_floor['overall'])}")
    if report["reasons"]:
        lines.append("No verdict, because:")
    for reason in report["reasons"]:
        lines.append(f"  {reason}")
    for pair in report["pairs"]:
        lines.append("")
        lines.append(f"{pair['a']} - {pair['b']}")
        rows = [("overall", pair["overall"], floor and floor["overall"])]
        rows += [(f"band {b['centre']:.4f}", b, floor and floor["bands"][k]) for k, b in enumerate(pair["bands"])]
        noise = pair["measurement_noise"]
        for k, (label, row, row_floor) in enumerate(rows):
            text = f"  {label:<12} gap {_f(row['gap'])}  95% {_interval(row['interval'])}  floor {_plain(row_floor)}"
            if noise is not None:
                spread = noise["overall"] if k == 0 else noise["bands"][k - 1]
                text += f"  draws sd {spread['sd']:.5f} (min {_f(spread['min'])}, max {_f(spread['max'])})"
                if spread["as_large_as_gap"]:
                    text += "  measurement noise alone is as large as the gap"
            verdict = row["verdict"]
            if verdict.startswith("no verdict: "):
                verdict = "no verdict (reasons above)"  # the full reason is in the JSON
            lines.append(text + f"  -> {verdict}")
    lines.append("")
    lines.extend(report["notes"])
    return "\n".join(lines) + "\n"
