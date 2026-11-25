"""The comparison's report: the noise floor and every pair's verdict over the finished runs, with
each run's telemetry and effective sampler.

Every finished run passes the same guards as a reuse (``runner.reusable``: identity, verdict,
sampler, seed, split) before any comparison; one refused run leaves the report without verdicts,
naming it, rather than comparing the rest silently.
"""
from __future__ import annotations

import datetime
import json
from typing import Any

from app.core.sampler_compare import isolation, plan, runner
from app.core.sampler_compare.plan import Comparison
from app.engine.core import validation_compare as vc

REPORT_VERSION = 1
_SOURCES = {"arm": "from the arm", "family": "from the family", "schema": "from the schema"}
_PEAKS = ("minutes", "peak_temperature_c", "peak_vram_mib", "peak_power_w", "min_available_ram_gib",
          "peak_rss_gib", "mean_step_time_s", "stopped_for")


def _last_runs(c: Comparison) -> dict[str, dict]:
    """Each label's last record of a run that trained."""
    path = c.work_dir / "runs.jsonl"
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("ran"):
                out[record.get("label")] = record
    return out


def _sampler_text(facts: dict, arm: str) -> dict[str, str]:
    resolved = facts["arms"][arm]
    return {key: f"{value} {_SOURCES.get(resolved['source'].get(key), 'from ' + str(resolved['source'].get(key)))}"
            for key, value in resolved["effective"].items()}


def _settings(c: Comparison, definition_id: str) -> dict[str, Any]:
    return {
        "name": c.name, "model": definition_id, "reference_job": c.reference_job, "dataset": c.dataset,
        "arms": c.arms, "seeds": list(c.seeds), "steps": c.steps,
        "passes": plan.passes(c.steps, c.validation.every),
        "validation": {"every": c.validation.every, "noise_draws": c.validation.noise_draws,
                       "holdout_fraction": c.validation.holdout_fraction, "split_seed": c.validation.split_seed},
        "limits": vars(c.limits), "allow_downloads": c.allow_downloads,
    }


def _rows(c: Comparison, facts: dict, requested: dict, definition_id: str) -> tuple[list[dict], list, list[str]]:
    """(rows, accepted (arm, folder) pairs, refusals) over the queue."""
    target = c.work_dir / isolation.DATASET_FOLDER
    manifest = isolation.read_manifest(target) if isolation.manifest_path(target).exists() else {}
    last = _last_runs(c)
    rows, accepted, refusals = [], [], []
    for arm, seed in plan.queue(c):
        name = plan.label(arm, seed)
        folder = runner.run_dir(c, name, definition_id)
        finished = folder.exists() and runner.is_complete(folder, c)
        why = runner.reusable(folder, c, name, requested[name], definition_id, manifest) if finished else None
        if finished and why:
            refusals.append(f"{name}: {why}")
        elif finished:
            accepted.append((arm, folder))
        record = last.get(name, {})
        rows.append({
            "label": name, "arm": arm, "seed": seed, "finished": finished, "accepted": finished and why is None,
            "refused": why, "run_dir": str(folder) if folder.exists() else None,
            **{key: record.get(key) for key in _PEAKS},
            "sampler": _sampler_text(facts, arm),
        })
    return rows, accepted, refusals


def _pairs(result: dict) -> list[dict]:
    return [{"a": p["a"], "b": p["b"], "gap": p["overall"]["gap"], "interval": p["overall"]["interval"],
             "verdict": p["overall"]["verdict"]} for p in result["pairs"]]


def _cell(value: Any, width: int) -> str:
    return f"{'-' if value is None else value:>{width}}"


def _text(out: dict) -> str:
    lines = []
    if out["queue_stop"]:
        lines.append(f"Queue stopped: {out['queue_stop']}")
    floor = out["floor"]
    if out["compare_error"]:
        lines.append(f"No verdicts: {out['compare_error']}")
    else:
        overall = floor and floor["overall"]
        lines.append(f"Noise floor: {'none' if overall is None else f'{overall:.5f}'} "
                     "(the largest gap between two seeds of one arm)")
        for p in out["pairs"]:
            interval = "no interval" if p["interval"] is None else f"[{p['interval'][0]:+.5f}, {p['interval'][1]:+.5f}]"
            lines.append(f"{p['a']} - {p['b']}: gap {p['gap']:+.5f}, 95% {interval}: {p['verdict']}")
    lines += ["", f"Settings: {json.dumps(out['settings'])}", ""]
    lines.append(f"{'run':<28}{'done':<6}{'used':<6}{'min':>7}{'peak C':>8}{'VRAM MiB':>10}{'W':>7}"
                 f"{'min RAM GiB':>13}  stop")
    for row in out["runs"]:
        used = "yes" if row["accepted"] else ("no" if row["finished"] else "-")
        stop = row["refused"] or row["stopped_for"] or ""
        lines.append(f"{row['label']:<28}{'yes' if row['finished'] else 'no':<6}{used:<6}{_cell(row['minutes'], 7)}"
                     f"{_cell(row['peak_temperature_c'], 8)}{_cell(row['peak_vram_mib'], 10)}"
                     f"{_cell(row['peak_power_w'], 7)}{_cell(row['min_available_ram_gib'], 13)}  {stop}")
        lines.append("    sampler: " + ", ".join(f"{k} {v}" for k, v in row["sampler"].items()))
    lines.append("")
    lines += [f"- {note}" for note in out["notes"]]
    return "\n".join(lines) + "\n"


def write_report(c: Comparison, queue_stop: str | None, facts: dict, requested: dict[str, dict],
                 definition_id: str) -> int:
    """Write ``report.json`` and ``report.txt``; -> 0 with a comparison, 1 without."""
    rows, accepted, refusals = _rows(c, facts, requested, definition_id)
    result, compare_error = None, None
    arms = {arm for arm, _ in accepted}
    if refusals:
        compare_error = "not compared: a finished run is refused: " + "; ".join(refusals)
    elif len(arms) < 2:
        compare_error = f"not compared: {len(arms)} arm(s) finished"
    else:
        try:
            result = vc.compare(accepted)
        except (vc.Refusal, vc.Unreadable) as exc:
            compare_error = f"not compared: {exc}"
    notes = list(result["notes"]) if result else [vc.FLOOR_CAVEAT]
    notes.append("Gaps read the first arm minus the second: negative means the first arm's held-out loss is lower.")
    notes.append("Weight downloads were allowed: a missing weight was written into the shared weight cache."
                 if c.allow_downloads else "Weight downloads were off: nothing was written into the shared weight cache.")
    out = {
        "version": REPORT_VERSION,
        "written": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        "settings": _settings(c, definition_id),
        "queue_stop": queue_stop,
        "runs": rows,
        "floor": result["floor"] if result else None,
        "pairs": _pairs(result) if result else [],
        "notes": notes,
        "compare_error": compare_error,
    }
    isolation.write_json(c.work_dir / "report.json", out)
    isolation.write_bytes(c.work_dir / "report.txt", _text(out).encode("utf-8"))
    return 0 if result else 1
