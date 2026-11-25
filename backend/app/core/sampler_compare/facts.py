"""Product facts the comparison needs from the engine: each arm's effective sampler, the
timestep keys, and the port the trainer's loopback calls go to.

The engine imports torch, so the runner asks a child (``facts_in_child``); ``facts_here`` is
what that child runs. ``canonical_sampler`` and ``sampler_mismatch`` stay torch-free: they read
the sampler's names from its source (``plan.timestep_source``).
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from typing import Any

from app.core.sampler_compare.plan import BACKEND, timestep_source

MARKER = "SAMPLER-COMPARE-FACTS "
CHILD_TIMEOUT_S = 300
_TOLERANCE = 1e-9


def _canonical_value(key: str, value: Any) -> Any:
    if key == "timestep_sampling":
        return value
    if key == "fixed_sigmas":
        return [float(v) for v in value] if isinstance(value, list) else value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def canonical_sampler(config: dict) -> dict:
    """The config's timestep keys under their canonical names, as the sampler reads them: the
    canonical key first, else its older name. No defaults are added."""
    source = timestep_source()
    aliases = source["_PARAMETER_ALIASES"]
    out: dict[str, Any] = {}
    for key in ("timestep_sampling", *source["TIMESTEP_PARAMETERS"]):
        if config.get(key) is not None:
            out[key] = _canonical_value(key, config[key])
            continue
        for alias in aliases.get(key, ()):
            if config.get(alias) is not None:
                out[key] = _canonical_value(key, config[alias])
                break
    return out


def _differs(got: Any, want: Any) -> bool:
    if isinstance(got, float) and isinstance(want, float):
        return abs(got - want) > _TOLERANCE
    return got != want


def sampler_mismatch(logged_config: dict, effective: dict) -> str | None:
    """Why the sampler a run logged is not the effective one asked for, or None. A key the log
    carries under neither name counts as its schema default."""
    logged = canonical_sampler(logged_config)
    defaults = timestep_source()["TIMESTEP_SCHEMA_DEFAULTS"]
    for key, want in effective.items():
        got = logged.get(key, defaults.get(key))
        if not _differs(got, want):
            continue
        if key == "timestep_sampling":
            return f"the run's sampler is {got!r}, not {want!r}"
        return f"the run's {key} is {got!r}, not {want!r}"
    return None


def facts_here(definition_id: str, arms: dict[str, dict]) -> dict:
    """The facts, resolved in this process (it imports the engine, and with it torch)."""
    from app.engine.core.pipeline.pipeline_data import _resolve_api_port
    from app.engine.models.registry import registry
    from app.engine.strategies.timestep_sampling import (
        _PARAMETER_ALIASES,
        TIMESTEP_PARAMETERS,
        TIMESTEP_SCHEMA_DEFAULTS,
        apply_family_timestep_defaults,
    )

    registry.initialize()
    definition = registry.get_definition(definition_id)
    if definition is None:
        raise ValueError(f"no model definition {definition_id!r}")
    resolved = {}
    for arm, settings in arms.items():
        given = canonical_sampler(settings)
        config = apply_family_timestep_defaults(definition, copy.deepcopy(settings))
        filled = canonical_sampler(config)
        effective: dict[str, Any] = {}
        source: dict[str, str] = {}
        for key in ("timestep_sampling", *TIMESTEP_PARAMETERS):
            if key in filled:
                effective[key] = filled[key]
                source[key] = "arm" if key in given else "family"
            elif key in TIMESTEP_SCHEMA_DEFAULTS:
                effective[key] = _canonical_value(key, TIMESTEP_SCHEMA_DEFAULTS[key])
                source[key] = "schema"
        resolved[arm] = {"effective": effective, "source": source}
    aliases = [name for names in _PARAMETER_ALIASES.values() for name in names]
    return {
        "timestep_keys": sorted({"timestep_sampling", *TIMESTEP_PARAMETERS, *aliases}),
        "api_port": _resolve_api_port(),
        "arms": resolved,
    }


def facts_in_child(definition_id: str, arms: dict[str, dict], env: dict[str, str], python: str) -> dict:
    """``facts_here`` in a child of ``python`` under ``env``, bounded by ``CHILD_TIMEOUT_S``."""
    argv = [python, "-m", "app.core.sampler_compare.facts", definition_id, json.dumps(arms)]
    try:
        done = subprocess.run(argv, cwd=BACKEND, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"the product facts child did not answer within {CHILD_TIMEOUT_S} s") from None
    if done.returncode != 0:
        raise RuntimeError(f"the product facts child exited with {done.returncode}: {done.stderr[-1500:]}")
    for line in reversed(done.stdout.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    raise RuntimeError(f"the product facts child printed no facts: {done.stdout[-1500:]}")


def main(argv: list[str]) -> int:
    definition_id, arms = argv
    sys.stdout.write(MARKER + json.dumps(facts_here(definition_id, json.loads(arms))) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
