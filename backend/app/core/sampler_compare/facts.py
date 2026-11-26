"""Product facts the comparison needs from the engine: each arm's effective sampler and the port
the trainer's loopback calls go to.

The engine imports torch, so the runner asks a child (``facts_in_child``); ``facts_here`` is
what that child runs. ``canonical_sampler`` and ``sampler_mismatch`` stay torch-free: they read
the sampler's names from ``timestep_keys``.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from typing import Any

import psutil

from app.core.sampler_compare.plan import BACKEND
from app.engine.strategies.timestep_keys import PARAMETER_ALIASES, TIMESTEP_PARAMETERS, TIMESTEP_SCHEMA_DEFAULTS

MARKER = "SAMPLER-COMPARE-FACTS "
CHILD_TIMEOUT_S = 300
KILL_WAIT_S = 30
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
    out: dict[str, Any] = {}
    for key in ("timestep_sampling", *TIMESTEP_PARAMETERS):
        if config.get(key) is not None:
            out[key] = _canonical_value(key, config[key])
            continue
        for alias in PARAMETER_ALIASES.get(key, ()):
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
    for key, want in effective.items():
        got = logged.get(key, TIMESTEP_SCHEMA_DEFAULTS.get(key))
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
    from app.engine.strategies.timestep_sampling import apply_family_timestep_defaults

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
    # Every component the loader fetches from the hub (``ModelPathResolver``'s ``huggingface:`` paths,
    # as ``model_utils`` prefetches them): ``isolation.model_source`` records each one's cached snapshot.
    hub_paths = sorted({c.path for c in definition.components.values()
                        if isinstance(getattr(c, "path", None), str) and c.path.startswith("huggingface:")})
    return {"api_port": _resolve_api_port(), "arms": resolved, "hub_paths": hub_paths}


def kill_all(processes: list[psutil.Process], wait_s: float) -> None:
    """Kill every process given and wait up to ``wait_s`` for them to end."""
    for p in processes:
        try:
            p.kill()
        except psutil.Error:
            pass  # gone already
    psutil.wait_procs(processes, timeout=wait_s)


def tree(pid: int) -> list[psutil.Process]:
    """The process ``pid`` and everything under it, as far as it still runs."""
    try:
        root = psutil.Process(pid)
        return [root, *root.children(recursive=True)]
    except psutil.Error:
        return []


def run_child(argv: list[str], env: dict[str, str], timeout_s: float) -> tuple[int, str, str]:
    """(exit code, stdout, stderr) of ``argv``; past ``timeout_s`` it and every process under it
    are killed and ``subprocess.TimeoutExpired`` raises."""
    proc = subprocess.Popen(argv, cwd=BACKEND, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace")
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as expired:
        kill_all(tree(proc.pid), KILL_WAIT_S)
        try:
            proc.communicate(timeout=KILL_WAIT_S)
        except subprocess.TimeoutExpired:
            pass  # a survivor still holds the pipes: the timeout is the answer either way
        raise expired from None
    return proc.returncode, out, err


def facts_in_child(definition_id: str, arms: dict[str, dict], env: dict[str, str], python: str) -> dict:
    """``facts_here`` in a child of ``python`` under ``env``, bounded by ``CHILD_TIMEOUT_S``."""
    argv = [python, "-m", "app.core.sampler_compare.facts", definition_id, json.dumps(arms)]
    try:
        code, stdout, stderr = run_child(argv, env, CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"the product facts child did not answer within {CHILD_TIMEOUT_S} s") from None
    if code != 0:
        raise RuntimeError(f"the product facts child exited with {code}: {stderr[-1500:]}")
    for line in reversed(stdout.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    raise RuntimeError(f"the product facts child printed no facts: {stdout[-1500:]}")


def main(argv: list[str]) -> int:
    definition_id, arms = argv
    sys.stdout.write(MARKER + json.dumps(facts_here(definition_id, json.loads(arms))) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
