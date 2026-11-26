"""Product facts the comparison needs from the engine: each arm's effective sampler and the port
the trainer's loopback calls go to.

The engine imports torch, so the runner asks a child (``facts_in_child``); ``facts_here`` is
what that child runs. ``canonical_sampler`` and ``sampler_mismatch`` stay torch-free: they read
the sampler's names from ``timestep_keys``.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil

from app.core.sampler_compare.plan import BACKEND
from app.engine.strategies.timestep_keys import PARAMETER_ALIASES, TIMESTEP_PARAMETERS, TIMESTEP_SCHEMA_DEFAULTS

MARKER = "SAMPLER-COMPARE-FACTS "
CHILD_TIMEOUT_S = 300
DOWNLOAD_TIMEOUT_S = 6 * 3600  # a missing model of tens of GB on a slow line, still bounded
MASKS_TIMEOUT_S = 3600  # every masked picture of a large dataset, made again once
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


def refusal(definition: Any) -> str | None:
    """Why no comparison can train ``definition``, or None: the product's own checks, as its
    trainer would make them once the weights are loaded. A definition whose trainer draws its own
    timesteps (``supports_flow_timestep_sampling`` off) takes no sampler setting, and every arm
    measures validation loss, which ``validation_split.refusal_reason`` refuses without the
    trainer's ``validation_timestep_domain``."""
    from app.engine.core.noise_map import NO_FLOW_SAMPLING, capabilities
    from app.engine.core.pipeline import validation_split
    from app.engine.models.registry import registry

    if not capabilities(definition).get("supports_flow_timestep_sampling", True):
        return f"{definition.id} {NO_FLOW_SAMPLING}; no sampler can be compared on it"
    family = registry.get_family_class(definition.family)(definition, {})
    trainer = family.get_trainer_class()
    return validation_split.refusal_reason(getattr(family, "family_name", None) or definition.family,
                                           getattr(trainer, "validation_timestep_domain", None), [], {}, {})


def hub_paths(definition: Any) -> list[str]:
    """Every component the loader fetches from the hub (``ModelPathResolver``'s ``huggingface:``
    paths, as ``model_utils`` prefetches them): ``isolation.model_source`` records each one's
    cached snapshot."""
    return sorted({c.path for c in definition.components.values()
                   if isinstance(getattr(c, "path", None), str) and c.path.startswith("huggingface:")})


def facts_here(definition_id: str, arms: dict[str, dict]) -> dict:
    """The facts, resolved in this process (it imports the engine, and with it torch). A definition
    no comparison can train (``refusal``) raises ``ValueError`` naming why."""
    from app.engine.core.pipeline.pipeline_data import _resolve_api_port
    from app.engine.models.registry import registry
    from app.engine.strategies.timestep_sampling import apply_family_timestep_defaults

    registry.initialize()
    definition = registry.get_definition(definition_id)
    if definition is None:
        raise ValueError(f"no model definition {definition_id!r}")
    reason = refusal(definition)
    if reason:
        raise ValueError(f"the comparison can't train {definition_id}: {reason}")
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
    return {"api_port": _resolve_api_port(), "arms": resolved, "hub_paths": hub_paths(definition),
            "local_hub_paths": loader_hub_paths(definition)}


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


def _family_loader(definition: Any) -> Any:
    """The family's own component loader (the one ``GenericComponentLoader`` subclass its
    ``loader`` module defines, which its trainer builds), on the CPU; nothing is loaded."""
    import importlib
    import inspect

    from app.engine.core.pipeline.loader_base import GenericComponentLoader

    module = importlib.import_module(f"app.engine.models.families.{definition.family}.loader")
    found = [cls for _, cls in inspect.getmembers(module, inspect.isclass)
             if issubclass(cls, GenericComponentLoader) and cls.__module__ == module.__name__]
    if len(found) != 1:
        raise RuntimeError(f"family {definition.family} defines {len(found)} component loaders, not one")
    return found[0]("cpu")


def loader_hub_paths(definition: Any) -> list[str]:
    """The ``huggingface:`` paths the family's loader still resolves from the hub under a local
    Diffusers root: its own ``_resolve_component_path`` for every component of its manifest, run
    against an empty stand-in root, with ``ModelPathResolver.resolve`` only noting hub paths."""
    import tempfile

    from app.engine.utils.model_utils import ModelPathResolver

    loader, seen = _family_loader(definition), set()
    real = ModelPathResolver.resolve
    with tempfile.TemporaryDirectory(prefix="sampler-compare-root-") as root:
        def noting(path_str, base_dir=None, *, local_files_only=False):
            if isinstance(path_str, str) and path_str.startswith("huggingface:"):
                seen.add(path_str)
                return os.path.join(root, "hub")
            return real(path_str, base_dir, local_files_only=local_files_only)

        ModelPathResolver.resolve = staticmethod(noting)
        try:
            for spec in loader.get_component_manifest(definition):
                loader._resolve_component_path(spec, definition, root)
        finally:
            ModelPathResolver.resolve = staticmethod(real)
    return sorted(seen)


def prefetch_here(definition_id: str) -> None:
    """Download what the definition loads from the hub and the cache lacks, the product's way
    (``ModelPathResolver.ensure_definition_cached``, which leaves a local override alone), and
    under a local Diffusers root the hub components its loader still resolves (``loader_hub_paths``).
    The hub token is applied first, as the backend and the trainer apply it at their start."""
    from app.core.hf_auth import apply_hf_auth
    from app.core.schemas.model_overrides import ModelSourceType
    from app.engine.models.registry import registry
    from app.engine.utils.model_override_manager import ModelOverrideManager
    from app.engine.utils.model_utils import ModelPathResolver

    apply_hf_auth(ModelOverrideManager.get_all().hf_token)
    registry.initialize()
    definition = registry.get_definition(definition_id)
    if definition is None:
        raise ValueError(f"no model definition {definition_id!r}")
    ModelPathResolver.ensure_definition_cached(definition)
    source_type, _, _ = ModelOverrideManager.resolve_effective_source(definition.id)
    if source_type == ModelSourceType.LOCAL_DIFFUSERS:
        for path in loader_hub_paths(definition):
            ModelPathResolver.resolve(path, local_files_only=False)


def prefetch_in_child(definition_id: str, env: dict[str, str], python: str) -> None:
    """``prefetch_here`` in a child of ``python`` under ``env``, bounded by ``DOWNLOAD_TIMEOUT_S``."""
    argv = [python, "-m", "app.core.sampler_compare.facts", "--prefetch", definition_id]
    try:
        code, _, stderr = run_child(argv, env, DOWNLOAD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"the weight download did not finish within {DOWNLOAD_TIMEOUT_S / 3600:g} h") from None
    if code != 0:
        raise RuntimeError(f"the weight download exited with {code}: {stderr[-1500:]}")


def rebuild_masks_here(dataset: str, opacity: float) -> dict:
    """The masked pictures of ``dataset`` made again the way the trainer makes them for a job with
    ``recreate_masks`` (``pipeline_data``: ``MaskingService().mass_apply(path, opacity,
    overwrite=True)``); -> its counts and the stems of every mask whose picture was not written."""
    from app.core.masking.masking_service import MaskingService

    masks, masked = Path(dataset) / "masks", Path(dataset) / "masked"
    stems = sorted(os.path.splitext(name)[0] for name in os.listdir(masks)
                   if name.lower().endswith(".png")) if masks.is_dir() else []

    def written(stem: str) -> int | None:
        path = masked / f"{stem}.jpg"
        return path.stat().st_mtime_ns if path.exists() else None

    before = {stem: written(stem) for stem in stems}
    result = MaskingService().mass_apply(str(dataset), opacity, overwrite=True)
    failed = [stem for stem in stems if written(stem) is None or written(stem) == before[stem]]
    return {"applied": result["applied"], "skipped": result["skipped"], "failed": failed}


def rebuild_masks_in_child(dataset: Path, opacity: float, env: dict[str, str], python: str) -> None:
    """``rebuild_masks_here`` in a child of ``python`` under ``env``, bounded by ``MASKS_TIMEOUT_S``;
    raises naming every mask whose picture could not be rebuilt."""
    argv = [python, "-m", "app.core.sampler_compare.facts", "--masks", str(dataset), repr(float(opacity))]
    try:
        code, stdout, stderr = run_child(argv, env, MASKS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"rebuilding the masked pictures did not finish within {MASKS_TIMEOUT_S} s") from None
    if code != 0:
        raise RuntimeError(f"rebuilding the masked pictures exited with {code}: {stderr[-1500:]}")
    result = next((json.loads(line[len(MARKER):]) for line in reversed(stdout.splitlines())
                   if line.startswith(MARKER)), None)
    if result is None:
        raise RuntimeError(f"rebuilding the masked pictures printed no result: {stdout[-1500:]}")
    if result["failed"] or result["skipped"]:
        names = ", ".join(result["failed"]) or f"{result['skipped']} skipped"
        raise RuntimeError(f"the masked pictures of {names} could not be rebuilt from their masks; nothing "
                           "trained: fix those masks, or set recreate_masks off in the reference job")


def main(argv: list[str]) -> int:
    if argv[:1] == ["--prefetch"]:
        prefetch_here(argv[1])
        return 0
    if argv[:1] == ["--masks"]:
        sys.stdout.write(MARKER + json.dumps(rebuild_masks_here(argv[1], float(argv[2]))) + "\n")
        sys.stdout.flush()
        return 0
    definition_id, arms = argv
    sys.stdout.write(MARKER + json.dumps(facts_here(definition_id, json.loads(arms))) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
