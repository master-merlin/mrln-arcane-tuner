"""Config Harvester — generically reads HuggingFace component config.json files
and stores ALL parameters in dot-namespaced form for YAML definitions.

Scans a model root directory for component subdirectories (transformer, vae,
text_encoder, scheduler, etc.), reads their config.json files, and returns
a unified dict of ``namespace.key`` parameters.

The YAML ``architecture_params`` should be a faithful mirror of the HF repo's
config.json files.  Consumers pick what they need; the harvester does NOT
cherry-pick or inject defaults.

Works with any HF-structured model repo (SDXL, Flux1, Flux2, future families).
"""

import json
import os
from typing import Any
import structlog

logger = structlog.get_logger(__name__)


# ── Component definitions ─────────────────────────────────────────────
# (namespace_prefix, [directory_candidates])
# Order doesn't matter — every component is harvested independently.

COMPONENTS: list[tuple[str, list[str]]] = [
    ("transformer", ["transformer", "unet"]),
    ("te",          ["text_encoder", "text_encoder_1"]),
    ("te2",         ["text_encoder_2"]),
    ("scheduler",   ["scheduler"]),
    ("vae",         ["vae", "ae"]),
]

# Keys to skip during harvesting (internal HF/Diffusers metadata)
_SKIP_KEYS: set[str] = {
    "_diffusers_version",
    "_name_or_path",
    "transformers_version",
}


def _find_config_file(root_path: str, dir_candidates: list[str]) -> str | None:
    """Find a config.json (or scheduler_config.json) in candidate subdirectories.

    Args:
        root_path: Model root directory.
        dir_candidates: List of subdirectory names to check.

    Returns:
        Absolute path to the config file, or None.
    """
    for dirname in dir_candidates:
        dirpath = os.path.join(root_path, dirname)
        if not os.path.isdir(dirpath):
            continue
        for fname in ["config.json", "scheduler_config.json"]:
            fpath = os.path.join(dirpath, fname)
            if os.path.isfile(fpath):
                return fpath
    return None


def _flatten_dict(
    data: dict[str, Any],
    prefix: str,
    result: dict[str, Any],
) -> None:
    """Recursively flatten a nested dict into dot-separated keys.

    Example::

        {"text_config": {"hidden_size": 4096}}
        → {"te.text_config.hidden_size": 4096}

    Also stores the nested dict itself so consumers can access either form::

        te.text_config = {"hidden_size": 4096}
        te.text_config.hidden_size = 4096

    Args:
        data: The dict to flatten.
        prefix: Current key prefix (e.g. ``"te.text_config"``).
        result: Output dict to populate.
    """
    for key, value in data.items():
        if key in _SKIP_KEYS:
            continue
        full_key = f"{prefix}.{key}"
        result[full_key] = value
        if isinstance(value, dict):
            _flatten_dict(value, full_key, result)


def _derive_fields(result: dict[str, Any]) -> None:
    """Compute derived fields from harvested values.

    These are cross-component computations that cannot be read from a
    single config.json.  The derived keys use the ``te.`` namespace for
    text-encoder-related derivations.

    Currently derived:
        - ``transformer.hidden_size`` from heads × head_dim
        - ``te.concat_layers`` from te.hidden_size / transformer.joint_attention_dim
    """
    # hidden_size = num_heads × head_dim (if not already present)
    nh = result.get("transformer.num_attention_heads")
    hd = result.get("transformer.attention_head_dim")
    if (
        "transformer.hidden_size" not in result
        and isinstance(nh, int)
        and isinstance(hd, int)
    ):
        result["transformer.hidden_size"] = nh * hd

    # te_concat_layers: context_dim / te_hidden_size
    te_hidden = result.get("te.text_config.hidden_size") or result.get("te.hidden_size")
    context_dim = result.get("transformer.joint_attention_dim")
    if (
        te_hidden
        and context_dim
        and isinstance(te_hidden, int)
        and isinstance(context_dim, int)
        and context_dim % te_hidden == 0
    ):
        result["te.concat_layers"] = context_dim // te_hidden


def harvest(root_path: str) -> dict[str, Any]:
    """Scan a model root directory for component config.json files and
    extract ALL parameters with dot-namespace notation.

    Args:
        root_path: Absolute path to the model directory (e.g., HuggingFace cache dir).

    Returns:
        Dict of ``namespace.key`` → value pairs.  Every key from every
        component config.json is included (except internal HF metadata).
        Derived cross-component fields are appended.
    """
    if not root_path or not os.path.isdir(root_path):
        logger.debug("config_harvest_skip_no_dir", root_path=root_path)
        return {}

    result: dict[str, Any] = {}
    configs_found = 0

    for prefix, dir_candidates in COMPONENTS:
        config_path = _find_config_file(root_path, dir_candidates)
        if not config_path:
            continue

        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "config_harvest_parse_error", path=config_path, error=str(e)
            )
            continue

        configs_found += 1

        # Store every key with namespace prefix
        for key, value in cfg.items():
            if key in _SKIP_KEYS:
                continue
            namespaced_key = f"{prefix}.{key}"
            result[namespaced_key] = value
            # Recursively flatten nested dicts
            if isinstance(value, dict):
                _flatten_dict(value, namespaced_key, result)

    # Derived cross-component fields
    _derive_fields(result)

    logger.info(
        "config_harvest_complete",
        root_path=root_path,
        configs_found=configs_found,
        params_extracted=len(result),
        namespaces=sorted({k.split(".")[0] for k in result}),
    )

    return result
