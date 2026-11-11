"""MiniMax-H3-Image: what the family reads from its definition, in one place.

The model is distilled to a fixed 6-step schedule with one output head per step
(``pdd_proj_out[0..5]``, the checkpoint's ``num_pdd_heads: 6``). The head grid is
the model's sampling sigmas without the terminal 0.0:

    1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132

(``modular_model_index.json`` ``sample_sigmas`` of MiniMaxAI/MiniMax-H3-Image at
00dd081e). The definition states it once, as ``timestep.fixed_sigmas``: the same
list is the family's default training draw (``timestep.default_mode:
fixed_sigmas``, ostris/ai-toolkit's rule "each head only ever sees its own
sigma") and the grid a sigma's head is looked up on. Any other draw uses the
head of the nearest grid sigma (DiffSynth-Studio's rule "any other schedule
picks the nearest head").

Every reader of those two keys goes through this module.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch

from app.engine.core.definitions import ModelDefinition

DEFAULT_MODE_KEY = "timestep.default_mode"
FIXED_SIGMAS_KEY = "timestep.fixed_sigmas"


def head_sigmas(definition: ModelDefinition) -> list[float]:
    """The head grid: one sigma per output head, head ``i`` at index ``i``.

    Raises ``ValueError`` naming the key when the definition does not state it,
    or states a list ``validate_fixed_sigmas`` refuses."""
    from app.engine.strategies.timestep_sampling import validate_fixed_sigmas

    arch = definition.architecture_params or {}
    if arch.get(FIXED_SIGMAS_KEY) is None:
        raise ValueError(
            f"minimax_h3_image {definition.id}: architecture_params[{FIXED_SIGMAS_KEY!r}] is missing; "
            "it is the model's sampling grid, one sigma per output head"
        )
    return validate_fixed_sigmas(arch[FIXED_SIGMAS_KEY])


def head_for_sigma(sigma: float | torch.Tensor, grid: list[float]) -> int:
    """The index of the head trained for the grid sigma nearest to ``sigma``
    (a grid value maps to its own index; a tie goes to the lower index)."""
    value = float(sigma.reshape(-1)[0]) if isinstance(sigma, torch.Tensor) else float(sigma)
    distances = [abs(g - value) for g in grid]
    return distances.index(min(distances))


def apply_timestep_default(definition: ModelDefinition, config: MutableMapping[str, Any]) -> tuple[str, str]:
    """Give a config without ``timestep_sampling`` the family default, and a
    ``fixed_sigmas`` draw without a list the definition's list.

    ``setdefault``-style: a mode the config names is never changed, and a list
    it carries is never replaced. Returns ``(mode, source)``, the source being
    ``"config"`` or ``"definition"``. A ``fixed_sigmas`` draw is validated here,
    before any weight loads."""
    from app.engine.strategies.timestep_sampling import validate_fixed_sigmas

    arch = definition.architecture_params or {}
    if config.get("timestep_sampling"):
        source = "config"
    else:
        default = arch.get(DEFAULT_MODE_KEY)
        if not default:
            raise ValueError(
                f"minimax_h3_image {definition.id}: the run config names no timestep_sampling and "
                f"architecture_params[{DEFAULT_MODE_KEY!r}] is missing"
            )
        config["timestep_sampling"] = str(default)
        source = "definition"
    mode = str(config["timestep_sampling"])
    if mode == "fixed_sigmas":
        if not config.get("fixed_sigmas"):
            config["fixed_sigmas"] = head_sigmas(definition)
        config["fixed_sigmas"] = validate_fixed_sigmas(config["fixed_sigmas"])
    return mode, source


def architecture_int(definition: ModelDefinition, key: str) -> int:
    """An integer the definition must state (``te.hidden_state_tap_index``, ``te.max_length``)."""
    arch: Mapping[str, Any] = definition.architecture_params or {}
    if arch.get(key) is None:
        raise ValueError(f"minimax_h3_image {definition.id}: architecture_params[{key!r}] is missing")
    return int(arch[key])
