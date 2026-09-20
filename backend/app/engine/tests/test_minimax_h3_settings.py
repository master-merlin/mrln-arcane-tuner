"""minimax_h3 — the ONE settings resolver (plan row 1.0, RULE-21).

``families/minimax_h3/settings.py`` is the only place that reads the family's
run-config keys (``audio_loss_weight``, ``sigma_shift_video``,
``sigma_shift_audio``, ``cfg_augment_scale``, ``train_audio``) and the matching
``architecture_params`` keys (``audio.loss_weight``, ``video.sigma_shift``,
``audio.sigma_shift``, ``cfg_augment.scale``). Every consumer reads the
``H3EffectiveSettings`` it produces.

Three guards:

* ``settings_precedence_matrix`` — 4 keys x {omitted, explicit,
  explicit-equals-schema-default, definition-absent} x {plain mapping, the
  REAL ``BaseTrainingConfig`` object (row 3.3)}, each case asserting the
  resolved VALUE and its ``sources`` entry (plan § Configuration contract).
* ``settings_reject_out_of_range`` — validation raises ``ValueError`` naming
  the key.
* ``only_settings_reads_config_keys`` — a source scan over the family package
  with the positive control ``fixtures/h3_settings_control.py``, which MUST be
  flagged (an allowlist that quietly swallows the tree passes green).
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import pytest

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.minimax_h3.settings import (
    H3EffectiveSettings,
    resolve_h3_settings,
)

_TESTS_DIR = Path(__file__).resolve().parent
_FAMILY_DIR = _TESTS_DIR.parents[0] / "models" / "families" / "minimax_h3"
_CONTROL = _TESTS_DIR / "fixtures" / "h3_settings_control.py"

# The shipped t2va values (minimax_h3_t2va.yaml:66,73-74) plus the
# cfg_augment.scale row 3.3 adds; a fixture, not a YAML read, so this file
# tests precedence and not the YAML.
_ARCH_FULL: dict[str, Any] = {
    "audio.loss_weight": 0.1,
    "video.sigma_shift": 12.0,
    "audio.sigma_shift": 3.0,
    "cfg_augment.scale": 4.0,
}


def _definition(
    arch: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
) -> ModelDefinition:
    return ModelDefinition(
        id="minimax-h3-t2va",
        family="minimax_h3",
        name="fixture",
        defaults={"train_audio": True} if defaults is None else defaults,
        architecture_params=dict(_ARCH_FULL) if arch is None else arch,
    )


def _arch_without(key: str) -> dict[str, Any]:
    arch = dict(_ARCH_FULL)
    del arch[key]
    return arch


# (effective field, definition key, explicit config value, schema default)
_KEYS = [
    ("audio_loss_weight", "audio.loss_weight", 0.5, 1.0),
    ("sigma_shift_video", "video.sigma_shift", 7.0, None),
    ("sigma_shift_audio", "audio.sigma_shift", 2.0, None),
    ("cfg_augment_scale", "cfg_augment.scale", 2.0, None),
]

# What "definition-absent + config omitted" resolves to per key.
_DEFINITION_ABSENT: dict[str, tuple[Any, str] | type[Exception]] = {
    "audio_loss_weight": (1.0, "schema_default"),
    "sigma_shift_video": ValueError,
    "sigma_shift_audio": ValueError,
    "cfg_augment_scale": (1.0, "family_default"),
}


def _typed(config: dict[str, Any]) -> Any:
    """The REAL schema object (row 3.3: the config route is typed) — what the
    trainer holds once the job config has been validated."""
    from app.engine.models.base import BaseTrainingConfig

    return BaseTrainingConfig.model_validate(
        {**config, "datasets": [{"dataset_name": "ds"}]}
    )


@pytest.mark.parametrize("route", ["mapping", "schema"])
@pytest.mark.parametrize(
    "case", ["omitted", "explicit", "explicit_equals_schema_default", "definition_absent"]
)
@pytest.mark.parametrize("field,def_key,explicit,schema_default", _KEYS)
def test_settings_precedence_matrix(field, def_key, explicit, schema_default, case, route):
    definition = _definition()
    config: Any = {"train_audio": True}
    if case == "omitted":
        expected = (_ARCH_FULL[def_key], "definition")
    elif case == "explicit":
        config[field] = explicit
        expected = (explicit, "config")
    elif case == "explicit_equals_schema_default":
        # The schema default carries no information: on this family it means
        # "use the definition's value" (plan RES-1 for audio_loss_weight; None
        # for the NEW keys).
        config[field] = schema_default
        expected = (_ARCH_FULL[def_key], "definition")
    else:
        definition = _definition(arch=_arch_without(def_key))
        expected = _DEFINITION_ABSENT[field]

    if route == "schema":
        config = _typed(config)

    if isinstance(expected, type):
        with pytest.raises(expected, match=field):
            resolve_h3_settings(definition, config)
        return

    settings = resolve_h3_settings(definition, config)
    value, source = expected
    assert getattr(settings, field) == pytest.approx(value), (
        f"{field} [{case}]: explicit config ignored or wrong precedence"
    )
    assert settings.sources[field] == source, f"{field} [{case}]: source"


def test_train_audio_off_zeroes_the_audio_loss_weight():
    settings = resolve_h3_settings(_definition(), {"train_audio": False})
    assert settings.train_audio is False
    assert settings.audio_loss_weight == 0.0
    assert settings.sources["audio_loss_weight"] == "train_audio_off"


def test_train_audio_resolution_order():
    # config wins
    s = resolve_h3_settings(_definition(), {"train_audio": False})
    assert (s.train_audio, s.sources["train_audio"]) == (False, "config")
    # absent -> definition defaults
    s = resolve_h3_settings(_definition(), {})
    assert (s.train_audio, s.sources["train_audio"]) == (True, "definition")
    # absent everywhere -> the schema default (base.py train_audio = False)
    s = resolve_h3_settings(_definition(defaults={}), {})
    assert (s.train_audio, s.sources["train_audio"]) == (False, "schema_default")


def test_settings_accept_an_attribute_config_object():
    """Row 3.3 routes the schema object through; getattr must work too."""

    class Cfg:
        train_audio = True
        sigma_shift_video = 9.0

    s = resolve_h3_settings(_definition(), Cfg())
    assert (s.sigma_shift_video, s.sources["sigma_shift_video"]) == (9.0, "config")
    assert (s.sigma_shift_audio, s.sources["sigma_shift_audio"]) == (3.0, "definition")


def test_effective_settings_is_frozen_and_versioned():
    s = resolve_h3_settings(_definition(), {"train_audio": True})
    assert isinstance(s, H3EffectiveSettings)
    assert s.version == 1
    assert dataclasses.is_dataclass(s)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.sigma_shift_video = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "config,arch_patch,key",
    [
        ({"train_audio": True, "audio_loss_weight": 10.5}, {}, "audio_loss_weight"),
        ({"train_audio": True, "audio_loss_weight": -0.1}, {}, "audio_loss_weight"),
        ({"train_audio": True, "sigma_shift_video": 0.0}, {}, "sigma_shift_video"),
        ({"train_audio": True, "sigma_shift_audio": -3.0}, {}, "sigma_shift_audio"),
        ({"train_audio": True, "cfg_augment_scale": 0.5}, {}, "cfg_augment_scale"),
        ({"train_audio": True}, {"video.sigma_shift": 0.0}, "sigma_shift_video"),
        ({"train_audio": True}, {"audio.loss_weight": 11.0}, "audio_loss_weight"),
    ],
)
def test_settings_reject_out_of_range(config, arch_patch, key):
    arch = dict(_ARCH_FULL, **arch_patch)
    with pytest.raises(ValueError, match=key):
        resolve_h3_settings(_definition(arch=arch), config)


# ── Source guard ────────────────────────────────────────────────────────

_FORBIDDEN_KEYS = (
    "audio_loss_weight",
    "sigma_shift_video",
    "sigma_shift_audio",
    "cfg_augment_scale",
    "train_audio",
    "video.sigma_shift",
    "audio.sigma_shift",
    "audio.loss_weight",
    "cfg_augment.scale",
)
_FORBIDDEN_RE = re.compile(
    r"""["'](?:""" + "|".join(re.escape(k) for k in _FORBIDDEN_KEYS) + r""")["']"""
)


def _offending_lines(path: Path) -> list[int]:
    """Line numbers where a forbidden quoted key appears in CODE (comments
    stripped: the YAML-facing docstrings and comments may name the keys)."""
    out: list[int] = []
    in_docstring = False
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0]
        if line.count('"""') % 2 == 1:
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if _FORBIDDEN_RE.search(line):
            out.append(n)
    return out


def _scan(paths: list[Path]) -> dict[str, list[int]]:
    return {str(p): hits for p in paths if (hits := _offending_lines(p))}


def test_only_settings_reads_config_keys():
    family_files = sorted(
        p for p in _FAMILY_DIR.rglob("*.py") if p.name != "settings.py"
    )
    assert family_files, f"no family sources under {_FAMILY_DIR}"
    # Positive control: the guard must be able to see an offender at all.
    control_hits = _scan([_CONTROL])
    assert control_hits, f"control not flagged: {_CONTROL} — the guard is blind"
    offenders = _scan(family_files)
    assert not offenders, (
        "H3 config / architecture_params keys read outside settings.py "
        f"(RULE-21, plan row 1.0): {offenders}"
    )
