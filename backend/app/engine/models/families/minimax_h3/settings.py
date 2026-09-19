"""minimax_h3 — the ONE settings resolver (RULE-21; plan row 1.0).

H3 is the first family with two COUPLED sigma schedules, a joint audio loss
and a CFG-augmentation term, each of which can come from the run config OR the
definition's ``architecture_params``. Two readers of the same key disagree
eventually, so every consumer — the schedule, ``compute_loss``, the
CFG-augmentation branch, the step-0 banner, the trainer — reads the
:class:`H3EffectiveSettings` produced here and never ``config`` or
``architecture_params`` directly. ``test_minimax_h3_settings.py`` pins that
with a source guard over the family package.

Resolution order per key (plan § Configuration contract):

* ``audio_loss_weight``: ``train_audio`` off -> ``0.0`` (``train_audio_off``);
  config value present and ``!= 1.0`` -> ``config``; definition
  ``audio.loss_weight`` -> ``definition``; else ``1.0`` (``schema_default``).
  The schema default ``1.0`` is indistinguishable from an explicit ``1.0`` on
  this family (plan RES-1 / REQUEST-16); it means "the definition's value".
* ``sigma_shift_video`` / ``sigma_shift_audio``: config not ``None`` ->
  ``config``; definition ``video.sigma_shift`` / ``audio.sigma_shift`` ->
  ``definition``; absent or ``<= 0`` -> ``ValueError`` naming the key.
* ``cfg_augment_scale``: config not ``None`` -> ``config`` (``1.0`` = OFF);
  definition ``cfg_augment.scale`` -> ``definition``; else ``1.0``
  (``family_default``).
* ``train_audio``: config -> ``config``; definition ``defaults`` ->
  ``definition``; else ``False`` (``schema_default``).

Config is read with ``Mapping.get`` / ``getattr(config, key, None)`` so the
plain job-config dict and the schema object (row 3.3) both route through.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from app.engine.core.definitions import ModelDefinition

SETTINGS_VERSION = 1

# The schema default of `audio_loss_weight` (base.py BaseTrainingConfig); an
# explicit value equal to it carries no information on this family.
_AUDIO_LOSS_WEIGHT_SCHEMA_DEFAULT = 1.0
_TRAIN_AUDIO_SCHEMA_DEFAULT = False
_CFG_AUGMENT_FAMILY_DEFAULT = 1.0  # 1.0 = CFG augmentation OFF

_AUDIO_LOSS_WEIGHT_RANGE = (0.0, 10.0)


@dataclass(frozen=True)
class H3EffectiveSettings:
    """The resolved, validated settings every H3 consumer reads.

    ``sources`` records, per field, which producer won (``config`` /
    ``definition`` / ``schema_default`` / ``family_default`` /
    ``train_audio_off``) so the step-0 banner can print provenance.
    """

    audio_loss_weight: float
    sigma_shift_video: float
    sigma_shift_audio: float
    cfg_augment_scale: float
    train_audio: bool
    sources: dict[str, str] = field(default_factory=dict)
    version: int = SETTINGS_VERSION


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def _float_or_raise(value: Any, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key}: not a number ({value!r})") from exc


def _resolve_sigma_shift(
    config: Any, arch: Mapping[str, Any], config_key: str, def_key: str
) -> tuple[float, str]:
    raw = _config_value(config, config_key)
    if raw is not None:
        value, source = _float_or_raise(raw, config_key), "config"
    elif arch.get(def_key) is not None:
        value, source = _float_or_raise(arch[def_key], def_key), "definition"
    else:
        raise ValueError(
            f"{config_key}: the definition declares no `{def_key}` and the "
            "run config does not set it — H3 needs both sigma shifts"
        )
    if value <= 0:
        raise ValueError(f"{config_key}: must be > 0, got {value} (source={source})")
    return value, source


def _resolve_train_audio(config: Any, definition: ModelDefinition) -> tuple[bool, str]:
    raw = _config_value(config, "train_audio")
    if raw is not None:
        return bool(raw), "config"
    if definition.defaults.get("train_audio") is not None:
        return bool(definition.defaults["train_audio"]), "definition"
    return _TRAIN_AUDIO_SCHEMA_DEFAULT, "schema_default"


def _resolve_audio_loss_weight(
    config: Any, arch: Mapping[str, Any], train_audio: bool
) -> tuple[float, str]:
    if not train_audio:
        return 0.0, "train_audio_off"
    raw = _config_value(config, "audio_loss_weight")
    if raw is not None and float(raw) != _AUDIO_LOSS_WEIGHT_SCHEMA_DEFAULT:
        value, source = _float_or_raise(raw, "audio_loss_weight"), "config"
    elif arch.get("audio.loss_weight") is not None:
        value, source = (
            _float_or_raise(arch["audio.loss_weight"], "audio.loss_weight"),
            "definition",
        )
    else:
        value, source = _AUDIO_LOSS_WEIGHT_SCHEMA_DEFAULT, "schema_default"
    lo, hi = _AUDIO_LOSS_WEIGHT_RANGE
    if not lo <= value <= hi:
        raise ValueError(
            f"audio_loss_weight: must be within [{lo}, {hi}], got {value} (source={source})"
        )
    return value, source


def _resolve_cfg_augment_scale(config: Any, arch: Mapping[str, Any]) -> tuple[float, str]:
    raw = _config_value(config, "cfg_augment_scale")
    if raw is not None:
        value, source = _float_or_raise(raw, "cfg_augment_scale"), "config"
    elif arch.get("cfg_augment.scale") is not None:
        value, source = (
            _float_or_raise(arch["cfg_augment.scale"], "cfg_augment.scale"),
            "definition",
        )
    else:
        value, source = _CFG_AUGMENT_FAMILY_DEFAULT, "family_default"
    if value < 1.0:
        raise ValueError(
            f"cfg_augment_scale: must be >= 1.0 (1.0 = off), got {value} (source={source})"
        )
    return value, source


def resolve_h3_settings(definition: ModelDefinition, config: Any) -> H3EffectiveSettings:
    """Resolve the effective H3 settings from ``definition`` + ``config``.

    ``config`` is the run-config mapping (or an attribute object); ``None`` /
    absent means "use the definition's value". Raises ``ValueError`` naming the
    key on a missing sigma shift or an out-of-range value.
    """
    arch = definition.architecture_params
    sources: dict[str, str] = {}

    train_audio, sources["train_audio"] = _resolve_train_audio(config, definition)
    audio_loss_weight, sources["audio_loss_weight"] = _resolve_audio_loss_weight(
        config, arch, train_audio
    )
    sigma_shift_video, sources["sigma_shift_video"] = _resolve_sigma_shift(
        config, arch, "sigma_shift_video", "video.sigma_shift"
    )
    sigma_shift_audio, sources["sigma_shift_audio"] = _resolve_sigma_shift(
        config, arch, "sigma_shift_audio", "audio.sigma_shift"
    )
    cfg_augment_scale, sources["cfg_augment_scale"] = _resolve_cfg_augment_scale(
        config, arch
    )

    return H3EffectiveSettings(
        audio_loss_weight=audio_loss_weight,
        sigma_shift_video=sigma_shift_video,
        sigma_shift_audio=sigma_shift_audio,
        cfg_augment_scale=cfg_augment_scale,
        train_audio=train_audio,
        sources=sources,
    )
