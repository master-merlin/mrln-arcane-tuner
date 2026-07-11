"""W4-1 backward-compat rail: deleting dead config fields must not break the
loading of pre-existing DB configs / templates / exported bundles that still
carry the removed keys.

``BaseTrainingConfig`` uses Pydantic's default ``extra='ignore'`` policy, so an
unknown key is silently dropped rather than raising. These tests pin that
contract for the four W4-1 deletions specifically, and confirm the fifth recon
candidate (``radc_seqlen_influence``) is still a first-class field.
"""

from __future__ import annotations

import pytest

from app.engine.models.base import BaseTrainingConfig

# The four fields deleted in W4-1 (with representative legacy values).
DELETED_KEYS: dict[str, object] = {
    "quantization_strategy": "vram_safe",
    "resolution_strategy": "progressive",
    "boundary_ratio_override": 0.5,
    "still_resolutions": [512, 768],
}

# Retained: read by video_contract.validate_video_config, so NOT deleted.
RETAINED_KEY = "radc_seqlen_influence"


def _minimal(**overrides):
    base: dict[str, object] = {"datasets": [{"dataset_name": "demo"}]}
    base.update(overrides)
    return base


def test_deleted_keys_are_gone_from_the_model():
    fields = set(BaseTrainingConfig.model_fields)
    still_present = sorted(k for k in DELETED_KEYS if k in fields)
    assert not still_present, (
        f"W4-1 deleted fields still present on BaseTrainingConfig: {still_present}"
    )


def test_legacy_config_with_deleted_keys_loads_without_error():
    # A config blob straight out of an old DB row / exported template.
    legacy = _minimal(**DELETED_KEYS)
    cfg = BaseTrainingConfig.model_validate(legacy)  # must not raise
    assert cfg is not None


def test_deleted_keys_do_not_survive_round_trip():
    legacy = _minimal(**DELETED_KEYS)
    dumped = BaseTrainingConfig.model_validate(legacy).model_dump()
    leaked = sorted(k for k in DELETED_KEYS if k in dumped)
    assert not leaked, f"deleted keys leaked back into model_dump(): {leaked}"


def test_retained_radc_field_still_round_trips():
    cfg = BaseTrainingConfig.model_validate(_minimal(**{RETAINED_KEY: 0.3}))
    dumped = cfg.model_dump()
    assert RETAINED_KEY in dumped
    assert dumped[RETAINED_KEY] == pytest.approx(0.3)


def test_legacy_config_mixing_deleted_and_retained_keys_loads():
    legacy = _minimal(radc_seqlen_influence=0.3, **DELETED_KEYS)
    dumped = BaseTrainingConfig.model_validate(legacy).model_dump()
    assert RETAINED_KEY in dumped
    assert not any(k in dumped for k in DELETED_KEYS)
