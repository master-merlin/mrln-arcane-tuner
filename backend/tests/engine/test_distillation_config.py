"""Distillation handling, the shared part: the definition block, the job fields and the
descriptor the Training form reads.

A definition that is distilled declares ``distillation: {step, guidance, guidance_scale,
training_adapter, default_handling}``; ``core/definitions.py`` validates it. The job carries
``distillation_handling``, ``distillation_guidance_scale`` and
``distillation_guidance_schedule``; an unset field means the definition's default, and a value
the definition does not allow is refused (``resolve_distillation``, which ``setup()`` calls
before any weight loads). ``resolve_capabilities`` hands the form the allowed handlings, the
default and the guidance targets. A definition without the block is not distilled: it gains
three hidden fields and nothing else.
"""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from app.engine.core.archetypes import resolve_capabilities
from app.engine.core.definitions import (
    DistillationSpec,
    ModelDefinition,
    ResolvedDistillation,
    resolve_distillation,
)
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.registry import registry

ADAPTER = {
    "repo": "someone/some_training_adapter",
    "revision": "e4e119caa391053286145e48ebbaf0e0c00293d3",
    "file": "adapter_v1.safetensors",
    "sha256": "45a5d60d7dfb008461d9546a6849d8d41bd00fe45849bf2d525a9db5ed68bb60",
}
FULL_BLOCK = {
    "step": True,
    "guidance": True,
    "guidance_scale": 3.5,
    "training_adapter": ADAPTER,
    "default_handling": "adapter",
}
FIELDS = ("distillation_handling", "distillation_guidance_scale", "distillation_guidance_schedule")


def _block(**changes) -> dict:
    block = copy.deepcopy(FULL_BLOCK)
    block.update(changes)
    return block


def _config(**fields) -> BaseTrainingConfig:
    return BaseTrainingConfig.model_validate({"datasets": [{"dataset_name": "d"}], **fields})


def _definition(block: dict | None, *, family: str = "sdxl") -> ModelDefinition:
    data = {"id": "test-distilled", "family": family, "name": "Test distilled"}
    if block is not None:
        data["distillation"] = block
    return ModelDefinition(**data)


@pytest.fixture(scope="module", autouse=True)
def _loaded_registry():
    registry.initialize()
    return registry


# ── The definition block ────────────────────────────────────────────────────


def test_a_definition_without_the_block_is_not_distilled():
    assert _definition(None).distillation is None


def test_the_full_block_validates():
    spec = _definition(FULL_BLOCK).distillation
    assert isinstance(spec, DistillationSpec)
    assert spec.training_adapter.repo == ADAPTER["repo"]
    assert spec.allowed_handlings() == ["adapter", "contrastive_guidance", "both", "none"]


@pytest.mark.parametrize(
    ("changes", "allowed"),
    [
        ({}, ["adapter", "contrastive_guidance", "both", "none"]),
        ({"training_adapter": None, "default_handling": "contrastive_guidance"}, ["contrastive_guidance", "none"]),
        ({"guidance": False, "guidance_scale": None}, ["adapter", "none"]),
        (
            {"guidance": False, "guidance_scale": None, "training_adapter": None, "default_handling": "none"},
            ["none"],
        ),
    ],
)
def test_the_allowed_handlings_follow_the_block(changes, allowed):
    assert _definition(_block(**changes)).distillation.allowed_handlings() == allowed


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"training_adapter": None}, "default_handling 'adapter' is not allowed"),
        ({"guidance": False, "guidance_scale": None, "default_handling": "both"}, "default_handling 'both'"),
        ({"guidance": False, "guidance_scale": None, "default_handling": "contrastive_guidance"}, "not allowed"),
        ({"guidance_scale": None}, "guidance_scale must be a number >= 1"),
        ({"guidance_scale": 0.5}, "guidance_scale must be a number >= 1"),
        ({"guidance": False}, "only meaningful with guidance: true"),
        ({"default_handling": "maybe"}, "default_handling"),
        ({"extra_key": 1}, "extra_key"),
    ],
)
def test_a_block_that_contradicts_itself_is_refused(changes, message):
    with pytest.raises(ValidationError, match=message):
        _definition(_block(**changes))


@pytest.mark.parametrize("missing", ["step", "guidance", "guidance_scale", "training_adapter", "default_handling"])
def test_every_key_of_the_block_is_required(missing):
    block = _block()
    del block[missing]
    with pytest.raises(ValidationError, match=missing):
        _definition(block)


@pytest.mark.parametrize(
    ("field", "value"),
    [("sha256", "45A5D60D"), ("sha256", "0" * 63), ("revision", ""), ("repo", ""), ("file", "")],
)
def test_the_adapter_pin_is_checked(field, value):
    with pytest.raises(ValidationError, match=field):
        _definition(_block(training_adapter={**ADAPTER, field: value}))


def test_the_guidance_target_each_handling_starts_from():
    spec = _definition(FULL_BLOCK).distillation
    assert spec.default_guidance_scale("contrastive_guidance") == 3.5
    assert spec.default_guidance_scale("both") == 2.5  # 0.7 x 3.5 = 2.45, to 0.1
    assert spec.default_guidance_scale("adapter") is None
    assert spec.default_guidance_scale("none") is None
    assert _definition(_block(guidance_scale=4.0)).distillation.default_guidance_scale("both") == 2.8


def test_every_shipped_definition_without_a_block_loads_as_before():
    """The block is additive: the shipped definitions still load, and the ones that do not
    declare it are not distilled."""
    ids = registry.list_models()
    assert ids
    for definition in map(registry.get_definition, ids):
        spec = definition.distillation
        assert spec is None or isinstance(spec, DistillationSpec), definition.id
    assert registry.get_definition("sdxl_base_1.0").distillation is None


# ── The job fields ──────────────────────────────────────────────────────────


def test_the_job_fields_default_to_unset():
    config = _config()
    assert config.distillation_handling is None
    assert config.distillation_guidance_scale is None
    assert config.distillation_guidance_schedule is None


def test_the_job_fields_take_their_values():
    config = _config(
        distillation_handling="both",
        distillation_guidance_scale=2.5,
        distillation_guidance_schedule="sigma",
    )
    assert (config.distillation_handling, config.distillation_guidance_scale, config.distillation_guidance_schedule) == (
        "both",
        2.5,
        "sigma",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("distillation_handling", "sometimes"), ("distillation_guidance_schedule", "linear")],
)
def test_the_job_fields_refuse_unknown_values(field, value):
    with pytest.raises(ValidationError, match=field):
        _config(**{field: value})


def test_the_guidance_fields_show_only_with_guidance():
    props = BaseTrainingConfig.model_json_schema()["properties"]
    for field in ("distillation_guidance_scale", "distillation_guidance_schedule"):
        assert props[field]["depends_on"] == "distillation_handling:contrastive_guidance,both"
    assert props["distillation_handling"]["title"] == "Distillation handling"


# ── Resolving a job against its definition ──────────────────────────────────


def test_an_unset_handling_means_the_definitions_default():
    assert resolve_distillation(_definition(FULL_BLOCK), {}) == ResolvedDistillation("adapter", None, None)
    cg = _definition(_block(default_handling="contrastive_guidance"))
    assert resolve_distillation(cg, {"distillation_handling": None}) == ResolvedDistillation(
        "contrastive_guidance", 3.5, "constant"
    )


@pytest.mark.parametrize(
    ("handling", "expected"),
    [
        ("adapter", ResolvedDistillation("adapter", None, None)),
        ("contrastive_guidance", ResolvedDistillation("contrastive_guidance", 3.5, "constant")),
        ("both", ResolvedDistillation("both", 2.5, "constant")),
        ("none", ResolvedDistillation("none", None, None)),
    ],
)
def test_an_allowed_handling_is_kept_with_its_defaults(handling, expected):
    assert resolve_distillation(_definition(FULL_BLOCK), {"distillation_handling": handling}) == expected


def test_explicit_guidance_settings_are_kept():
    resolved = resolve_distillation(
        _definition(FULL_BLOCK),
        {
            "distillation_handling": "contrastive_guidance",
            "distillation_guidance_scale": "2.0",
            "distillation_guidance_schedule": "sigma",
        },
    )
    assert resolved == ResolvedDistillation("contrastive_guidance", 2.0, "sigma")
    assert resolved.uses_contrastive_guidance and not resolved.uses_adapter


def test_guidance_settings_are_ignored_without_guidance():
    resolved = resolve_distillation(
        _definition(FULL_BLOCK),
        {"distillation_handling": "adapter", "distillation_guidance_scale": 9.0, "distillation_guidance_schedule": "x"},
    )
    assert resolved == ResolvedDistillation("adapter", None, None)
    assert resolved.uses_adapter and not resolved.uses_contrastive_guidance


@pytest.mark.parametrize(
    ("block_changes", "handling"),
    [
        ({"training_adapter": None, "default_handling": "contrastive_guidance"}, "adapter"),
        ({"training_adapter": None, "default_handling": "contrastive_guidance"}, "both"),
        ({"guidance": False, "guidance_scale": None}, "contrastive_guidance"),
        ({}, "sometimes"),
    ],
)
def test_a_handling_the_definition_does_not_allow_is_refused(block_changes, handling):
    with pytest.raises(ValueError, match=f"distillation_handling={handling!r} is not supported"):
        resolve_distillation(_definition(_block(**block_changes)), {"distillation_handling": handling})


@pytest.mark.parametrize("scale", [0.9, 0.0, -2.0, "many", float("nan"), float("inf")])
def test_a_guidance_target_that_is_not_a_number_of_at_least_1_is_refused(scale):
    with pytest.raises(ValueError, match="distillation_guidance_scale"):
        resolve_distillation(
            _definition(FULL_BLOCK),
            {"distillation_handling": "both", "distillation_guidance_scale": scale},
        )


def test_an_unknown_schedule_is_refused():
    with pytest.raises(ValueError, match="distillation_guidance_schedule='linear'"):
        resolve_distillation(
            _definition(FULL_BLOCK),
            {"distillation_handling": "both", "distillation_guidance_schedule": "linear"},
        )


def test_a_model_that_is_not_distilled_allows_only_none():
    plain = _definition(None)
    assert resolve_distillation(plain, {}) == ResolvedDistillation("none", None, None)
    assert resolve_distillation(plain, {"distillation_handling": "none"}).handling == "none"
    with pytest.raises(ValueError, match="not distilled"):
        resolve_distillation(plain, {"distillation_handling": "adapter"})


def test_an_unset_schedule_is_constant_through_a_job_config_without_the_field():
    """The API route: a stored job config without the schedule (the field never set)."""
    config = _config(distillation_handling="contrastive_guidance").model_dump(
        exclude_unset=True
    )
    assert "distillation_guidance_schedule" not in config
    assert resolve_distillation(_definition(FULL_BLOCK), config).guidance_schedule == "constant"


# ── The descriptor the Training form reads ──────────────────────────────────


def test_a_definition_without_the_block_shows_nothing_new():
    caps = resolve_capabilities(registry.get_definition("sdxl_base_1.0"))
    assert caps["capabilities"]["supports_distillation_handling"] is False
    assert "distillation" not in caps["capabilities"]
    for field in FIELDS:
        assert caps["field_visibility"][field]["supported"] is False
        assert field not in caps["defaults"]


def test_a_distilled_definition_offers_its_handlings_with_their_defaults():
    caps = resolve_capabilities(_definition(FULL_BLOCK))
    assert caps["capabilities"]["supports_distillation_handling"] is True
    for field in FIELDS:
        assert caps["field_visibility"][field] == {"supported": True}
    assert caps["capabilities"]["distillation"] == {
        "step": True,
        "guidance": True,
        "handlings": ["adapter", "contrastive_guidance", "both", "none"],
        "default_handling": "adapter",
        "has_training_adapter": True,
        "guidance_scale_defaults": {"contrastive_guidance": 3.5, "both": 2.5},
        "guidance_schedules": ["constant", "sigma"],
        "default_guidance_schedule": "constant",
    }
    assert caps["defaults"]["distillation_handling"] == "adapter"
    assert caps["defaults"]["distillation_guidance_schedule"] == "constant"
    assert "distillation_guidance_scale" not in caps["defaults"]


def test_a_guidance_default_seeds_its_target():
    caps = resolve_capabilities(_definition(_block(default_handling="both")))
    assert caps["defaults"]["distillation_handling"] == "both"
    assert caps["defaults"]["distillation_guidance_scale"] == 2.5


def test_a_block_without_an_adapter_offers_no_adapter():
    caps = resolve_capabilities(_definition(_block(training_adapter=None, default_handling="contrastive_guidance")))
    descriptor = caps["capabilities"]["distillation"]
    assert descriptor["handlings"] == ["contrastive_guidance", "none"]
    assert descriptor["guidance_scale_defaults"] == {"contrastive_guidance": 3.5}
    assert caps["defaults"]["distillation_guidance_scale"] == 3.5


def test_every_shipped_definition_without_the_block_hides_the_fields():
    for definition in map(registry.get_definition, registry.list_models()):
        if definition.distillation is not None:
            continue
        caps = resolve_capabilities(definition)
        assert caps["capabilities"]["supports_distillation_handling"] is False, definition.id
        assert all(caps["field_visibility"][f]["supported"] is False for f in FIELDS), definition.id
