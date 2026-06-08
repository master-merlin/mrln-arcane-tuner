"""Unit tests for the pure template import-service helpers."""

from app.core.template import import_service as svc


def test_model_available_captioning_and_masking():
    assert svc.model_available("captioning", "qwen3-vl") is True
    assert svc.model_available("captioning", "nope") is False
    assert svc.model_available("masking", "sam3") is True
    assert svc.model_available("masking", "nope") is False
    # training has no model_id registry — always available (definition handles it)
    assert svc.model_available("training", "anything") is True


def test_validate_config_captioning_ok_and_bad():
    # A valid qwen3-vl config validates clean (extra keys ignored by Pydantic).
    assert svc.validate_config("captioning", "qwen3-vl", {"max_tokens": 256}) is None
    # An unknown model_id has no schema → no warning (availability is a separate check).
    assert svc.validate_config("captioning", "unknown", {"x": 1}) is None
    # A wrong-typed field yields a warning string.
    warning = svc.validate_config("captioning", "qwen3-vl", {"temperature": "hot"})
    assert warning is not None and "temperature" in warning


def test_validate_config_training_is_skipped():
    # Training configs legitimately omit datasets; never validated/blocked.
    assert svc.validate_config("training", None, {"anything": 1}) is None


def test_validate_carried_definition_ok_and_invalid():
    good = {"id": "x", "family": "flux2", "name": "X",
            "components": {"repo": {"path": "huggingface:o/r"}}}
    assert svc.validate_carried_definition(good) is None
    bad = {"name": "missing id and family"}
    assert svc.validate_carried_definition(bad) is not None
