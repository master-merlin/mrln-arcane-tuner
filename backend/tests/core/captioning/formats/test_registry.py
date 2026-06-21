from app.core.captioning.formats import (
    get_caption_format,
    PlainFormat,
    Ideogram4Format,
)


def test_unknown_family_returns_plain():
    fmt = get_caption_format("definitely-not-a-family")
    assert isinstance(fmt, PlainFormat)
    assert fmt.is_structured is False


def test_ideogram4_family_returns_structured_format():
    fmt = get_caption_format("ideogram4")
    assert isinstance(fmt, Ideogram4Format)
    assert fmt.id == "ideogram4_json"
    assert fmt.is_structured is True


def test_plain_passthrough_roundtrips_text():
    fmt = get_caption_format("flux1")
    raw = "a, b, c"
    data = fmt.parse_and_normalize(raw)
    assert fmt.serialize(data) == raw


def test_get_format_for_definition_returns_plain_when_registry_unavailable(monkeypatch):
    from app.core.captioning.formats import get_caption_format_for_definition
    import sys
    import types

    broken = types.ModuleType("app.engine.models.registry")

    class _R:
        def get_definition(self, _id):
            raise RuntimeError("boom")

    broken.registry = _R()
    monkeypatch.setitem(sys.modules, "app.engine.models.registry", broken)
    result = get_caption_format_for_definition("anything")
    assert isinstance(result, PlainFormat)
