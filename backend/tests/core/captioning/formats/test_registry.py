from app.core.captioning.formats import (
    get_caption_format, PlainFormat, Ideogram4Format,
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
