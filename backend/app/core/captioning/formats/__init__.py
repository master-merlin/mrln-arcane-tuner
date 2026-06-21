from __future__ import annotations
from app.core.captioning.formats.base import CaptionFormat
from app.core.captioning.formats.plain import PlainFormat
from app.core.captioning.formats.ideogram4 import Ideogram4Format

_PLAIN = PlainFormat()
_REGISTRY: dict[str, CaptionFormat] = {
    "ideogram4_json": Ideogram4Format(),
}
_FAMILY_FORMAT_ID: dict[str, str] = {
    "ideogram4": "ideogram4_json",
}


def get_caption_format(family: str) -> CaptionFormat:
    """Resolve a model family to its CaptionFormat. Unknown → PlainFormat.

    Reads the family class's ``caption_format`` attribute when registered, then
    falls back to the static family→id map (keeps the registry resilient if a
    family module is not imported yet)."""
    fmt_id = _FAMILY_FORMAT_ID.get(family)
    if fmt_id is None:
        try:
            from app.engine.models.registry import registry

            fam = (
                registry.get_family(family) if hasattr(registry, "get_family") else None
            )
            fmt_id = getattr(fam, "caption_format", None) if fam else None
        except Exception:
            fmt_id = None
    return _REGISTRY.get(fmt_id or "", _PLAIN)


def get_caption_format_for_definition(definition_id: str) -> CaptionFormat:
    from app.engine.models.registry import registry

    defn = registry.get_definition(definition_id)
    if defn is None:
        return _PLAIN
    return get_caption_format(defn.family)


__all__ = [
    "CaptionFormat",
    "PlainFormat",
    "Ideogram4Format",
    "get_caption_format",
    "get_caption_format_for_definition",
]
