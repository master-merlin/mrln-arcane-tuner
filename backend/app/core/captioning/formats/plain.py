from __future__ import annotations
from app.core.captioning.formats.base import CaptionFormat


class PlainFormat(CaptionFormat):
    """Flat text/tags — today's behavior, exactly. Passthrough."""

    id = "plain"
    is_structured = False

    def parse_and_normalize(self, raw: str) -> dict:
        return {"text": raw}

    def serialize(self, data: dict) -> str:
        return data.get("text", "")
