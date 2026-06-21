from __future__ import annotations
from app.core.captioning.formats.base import CaptionFormat
from app.core.captioning.formats.schema import ideogram4 as ix

_GEN_PROMPT = """You are an expert image captioner for the Ideogram 4 model. \
Output ONLY a single JSON object (no prose, no markdown fences) with EXACTLY these top-level keys: \
"high_level_description" (string), "style_description" (object), and \
"compositional_deconstruction" (object, required).

style_description keys, in order: "aesthetics", "lighting", then for photographs \
"photo" then "medium"; for non-photographs "medium" then "art_style"; finally \
"color_palette". Use EXACTLY ONE of "photo" / "art_style" depending on the medium. \
"medium" must be one of: photograph, illustration, 3d_render, painting, graphic_design. \
"color_palette" is a list of up to 16 UPPERCASE #RRGGBB hex colors.

compositional_deconstruction has "background" (string) and "elements" (array). \
Each element has "type" ("obj" or "text"), an optional "bbox" as \
[x_min, y_min, x_max, y_max] (left, top, right, bottom) normalized 0-1000 with \
the origin at the top-left corner, a "desc", a "color_palette" of up to 5 \
UPPERCASE #RRGGBB colors, and "text" (the literal string) for text elements. \
Describe every salient element."""


class Ideogram4Format(CaptionFormat):
    id = "ideogram4_json"
    is_structured = True

    def build_generation_prompt(self, user_instructions: str | None = None) -> str:
        prompt = _GEN_PROMPT
        if user_instructions and user_instructions.strip():
            prompt += "\n\nADDITIONAL INSTRUCTIONS:\n" + user_instructions.strip()
        return prompt

    def generation_overrides(self) -> dict:
        # Ceiling only — the structured JSON for a rich scene can be long, so we
        # raise max_tokens. We deliberately do NOT set a min-token FLOOR: forcing
        # a minimum makes a model that has already emitted a complete JSON object
        # keep generating, which spirals into garbage appended after the JSON
        # (and is slow). Let the prompt drive length and EOS stop naturally.
        return {"max_tokens": 4096}

    def json_schema(self) -> dict | None:
        return {
            "type": "object",
            "required": ["high_level_description", "compositional_deconstruction"],
            "properties": {
                "high_level_description": {"type": "string"},
                "style_description": {"type": "object"},
                "compositional_deconstruction": {
                    "type": "object",
                    "required": ["background", "elements"],
                    "properties": {
                        "background": {"type": "string"},
                        "elements": {"type": "array"},
                    },
                },
            },
        }

    def detect(self, text: str) -> bool:
        return ix.detect(text)

    def parse_and_normalize(self, raw: str) -> dict:
        parsed = ix.parse(raw)
        if parsed is None:
            return ix.skeleton(raw)
        return ix.normalize(parsed)

    def ingest_generated(self, raw: str) -> dict:
        # Generation path: the captioner emits bboxes x-first
        # [x_min,y_min,x_max,y_max]; swap to canonical y-first BEFORE normalize.
        # (parse_and_normalize, used by refine/editor on already-y-first data,
        # deliberately does NOT swap.)
        parsed = ix.parse(raw)
        if parsed is None:
            return ix.skeleton(raw)
        return ix.normalize(ix.swap_element_bboxes(parsed))

    def serialize(self, data: dict) -> str:
        return ix.serialize(data)

    def build_refine_prompt(self, target, data: dict) -> str:
        """Schema-preserving refine instruction. `target`/`data` are accepted for
        interface symmetry; the current prompt is fixed (future: per-item context)."""
        return (
            "You are editing an Ideogram 4 structured JSON caption. Improve clarity "
            "and accuracy of the description fields while PRESERVING the exact JSON "
            "schema, key order, the single photo/art_style branch, bbox values, and "
            "the medium token. Normalize colors to UPPERCASE #RRGGBB. Return ONLY the "
            "JSON object, no prose, no markdown fences."
        )
