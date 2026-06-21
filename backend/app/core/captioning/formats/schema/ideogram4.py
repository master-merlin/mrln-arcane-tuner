"""Ideogram 4 structured-caption schema — the single source of truth.

Port of ai-toolkit's toolkit/ideogram_caption.py contract: the captioner,
refine pass, dataloader, and the frontend editor all agree on this shape.
Pure functions only — no I/O.
"""

from __future__ import annotations

import json
import re

PHOTO_MEDIUM = "photograph"
CANONICAL_MEDIUMS = [
    "photograph",
    "illustration",
    "3d_render",
    "painting",
    "graphic_design",
]
MAX_IMAGE_PALETTE = 16
MAX_ELEMENT_PALETTE = 5
BBOX_MAX = 1000

_MEDIUM_ALIASES = {
    "photo": "photograph",
    "photograph": "photograph",
    "photography": "photograph",
    "illustration": "illustration",
    "drawing": "illustration",
    "3d": "3d_render",
    "3d_render": "3d_render",
    "3drender": "3d_render",
    "render": "3d_render",
    "cgi": "3d_render",
    "painting": "painting",
    "oil": "painting",
    "oil painting": "painting",
    "graphic_design": "graphic_design",
    "graphic design": "graphic_design",
}

_TOP_ORDER = [
    "high_level_description",
    "style_description",
    "compositional_deconstruction",
]
_STYLE_PHOTO_ORDER = ["aesthetics", "lighting", "photo", "medium", "color_palette"]
_STYLE_ART_ORDER = ["aesthetics", "lighting", "medium", "art_style", "color_palette"]
_ELEM_OBJ_ORDER = ["type", "bbox", "desc", "color_palette"]
_ELEM_TEXT_ORDER = ["type", "bbox", "text", "desc", "color_palette"]


def canon_medium(m: str) -> str:
    """Canonicalize a medium string. Unrecognized mediums are snake-cased and
    passed through unchanged -- custom mediums are allowed but discouraged."""
    key = (m or "").strip().lower().rstrip(".").strip()
    if key in _MEDIUM_ALIASES:
        return _MEDIUM_ALIASES[key]
    key2 = key.replace(" ", "_")
    return _MEDIUM_ALIASES.get(key2, key2 or PHOTO_MEDIUM)


def normalize_color(c: str) -> str | None:
    if not isinstance(c, str):
        return None
    s = c.strip().lstrip("#")
    if len(s) == 3 and re.fullmatch(r"[0-9a-fA-F]{3}", s):
        s = "".join(ch * 2 for ch in s)
    if re.fullmatch(r"[0-9a-fA-F]{6}", s):
        return "#" + s.upper()
    return None


def _palette(colors, cap: int) -> list[str]:
    out: list[str] = []
    for c in colors or []:
        nc = normalize_color(c)
        if nc and nc not in out:
            out.append(nc)
        if len(out) >= cap:
            break
    return out


def swap_bbox_xy(bbox: list) -> list:
    """[x1,y1,x2,y2] -> [y1,x1,y2,x2] (and vice-versa).

    X-first <-> y-first conversion helper for callers that hold x-first data,
    e.g. the frontend overlay (pixel-space x-first) or importers of
    ai-toolkit-style captions (x-first).  Call this BEFORE normalize() when
    your source bboxes are x-first.
    """
    if not bbox or len(bbox) != 4:
        return bbox
    a, b, c, d = bbox
    return [b, a, d, c]


def _clamp_bbox(bbox) -> list | None:
    # Incoming bboxes are assumed y-first [y_min, x_min, y_max, x_max].
    # The generation prompt instructs the VLM to emit y-first, so normalize()
    # only clamps values to [0, BBOX_MAX] and rounds — it does NOT swap axes.
    # Use swap_bbox_xy() to convert x-first data before calling normalize().
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        vals = [max(0, min(BBOX_MAX, int(round(float(v))))) for v in bbox]
    except (TypeError, ValueError):
        return None
    return vals


def _normalize_style(style: dict) -> dict:
    style = dict(style or {})
    medium = canon_medium(style.get("medium", PHOTO_MEDIUM))
    render = (
        style.get("photo") if style.get("photo") is not None else style.get("art_style")
    )
    out: dict = {
        "aesthetics": str(style.get("aesthetics", "") or ""),
        "lighting": str(style.get("lighting", "") or ""),
    }
    if medium == PHOTO_MEDIUM:
        out["photo"] = str(render or "")
        out["medium"] = medium
        ordered = {
            k: out[k] for k in _STYLE_PHOTO_ORDER if k in out and k != "color_palette"
        }
    else:
        out["medium"] = medium
        out["art_style"] = str(render or "")
        ordered = {
            k: out[k] for k in _STYLE_ART_ORDER if k in out and k != "color_palette"
        }
    ordered["color_palette"] = _palette(style.get("color_palette"), MAX_IMAGE_PALETTE)
    return ordered


def _normalize_element(el: dict) -> dict:
    el = dict(el or {})
    etype = "text" if el.get("type") == "text" else "obj"
    out: dict = {"type": etype}
    bbox = _clamp_bbox(el.get("bbox"))
    if bbox is not None:
        out["bbox"] = bbox
    if etype == "text":
        out["text"] = str(el.get("text", "") or "")
    out["desc"] = str(el.get("desc", "") or "")
    out["color_palette"] = _palette(el.get("color_palette"), MAX_ELEMENT_PALETTE)
    order = _ELEM_TEXT_ORDER if etype == "text" else _ELEM_OBJ_ORDER
    return {k: out[k] for k in order if k in out}


def _normalize_deconstruction(dec: dict) -> dict:
    dec = dict(dec or {})
    elements = dec.get("elements")
    if not isinstance(elements, list):
        elements = []
    return {
        "background": str(dec.get("background", "") or ""),
        "elements": [_normalize_element(e) for e in elements if isinstance(e, dict)],
    }


def migrate(data: dict) -> dict:
    """Best-effort old→new shape. Idempotent; safe on already-new docs."""
    return data  # normalize() subsumes the documented old-format differences


def normalize(data: dict) -> dict:
    data = migrate(dict(data or {}))
    out = {
        "high_level_description": str(data.get("high_level_description", "") or ""),
        "style_description": _normalize_style(data.get("style_description") or {}),
        "compositional_deconstruction": _normalize_deconstruction(
            data.get("compositional_deconstruction") or {}
        ),
    }
    return {k: out[k] for k in _TOP_ORDER}


def serialize(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _first_json_object(s: str) -> str | None:
    """Return the substring of the FIRST balanced ``{...}`` object in *s*, or None.

    String-aware brace matching (ignores braces inside JSON strings and handles
    escapes), so a valid JSON object followed by trailing prose — a common VLM
    failure mode when a model keeps chatting after the caption — is extracted
    cleanly instead of being widened to a stray ``}`` later in the text."""
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def parse(text: str) -> dict | None:
    if not text:
        return None
    s = text.strip()
    # Strip ```json fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, re.DOTALL)
    if fence:
        s = fence.group(1)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Fallback: the first balanced {...} object (tolerates trailing prose).
    block = _first_json_object(s)
    if block is not None:
        try:
            obj = json.loads(block)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def detect(text: str) -> bool:
    obj = parse(text)
    return bool(obj) and "compositional_deconstruction" in obj


def skeleton(raw_text: str) -> dict:
    return normalize(
        {
            "high_level_description": (raw_text or "").strip(),
            "style_description": {},
            "compositional_deconstruction": {"background": "", "elements": []},
        }
    )
