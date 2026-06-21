from __future__ import annotations
from app.core.captioning.formats.base import CaptionFormat
from app.core.captioning.formats.plain import PlainFormat
from app.core.captioning.formats.ideogram4 import Ideogram4Format

_PLAIN = PlainFormat()
_REGISTRY: dict[str, CaptionFormat] = {
    "ideogram4_json": Ideogram4Format(),
}
# ``_FAMILY_FORMAT_ID`` is the authoritative family->format registration point.
# The ``caption_format`` class attribute on ModelFamily subclasses is the
# primary source consulted first (via ``ModelRegistry.get_family_class``);
# this static map acts as the resilience fallback when the registry has not
# yet discovered the family module (e.g. during isolated unit tests).
_FAMILY_FORMAT_ID: dict[str, str] = {
    "ideogram4": "ideogram4_json",
}


def get_caption_format(family: str) -> CaptionFormat:
    """Resolve a model family to its CaptionFormat. Unknown -> PlainFormat.

    Consults the family class's ``caption_format`` attribute first (via
    ``ModelRegistry.get_family_class``), then falls back to the static
    ``_FAMILY_FORMAT_ID`` map.  Never raises -- unknown families return
    PlainFormat."""
    # Primary path: read caption_format from the registered family class.
    try:
        from app.engine.models.registry import registry

        fam_cls = registry.get_family_class(family)
        fmt_id = getattr(fam_cls, "caption_format", None)
    except Exception:
        fmt_id = None

    # Fallback: static map (covers the case where the family module is not yet
    # imported into the registry, e.g. in isolated unit tests).
    if fmt_id is None:
        fmt_id = _FAMILY_FORMAT_ID.get(family)

    return _REGISTRY.get(fmt_id or "", _PLAIN)


def get_caption_format_for_definition(definition_id: str) -> CaptionFormat:
    try:
        from app.engine.models.registry import registry

        defn = registry.get_definition(definition_id)
        if defn is None:
            return _PLAIN
        return get_caption_format(defn.family)
    except Exception:
        return _PLAIN


def apply_generation_seam(
    params: dict, caption_format: CaptionFormat, model_id: str
) -> None:
    """Mutate *params* in place to drive structured generation for *caption_format*.

    No-op when the format is not structured (plain path stays byte-identical).

    - Sets ``params['system_prompt']`` to the format's generation prompt — the
      output CONTRACT (it instructs the model to emit the exact JSON schema) and
      always authoritative. The user's configured template system prompt and the
      explicit "Additional instructions" field (``caption_instructions``) are
      both DEMOTED to additional guidance appended after the contract (via
      ``build_generation_prompt``): the user's intent (subject focus, naming
      rules, etc.) is preserved without letting a prose template override the
      JSON shape. A stray ``user_prompt`` from the plain template is dropped so
      it cannot fight the contract.
    - Merges ``caption_format.generation_overrides()`` into *params*.
    - For api-* models with a non-None ``json_schema()``, sets
      ``params['response_format'] = {'type': 'json_object'}``.
    """
    if not caption_format.is_structured:
        return
    # Fold the user's template system prompt + the Additional-instructions field
    # into one block of additional guidance under the authoritative contract.
    template_prompt = params.get("system_prompt") or ""
    extra = params.get("caption_instructions") or ""
    additional = "\n\n".join(
        p.strip() for p in (template_prompt, extra) if p and p.strip()
    )
    params["system_prompt"] = caption_format.build_generation_prompt(additional or None)
    params.pop("user_prompt", None)
    params.update(caption_format.generation_overrides())
    if model_id.startswith("api-") and caption_format.json_schema():
        params["response_format"] = {"type": "json_object"}


__all__ = [
    "CaptionFormat",
    "PlainFormat",
    "Ideogram4Format",
    "apply_generation_seam",
    "get_caption_format",
    "get_caption_format_for_definition",
]
