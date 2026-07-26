"""W1 fix wave — Finding 2: unguarded write primitive in caption variants
(plus the sibling read/delete operations audited alongside it).

Pre-fix, ``app/core/captioning/caption_variants.py`` built every path from
client-supplied ``definition_id``/``stem`` segments with plain
``os.path.join`` and no containment check:

- ``variant_dir``/``variant_path`` (backing ``write_variant``,
  ``read_variant``, ``has_variant``, ``list_variant_texts``) — reachable via
  ``PUT/GET /api/datasets/{name}/caption-variant[-map]``.
- ``_read_general``/``_read_masked`` (backing ``resolve_caption``'s
  general/masked fallback) — same ``stem`` parameter, same route.

The sibling module ``app/core/captioning/caption_suggestions.py``
(``suggestion_dir``/``suggestion_path``, backing ``write_suggestion``,
``read_suggestion``, ``list_suggestion_stems``, ``reject_suggestion``,
``accept_suggestion``) had the identical unguarded shape, reachable from the
same route file's ``/caption-suggestions*`` endpoints.

All of the above now resolve through the shared ``validate_path_within``
guard (``app/api/_path_guard.py``), which raises ``HTTPException(403)`` on
escape and returns the resolved path actually used for the I/O.

Traversal strings below use a generous run of ``../`` (more than the
nesting depth strictly requires) so the assertion doesn't depend on exactly
how many directories a given helper inserts before the client segment —
any escape past ``dataset_path`` trips the guard the same way.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core.captioning import caption_suggestions as sg
from app.core.captioning import caption_variants as cv

_UP = "../../../../"  # comfortably more than any helper's internal nesting


# ── caption_variants.py: write_variant (the primary finding) ───────────────


def test_write_variant_rejects_definition_id_traversal(tmp_path):
    ds = str(tmp_path)
    with pytest.raises(HTTPException):
        cv.write_variant(ds, _UP + "w1t2_outside_defid", "img1", "pwned")


def test_write_variant_rejects_stem_traversal(tmp_path):
    ds = str(tmp_path)
    with pytest.raises(HTTPException):
        cv.write_variant(ds, "flux1-schnell", _UP + "w1t2_outside_stem", "pwned")


def test_write_variant_preserves_outside_file_on_traversal(tmp_path):
    ds = str(tmp_path)
    outside_file = tmp_path.parent / "w1t2_write_variant_sentinel.txt"
    outside_file.write_text("do not touch me", encoding="utf-8")
    with pytest.raises(HTTPException):
        cv.write_variant(ds, _UP + "evil", "img1", "pwned")
    assert outside_file.read_text(encoding="utf-8") == "do not touch me"


# ── caption_variants.py sibling reads: read_variant / has_variant /
#    list_variant_texts / the general+masked fallback in resolve_caption ──


def test_read_variant_rejects_traversal(tmp_path):
    ds = str(tmp_path)
    with pytest.raises(HTTPException):
        cv.read_variant(ds, "flux1-schnell", _UP + "w1t2_secret")


def test_has_variant_rejects_traversal(tmp_path):
    with pytest.raises(HTTPException):
        cv.has_variant(str(tmp_path), _UP + "evil", "img1")


def test_list_variant_texts_rejects_definition_id_traversal(tmp_path):
    with pytest.raises(HTTPException):
        cv.list_variant_texts(str(tmp_path), _UP + "evil")


def test_resolve_caption_general_fallback_rejects_stem_traversal(tmp_path):
    """resolve_caption's general/masked fallback (_read_general/_read_masked)
    took the same client-supplied `stem` with no guard at all pre-fix."""
    ds = str(tmp_path)
    secret = tmp_path.parent / "w1t2_resolve_secret.txt"
    secret.write_text("leaked", encoding="utf-8")
    with pytest.raises(HTTPException):
        cv.resolve_caption(ds, "../w1t2_resolve_secret", None)


# ── caption_suggestions.py: the identical sibling-module vulnerability ────


def test_write_suggestion_rejects_definition_id_traversal(tmp_path):
    ds = str(tmp_path)
    with pytest.raises(HTTPException):
        sg.write_suggestion(ds, _UP + "w1t2_sugg_outside_defid", "img1", "pwned")


def test_write_suggestion_rejects_stem_traversal(tmp_path):
    ds = str(tmp_path)
    with pytest.raises(HTTPException):
        sg.write_suggestion(ds, "flux1-schnell", _UP + "w1t2_sugg_outside_stem", "pwned")


def test_reject_suggestion_rejects_traversal_and_preserves_outside_file(tmp_path):
    """reject_suggestion is the "delete" analog the brief asked us to check —
    it must not be able to os.remove a file outside the dataset."""
    ds = str(tmp_path)
    outside_file = tmp_path.parent / "w1t2_sugg_delete_sentinel.txt"
    outside_file.write_text("do not delete me", encoding="utf-8")
    with pytest.raises(HTTPException):
        sg.reject_suggestion(ds, _UP + "evil", "img1")
    assert outside_file.exists()


def test_read_suggestion_rejects_traversal(tmp_path):
    with pytest.raises(HTTPException):
        sg.read_suggestion(str(tmp_path), _UP + "evil", "img1")


def test_list_suggestion_stems_rejects_traversal(tmp_path):
    with pytest.raises(HTTPException):
        sg.list_suggestion_stems(str(tmp_path), _UP + "evil")


def test_accept_suggestion_rejects_traversal(tmp_path):
    with pytest.raises(HTTPException):
        sg.accept_suggestion(str(tmp_path), _UP + "evil", "img1")


# ── Sanity: legitimate round-trips are unaffected (regression) ────────────


def test_write_and_read_variant_roundtrip_still_works(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "a flux caption")
    assert cv.read_variant(ds, "flux1-schnell", "img1") == "a flux caption"
    assert cv.has_variant(ds, "flux1-schnell", "img1") is True


def test_suggestion_accept_roundtrip_still_works(tmp_path):
    ds = str(tmp_path)
    sg.write_suggestion(ds, "flux1-schnell", "img1", "new variant")
    sg.accept_suggestion(ds, "flux1-schnell", "img1")
    assert cv.read_variant(ds, "flux1-schnell", "img1") == "new variant"
    assert sg.read_suggestion(ds, "flux1-schnell", "img1") is None


def test_resolve_caption_general_fallback_still_works(tmp_path):
    ds = str(tmp_path)
    (tmp_path / "img1.txt").write_text("general caption", encoding="utf-8")
    assert cv.resolve_caption(ds, "img1", None) == "general caption"
