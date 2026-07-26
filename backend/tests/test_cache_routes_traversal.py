"""Purge segments must never escape the dataset .cache root."""

import pytest
from fastapi import HTTPException

from app.api.cache_routes import _validate_cache_segment


@pytest.mark.parametrize("bad", ["..", ".", "a/b", "a\\b", "..\\..", ""])
def test_bad_segments_rejected(bad):
    with pytest.raises(HTTPException) as exc:
        _validate_cache_segment(bad)
    assert exc.value.status_code == 400


@pytest.mark.parametrize("ok", ["qwen_image", "v3", "latents", "original_512x512"])
def test_plain_names_pass(ok):
    assert _validate_cache_segment(ok) == ok
