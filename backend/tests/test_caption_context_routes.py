# backend/tests/test_caption_context_routes.py
"""E2E tests for caption_context_routes."""

from unittest.mock import patch

from app.engine.core.caption_target import CaptionTarget

_MODULE = "app.api.caption_context_routes"


def test_list_definitions_returns_id_family_name_caption_format(client):
    resp = client.get("/api/caption-context/definitions")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    if body:
        first = body[0]
        assert set(first.keys()) == {"id", "family", "name", "caption_format"}


def test_list_definitions_serves_caption_format_for_selector(client):
    """The selector route MUST carry caption_format — it drives the frontend
    structured-editor swap. ideogram4 → 'ideogram4_json'; others → 'plain'."""
    body = client.get("/api/caption-context/definitions").json()
    by_family: dict[str, str] = {d["family"]: d["caption_format"] for d in body}
    # Every entry has a non-empty format key.
    assert all(d["caption_format"] for d in body)
    # ideogram4 family resolves to the structured format when present.
    if "ideogram4" in by_family:
        assert by_family["ideogram4"] == "ideogram4_json"
    # A representative flat family stays plain when present.
    for flat in ("flux1", "sdxl"):
        if flat in by_family:
            assert by_family[flat] == "plain"


@patch(f"{_MODULE}.resolve_caption_target")
@patch(f"{_MODULE}.TokenizerService")
def test_token_count_truncating(mock_svc_cls, mock_resolve, client):
    mock_resolve.return_value = CaptionTarget(
        "flux1", "t5", "google/t5-v1_1-xxl", 256, 255
    )
    mock_svc_cls.get_instance.return_value.count_with_cutoff.return_value = (300, 1024)

    resp = client.post(
        "/api/caption-context/token-count",
        json={"text": "x" * 2000, "definition_id": "flux1-schnell"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "tokens": 300,
        "limit": 255,
        "will_truncate": True,
        "cutoff_char_index": 1024,
    }


@patch(f"{_MODULE}.resolve_caption_target")
@patch(f"{_MODULE}.TokenizerService")
def test_token_count_within_limit(mock_svc_cls, mock_resolve, client):
    mock_resolve.return_value = CaptionTarget(
        "sdxl", "clip", "openai/clip-vit-large-patch14", 77, 75
    )
    mock_svc_cls.get_instance.return_value.count_with_cutoff.return_value = (10, None)

    resp = client.post(
        "/api/caption-context/token-count",
        json={"text": "a short caption", "definition_id": "sdxl_base_1.0"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["will_truncate"] is False
    assert body["cutoff_char_index"] is None
    assert body["limit"] == 75


@patch(f"{_MODULE}.resolve_caption_target", side_effect=ValueError("nope"))
def test_token_count_unknown_definition_404(_mock_resolve, client):
    resp = client.post(
        "/api/caption-context/token-count",
        json={"text": "hi", "definition_id": "ghost"},
    )
    assert resp.status_code == 404
