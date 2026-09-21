# backend/tests/test_caption_context_routes.py
"""E2E tests for caption_context_routes."""

from unittest.mock import patch

from app.engine.core.caption_target import CaptionTarget

_MODULE = "app.api.caption_context_routes"


def test_list_definitions_returns_id_family_name_caption_format(client):
    from app.engine.models.registry import registry

    registry.initialize()  # no lifespan under TestClient: an empty list proves nothing
    resp = client.get("/api/caption-context/definitions")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list) and body, "no selectable definitions served"
    for entry in body:
        assert set(entry.keys()) == {"id", "family", "name", "caption_format", "frame_rule", "ingest_fps"}


def test_definitions_carry_frame_rule(client):
    """REQUEST-6: the selector route serves each definition's ``video.frame_rule``
    (the SPA derives its frame guidance from it, never from a family literal).

    The typed response model DROPS undeclared fields, so this asserts on the
    served JSON, compared against the registry's own architecture_params.
    """
    from app.engine.models.registry import registry

    # The TestClient is built without the lifespan, so the registry is empty
    # unless this test fills it — an empty list would make every check vacuous.
    registry.initialize()
    body = client.get("/api/caption-context/definitions").json()
    assert body, "no selectable definitions served"
    served_rules: set[str] = set()
    for d in body:
        assert "frame_rule" in d, f"{d['id']}: frame_rule absent from the response"
        expected = registry.get_definition(d["id"]).architecture_params.get(
            "video.frame_rule"
        )
        assert d["frame_rule"] == expected, d["id"]
        if d["frame_rule"]:
            served_rules.add(d["frame_rule"])
    # Prove the positive AND the negative: video families state a rule, image
    # families serve null (never a defaulted string).
    assert {"4n+1", "8n+1"} <= served_rules
    assert any(d["frame_rule"] is None for d in body)
    # LANE-92 row 5.1 / REQUEST-15: this route excludes definitions carrying an
    # `unavailable_reason`, so the ungated H3 definition is served here — with
    # its own rule, from the real route (row 4.3 could only inject it).
    h3 = {d["id"]: d["frame_rule"] for d in body if d["id"].startswith("minimax-h3-")}
    assert h3 == {"minimax-h3-t2va": "17n+5"}, h3


def test_definitions_carry_the_ingestion_clock(client):
    """LANE-92 VERIFY 3.01: a frame rule counts TRAINING frames, and a
    definition that states ``video.ingest_at_native_fps`` trains every clip on
    its own clock, not the clip's. The SPA cannot turn a source frame count
    into a training frame count without that number, so it is served beside the
    rule — the same value ingestion resolves, from the same resolver.
    """
    from app.engine.core.video_contract import resolve_video_profile
    from app.engine.models.registry import registry

    registry.initialize()
    body = client.get("/api/caption-context/definitions").json()
    assert body, "no selectable definitions served"
    for d in body:
        assert "ingest_fps" in d, f"{d['id']}: ingest_fps absent from the response"
        defn = registry.get_definition(d["id"])
        assert d["ingest_fps"] == resolve_video_profile(defn).ingest_fps, d["id"]
    clocks = {d["id"]: d["ingest_fps"] for d in body if d["ingest_fps"] is not None}
    assert clocks == {"minimax-h3-t2va": 24.0}, clocks
    # The negative, by name: video definitions whose clips keep their own fps.
    served = {d["id"]: d for d in body}
    for def_id in ("wan2.1-t2v-1.3b", "wan2.2-t2v-a14b"):
        assert served[def_id]["frame_rule"] == "4n+1"
        assert served[def_id]["ingest_fps"] is None


def test_list_definitions_serves_caption_format_for_selector(client):
    """The selector route MUST carry caption_format — it drives the frontend
    structured-editor swap. ideogram4 → 'ideogram4_json'; others → 'plain'."""
    from app.engine.models.registry import registry

    registry.initialize()  # no lifespan under TestClient: `all()` over [] proves nothing
    body = client.get("/api/caption-context/definitions").json()
    assert body, "no selectable definitions served"
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
