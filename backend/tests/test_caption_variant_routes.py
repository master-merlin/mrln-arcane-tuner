# backend/tests/test_caption_variant_routes.py
"""E2E tests for caption variant/suggestion routes (real tmp-dir file ops)."""

from unittest.mock import MagicMock, patch

from app.core.captioning import caption_suggestions as sg
from app.core.captioning import caption_variants as cv

_MOD = "app.api.caption_variant_routes"


@patch(f"{_MOD}.dataset_manager")
def test_list_and_accept_suggestion(mock_dm, client, tmp_path):
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds
    sg.write_suggestion(str(tmp_path), "flux1-schnell", "img1", "new cap")
    (tmp_path / "img1.txt").write_text("old general", encoding="utf-8")

    r = client.get("/api/datasets/ds/caption-suggestions?definition_id=flux1-schnell")
    assert r.status_code == 200
    body = r.json()
    assert body["definition_id"] == "flux1-schnell"
    assert body["items"][0]["stem"] == "img1"
    assert body["items"][0]["suggestion"] == "new cap"
    assert body["items"][0]["current"] == "old general"

    r2 = client.post("/api/datasets/ds/caption-suggestions/accept",
                     json={"definition_id": "flux1-schnell", "stem": "img1"})
    assert r2.status_code == 200
    assert r2.json() == {"status": "accepted"}
    assert cv.read_variant(str(tmp_path), "flux1-schnell", "img1") == "new cap"


@patch(f"{_MOD}.dataset_manager")
def test_reject_suggestion(mock_dm, client, tmp_path):
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds
    sg.write_suggestion(str(tmp_path), "flux1-schnell", "img1", "x")
    r = client.post("/api/datasets/ds/caption-suggestions/reject",
                    json={"definition_id": "flux1-schnell", "stem": "img1"})
    assert r.status_code == 200
    assert r.json() == {"status": "rejected"}
    assert sg.read_suggestion(str(tmp_path), "flux1-schnell", "img1") is None


@patch(f"{_MOD}.dataset_manager")
def test_accept_all(mock_dm, client, tmp_path):
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds
    sg.write_suggestion(str(tmp_path), "flux1-schnell", "a", "ca")
    sg.write_suggestion(str(tmp_path), "flux1-schnell", "b", "cb")
    r = client.post("/api/datasets/ds/caption-suggestions/accept-all",
                    json={"definition_id": "flux1-schnell"})
    assert r.status_code == 200
    assert r.json() == {"accepted": 2}
    assert cv.read_variant(str(tmp_path), "flux1-schnell", "a") == "ca"
    assert cv.read_variant(str(tmp_path), "flux1-schnell", "b") == "cb"


@patch(f"{_MOD}.dataset_manager")
def test_list_suggestions_masked_axis(mock_dm, client, tmp_path):
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds
    sg.write_suggestion(str(tmp_path), "flux1-schnell", "img1", "masked sug", masked=True)

    r = client.get("/api/datasets/ds/caption-suggestions",
                   params={"definition_id": "flux1-schnell", "masked": "true"})
    assert r.status_code == 200
    stems = [i["stem"] for i in r.json()["items"]]
    assert stems == ["img1"]
    # original axis is empty
    r0 = client.get("/api/datasets/ds/caption-suggestions",
                    params={"definition_id": "flux1-schnell"})
    assert [i["stem"] for i in r0.json()["items"]] == []


@patch(f"{_MOD}.dataset_manager")
def test_accept_masked_suggestion(mock_dm, client, tmp_path):
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds
    sg.write_suggestion(str(tmp_path), "flux1-schnell", "img1", "masked sug", masked=True)

    r = client.post("/api/datasets/ds/caption-suggestions/accept",
                    json={"definition_id": "flux1-schnell", "stem": "img1", "masked": True})
    assert r.status_code == 200
    assert cv.read_variant(str(tmp_path), "flux1-schnell", "img1", masked=True) == "masked sug"
    assert sg.read_suggestion(str(tmp_path), "flux1-schnell", "img1", masked=True) is None


@patch(f"{_MOD}.dataset_manager")
def test_list_variant_definitions(mock_dm, client, tmp_path):
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds
    cv.write_variant(str(tmp_path), "flux1-schnell", "a", "x")
    r = client.get("/api/datasets/ds/caption-variants")
    assert r.status_code == 200
    assert r.json() == {"definition_ids": ["flux1-schnell"]}


@patch(f"{_MOD}.dataset_manager")
def test_unknown_dataset_404(mock_dm, client):
    mock_dm.get_dataset.return_value = None
    r = client.get("/api/datasets/ghost/caption-variants")
    assert r.status_code == 404


@patch(f"{_MOD}.dataset_manager")
def test_get_caption_variant_returns_resolved_text_and_flag(mock_dm, client, tmp_path):
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds
    cv.write_variant(str(tmp_path), "flux1-schnell", "img1", "variant text")

    resp = client.get("/api/datasets/ds/caption-variant",
                      params={"definition_id": "flux1-schnell", "stem": "img1"})
    assert resp.status_code == 200
    assert resp.json() == {"text": "variant text", "has_variant": True}


@patch(f"{_MOD}.dataset_manager")
def test_get_caption_variant_falls_back_to_general(mock_dm, client, tmp_path):
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds
    (tmp_path / "img1.txt").write_text("general text", encoding="utf-8")

    resp = client.get("/api/datasets/ds/caption-variant",
                      params={"definition_id": "flux1-schnell", "stem": "img1"})
    assert resp.status_code == 200
    assert resp.json() == {"text": "general text", "has_variant": False}


@patch(f"{_MOD}.dataset_manager")
def test_put_caption_variant_writes_variant(mock_dm, client, tmp_path):
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds

    resp = client.put("/api/datasets/ds/caption-variant",
                      json={"definition_id": "flux1-schnell", "stem": "img1", "text": "edited variant"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert cv.read_variant(str(tmp_path), "flux1-schnell", "img1") == "edited variant"


@patch(f"{_MOD}.dataset_manager")
def test_get_caption_variant_map_full_payload(mock_dm, client, tmp_path):
    """Pin the full {'variants': {stem: text}} payload for the grid overlay."""
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds
    cv.write_variant(str(tmp_path), "flux1-schnell", "img1", "variant one")
    cv.write_variant(str(tmp_path), "flux1-schnell", "img2", "variant two")

    resp = client.get("/api/datasets/ds/caption-variant-map",
                      params={"definition_id": "flux1-schnell"})
    assert resp.status_code == 200
    assert resp.json() == {
        "variants": {"img1": "variant one", "img2": "variant two"}
    }
