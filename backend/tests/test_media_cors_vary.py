"""MediaStaticFiles must emit `Vary: Origin` on origin-less (no-cors) responses.

Regression guard for the editor-canvas image-loading bug: without
`Vary: Origin`, a no-cors cached body (from a plain <img>) could be replayed
for a crossorigin (cors) request and rejected by the browser, breaking the
editor canvas.
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import MediaStaticFiles


def _client(tmp_path) -> TestClient:
    (tmp_path / "img.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\xff\xd9")
    app = FastAPI()
    app.mount("/media", MediaStaticFiles(directory=str(tmp_path)), name="media")
    return TestClient(app)


def test_no_origin_response_has_vary_origin(tmp_path):
    # A plain <img> request carries no Origin header.
    resp = _client(tmp_path).get("/media/img.jpg")
    assert resp.status_code == 200
    assert "origin" in resp.headers.get("vary", "").lower()


def test_vary_origin_present_even_without_cors_middleware(tmp_path):
    # The mount itself must add the header — it does not rely on
    # CORSMiddleware (which only fires when an Origin header is present).
    resp = _client(tmp_path).get("/media/img.jpg", headers={})
    assert resp.headers.get("vary", "").lower() == "origin"
