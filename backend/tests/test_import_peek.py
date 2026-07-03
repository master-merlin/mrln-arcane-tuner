"""Route test for the generic import-peek endpoint."""

from app.core.portable.archive import write_manifest_zip


def _upload(client, zip_bytes):
    return client.post(
        "/api/import/peek",
        files={"file": ("x.zip", zip_bytes, "application/zip")})


def test_peek_returns_kind(client):
    zb = write_manifest_zip({"kind": "template", "format_version": 1, "app_version": "v"}).getvalue()
    resp = _upload(client, zb)
    assert resp.status_code == 200
    assert resp.json()["kind"] == "template"


def test_peek_full_payload(client):
    """Pin the full {kind, format_version, app_version} manifest header."""
    zb = write_manifest_zip(
        {"kind": "project", "format_version": 2, "app_version": "0.7.1-beta"}
    ).getvalue()
    resp = _upload(client, zb)
    assert resp.status_code == 200
    assert resp.json() == {
        "kind": "project",
        "format_version": 2,
        "app_version": "0.7.1-beta",
    }


def test_peek_rejects_non_archive(client):
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.txt", b"x")
    resp = _upload(client, buf.getvalue())
    assert resp.status_code == 400
