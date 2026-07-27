"""W4.T11 practical proxy: routes that accept an uploaded archive must never
call a bare (full-buffer) ``UploadFile.read()`` — a project/dataset archive
can embed multi-GB video media, and a bare read() would buffer the whole
thing in RAM before ever touching a temp file.

This patches ``UploadFile.read`` to reject the argument-less (``size=-1``)
call while still honoring chunked reads (``read(1024 * 1024)``), so a real
(small) upload only succeeds if the route actually streams it in chunks —
demonstrated failing (RED) against the pre-W4.T11 routes, passing (GREEN)
after they were switched to chunked temp-file streaming.
"""

from unittest.mock import patch

import pytest
from starlette.datastructures import UploadFile

from app.core.portable.archive import write_bundle_zip, write_manifest_zip
from app.core.project import portable as pportable

_ORIG_READ = UploadFile.read


async def _guarded_read(self, size: int = -1) -> bytes:
    if size == -1:
        raise AssertionError(
            "route called UploadFile.read() with no size argument — a bare "
            "full-buffer read defeats streaming the upload to a temp file"
        )
    return await _ORIG_READ(self, size)


@pytest.fixture
def guard_bare_upload_read():
    with patch.object(UploadFile, "read", _guarded_read):
        yield


def _small_project_zip() -> bytes:
    manifest = pportable.build_project_manifest({"name": "P"}, {}, [], [], "v")
    return write_bundle_zip(manifest, {}).getvalue()


def test_peek_archive_streams_upload_in_chunks(client, guard_bare_upload_read):
    zb = write_manifest_zip(
        {"kind": "template", "format_version": 1, "app_version": "v"}
    ).getvalue()
    resp = client.post(
        "/api/import/peek", files={"file": ("x.zip", zb, "application/zip")}
    )
    assert resp.status_code == 200
    assert resp.json()["kind"] == "template"


def test_project_plan_import_streams_upload_in_chunks(client, guard_bare_upload_read):
    zb = _small_project_zip()
    with patch("app.api.project_routes._projects") as MockProjects:
        MockProjects.get_by_name.return_value = None
        resp = client.post(
            "/api/projects/import/plan",
            files={"file": ("p.project.zip", zb, "application/zip")},
        )
    assert resp.status_code == 200


def test_project_apply_import_streams_upload_in_chunks(client, guard_bare_upload_read):
    zb = _small_project_zip()
    with (
        patch("app.api.project_routes._projects") as MockProjects,
        patch("app.api.project_routes._prefs"),
    ):
        MockProjects.get_by_name.return_value = None
        MockProjects.create.return_value = {"id": "new_p", "name": "P"}
        resp = client.post(
            "/api/projects/import/apply",
            files={"file": ("p.project.zip", zb, "application/zip")},
        )
    assert resp.status_code == 200
    assert resp.json()["project_id"] == "new_p"
