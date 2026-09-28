"""LANE-100: a sidecar write must never land on a media filename.

``PUT /api/datasets/{name}/captions/{filename:path}`` and its lyrics twin pass
the client-supplied ``filename`` straight to ``DatasetManager._write_sidecar``
(via ``save_caption`` / ``save_lyrics``), which writes caption/lyrics TEXT to
whatever path it is given, regardless of extension. Point either route at a
media filename (``clip.mp4``, ``shot.png``, ...) and the media file is
replaced with caption text, the route answers ``{"status": "saved"}``, and no
error is raised anywhere -- the user's source material is gone. The UI never
sends a media name (it always writes the ``.txt`` sidecar), so only a direct
API call hits this: a script, a proxy neighbour, a future client.

Both tests below go through the REAL FastAPI route with a REAL
``DatasetManager`` rooted in ``tmp_path`` (only the DB layer is mocked, the
same shape as ``test_path_traversal_containment.py``'s ``env`` fixture,
because the containment guard the route relies on must not be stubbed here
either). Parametrized over every extension in
``app.core.dataset.media_types.MULTIMEDIA_EXTENSIONS`` -- imported, never
copied, so a container added there is covered automatically -- crossed with
both cases the fix must handle: an existing media file (must stay
byte-identical) and an absent media name (must not be created).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dataset.media_types import MULTIMEDIA_EXTENSIONS
from app.core.dataset_manager import DatasetManager

# The message every byte/existence assertion below carries, so a failure
# names the invariant rather than just a hash mismatch.
GUARD_MESSAGE = "sidecar save overwrote or created a media file"

ORIGINAL_BYTES = b"ORIGINAL-MEDIA-BYTES-\x00\x01\xff-not-a-caption"


@pytest.fixture()
def mock_settings():
    inst = MagicMock()
    inst.get_module_settings.return_value = {}
    inst.update_module_settings = MagicMock()
    with patch("app.core.dataset_manager.get_settings_manager", return_value=inst):
        yield inst


@pytest.fixture()
def env(tmp_path, mock_settings, monkeypatch):
    """A real DatasetManager rooted in tmp_path, wired into the real app.

    Mirrors ``test_path_traversal_containment.py``'s ``env`` fixture: the
    manager is real on purpose because the write path under test
    (``save_caption``/``save_lyrics`` -> ``_write_sidecar``) is what must be
    guarded, so stubbing the manager would stub the very thing under test.
    Only the DB layer (persistence side effects unrelated to this guard) is
    mocked.
    """
    default_root = tmp_path / "datasets"
    default_root.mkdir(parents=True, exist_ok=True)

    with patch.object(DatasetManager, "__init__", lambda self, **kw: None):
        mgr = DatasetManager()
    mgr.root_dir = str(tmp_path)
    mgr.storage_file = str(tmp_path / "dataset_locations.json")
    mgr.default_root = str(default_root)
    mgr.settings_manager = mock_settings
    mgr.datasets = {}
    mgr._loop = None
    mgr._db = MagicMock()
    mgr._dataset_repo = MagicMock()
    mgr._media_repo = MagicMock()

    ds = mgr.create_dataset("ds")

    from app.core import dataset_manager as dm_mod

    monkeypatch.setattr(dm_mod, "dataset_manager", mgr)
    for mod in ("app.api.dataset.crud_routes",):
        m = __import__(mod, fromlist=["dataset_manager"])
        monkeypatch.setattr(m, "dataset_manager", mgr, raising=False)

    return {"mgr": mgr, "ds": ds, "root": Path(ds.path)}


def _client():
    from app.main import app

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spellings(ext: str) -> list[tuple[str, str]]:
    """Return (spelling_id, spelled_ext) for the three case-forms under test.

    ``ext`` is as listed in ``MULTIMEDIA_EXTENSIONS`` (lowercase, e.g.
    ``.mp4``). ``lower`` is that value unchanged; ``upper`` uppercases the
    whole extension (``.MP4``); ``mixed`` uppercases only the first letter
    after the dot, keeping the rest lowercase (``.Mp4``).
    """
    stem = ext[1:]  # drop the leading dot
    lower = ext
    upper = "." + stem.upper()
    mixed = "." + (stem[0].upper() + stem[1:].lower() if stem else stem)
    return [
        (f"{stem}-lower", lower),
        (f"{stem}-upper", upper),
        (f"{stem}-mixed", mixed),
    ]


EXTENSIONS = sorted(MULTIMEDIA_EXTENSIONS)
CASES = [
    (spelled_ext, existing)
    for ext in EXTENSIONS
    for _spelling_id, spelled_ext in _spellings(ext)
    for existing in (True, False)
]
CASE_IDS = [
    f"{spelling_id}-{'existing' if existing else 'absent'}"
    for ext in EXTENSIONS
    for spelling_id, _spelled_ext in _spellings(ext)
    for existing in (True, False)
]


@pytest.mark.asyncio
@pytest.mark.parametrize("ext,existing", CASES, ids=CASE_IDS)
async def test_put_caption_on_media_name_leaves_media_untouched(env, ext, existing):
    media_name = f"clip{ext}"
    media_path = env["root"] / media_name
    before_hash = None
    if existing:
        media_path.write_bytes(ORIGINAL_BYTES)
        before_hash = _sha256(media_path)

    async with _client() as client:
        r = await client.put(
            f"/api/datasets/ds/captions/{media_name}",
            json={"content": "this is caption text, not media"},
        )

    if existing:
        assert media_path.exists(), (
            f"{GUARD_MESSAGE} (media file vanished): {media_name}"
        )
        assert _sha256(media_path) == before_hash, (
            f"{GUARD_MESSAGE}: {media_name} content changed"
        )
    else:
        assert not media_path.exists(), (
            f"{GUARD_MESSAGE}: {media_name} was created by a caption save"
        )

    assert r.status_code == 400, (
        f"expected a 400 refusal for a caption PUT on the media name "
        f"{media_name!r}, got {r.status_code}: {r.text[:200]}"
    )
    detail = r.json()["detail"]
    assert ext.lower() in detail.lower(), (
        f"expected the refusal detail to name the rejected extension "
        f"{ext!r}, got detail={detail!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("ext,existing", CASES, ids=CASE_IDS)
async def test_put_lyrics_on_media_name_leaves_media_untouched(env, ext, existing):
    media_name = f"track{ext}"
    media_path = env["root"] / media_name
    before_hash = None
    if existing:
        media_path.write_bytes(ORIGINAL_BYTES)
        before_hash = _sha256(media_path)

    async with _client() as client:
        r = await client.put(
            f"/api/datasets/ds/lyrics/{media_name}",
            json={"content": "these are lyrics, not media"},
        )

    if existing:
        assert media_path.exists(), (
            f"{GUARD_MESSAGE} (media file vanished): {media_name}"
        )
        assert _sha256(media_path) == before_hash, (
            f"{GUARD_MESSAGE}: {media_name} content changed"
        )
    else:
        assert not media_path.exists(), (
            f"{GUARD_MESSAGE}: {media_name} was created by a lyrics save"
        )

    assert r.status_code == 400, (
        f"expected a 400 refusal for a lyrics PUT on the media name "
        f"{media_name!r}, got {r.status_code}: {r.text[:200]}"
    )
    detail = r.json()["detail"]
    assert ext.lower() in detail.lower(), (
        f"expected the refusal detail to name the rejected extension "
        f"{ext!r}, got detail={detail!r}"
    )


# ── Positive controls: legitimate sidecar saves must keep working ──────────
# These must PASS both before and after the fix. They do not start with
# `test_put_caption_on_media`/`test_put_lyrics_on_media` so they are never
# mistaken for the guard tests above.


@pytest.mark.asyncio
async def test_plain_txt_sidecar_saves_and_reads_back(env):
    async with _client() as client:
        w = await client.put(
            "/api/datasets/ds/captions/shot.txt", json={"content": "a plain caption"}
        )
        assert w.status_code == 200, w.text
        r = await client.get("/api/datasets/ds/captions/shot.txt")
        assert r.status_code == 200
        assert r.json()["content"] == "a plain caption"

    saved = env["root"] / "shot.txt"
    assert saved.exists()
    assert saved.read_text(encoding="utf-8") == "a plain caption"


@pytest.mark.asyncio
async def test_variant_caption_path_saves_and_reads_back(env):
    # The route already consumes the literal "captions/" prefix
    # (`/datasets/{name}/captions/{filename:path}`), so the wire form below
    # resolves on disk to `<dataset>/captions/<definition>/<stem>.txt` -- the
    # exact layout `app.core.captioning.caption_variants.variant_path` uses.
    variant_wire = "captions/my_definition/shot.txt"
    async with _client() as client:
        w = await client.put(
            f"/api/datasets/ds/captions/{variant_wire}",
            json={"content": "a variant caption"},
        )
        assert w.status_code == 200, w.text
        r = await client.get(f"/api/datasets/ds/captions/{variant_wire}")
        assert r.status_code == 200
        assert r.json()["content"] == "a variant caption"

    saved = env["root"] / "captions" / "my_definition" / "shot.txt"
    assert saved.exists()
    assert saved.read_text(encoding="utf-8") == "a variant caption"


@pytest.mark.asyncio
async def test_lyrics_txt_sidecar_saves_and_reads_back(env):
    async with _client() as client:
        w = await client.put(
            "/api/datasets/ds/lyrics/track.lyrics.txt", json={"content": "la la la"}
        )
        assert w.status_code == 200, w.text
        r = await client.get("/api/datasets/ds/lyrics/track.lyrics.txt")
        assert r.status_code == 200
        assert r.json()["content"] == "la la la"

    saved = env["root"] / "track.lyrics.txt"
    assert saved.exists()
    assert saved.read_text(encoding="utf-8") == "la la la"
