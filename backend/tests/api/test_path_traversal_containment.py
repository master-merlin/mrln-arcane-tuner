"""Traversal containment for the `{filename:path}` routes, over real HTTP.

WHY THIS EXISTS, AND WHY THE ACCEPT CASES ARE NOT OPTIONAL
-----------------------------------------------------------
Until the client encoded its URL path segments, a raw ``../`` never reached
Python at all: URL normalisation collapsed it and the request 404'd on routing.
That was an *accidental* guard -- nobody designed it, and no code depended on it
knowingly. Percent-encoded, the same traversal arrives at the handler intact,
because the ASGI server decodes the path before Starlette matches it.

Measured, against this very transport:

    sent '../../etc/passwd'                -> 404, handler never reached
    sent '%2E%2E%2F%2E%2E%2Fetc%2Fpasswd'  -> 200, handler saw '../../etc/passwd'
    sent '%252E%252E%252Fetc%252Fpasswd'   -> 200, handler saw '%2E%2E%2Fetc%2Fpasswd'

So encoding did not weaken anything -- it removed a guard that was never ours,
and ``validate_path_within`` goes from second line of defence to the ONLY line
of defence. This module pins that line.

The accept half is therefore load-bearing, not decoration. A containment test
that only sends hostile input can pass while every request 404s for an
unrelated reason -- proving nothing at all. The accept cases prove the routes
still reach real files, so a green reject half means something. Do not delete
them to "focus the test".

PLATFORM
--------
The product ships on Windows, where ``\\`` is a path separator; CI runs Linux,
where it is an ordinary filename character. The invariant that holds on BOTH is
the one asserted everywhere here: *no file outside the dataset root is read or
written*. Status codes are asserted only where the platforms genuinely agree.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dataset_manager import DatasetManager

CANARY = "TOP-SECRET-CANARY-CONTENT-b7f3"


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

    The manager is real on purpose: `validate_path_within` is the seam under
    test, so stubbing the manager would stub the very thing being verified.
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

    # The canary sits OUTSIDE the dataset root, one level up -- the exact file a
    # successful "../" escape would reach.
    canary = Path(ds.path).parent / "canary.txt"
    canary.write_text(CANARY, encoding="utf-8")

    from app.core import dataset_manager as dm_mod

    monkeypatch.setattr(dm_mod, "dataset_manager", mgr)
    for mod in ("app.api.dataset.crud_routes",):
        m = __import__(mod, fromlist=["dataset_manager"])
        monkeypatch.setattr(m, "dataset_manager", mgr, raising=False)

    return {"mgr": mgr, "ds": ds, "root": Path(ds.path), "canary": canary}


def _client():
    from app.main import app

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


# Hostile inputs, as they appear ON THE WIRE. Encoded forms are the ones that
# actually reach the handler; the raw forms are kept so the test still means
# something if URL normalisation behaviour ever changes underneath us.
REJECT_CASES = [
    ("raw-dotdot", "../../canary.txt"),
    ("encoded-dotdot-upper", "%2E%2E%2Fcanary.txt"),
    ("encoded-dotdot-lower", "%2e%2e%2fcanary.txt"),
    ("encoded-deep", "%2e%2e%2f%2e%2e%2fcanary.txt"),
    ("encoded-backslash", "%2e%2e%5ccanary.txt"),
    ("raw-backslash", "..\\canary.txt"),
    ("absolute-posix", "/etc/passwd"),
    ("absolute-windows", "C:%5CWindows%5Cwin.ini"),
    ("unc", "%5C%5Cserver%5Cshare%5Cx.txt"),
    ("double-encoded", "%252E%252E%252Fcanary.txt"),
    ("nullbyte", "foo%00.txt"),
    ("trailing-dot", "foo.txt."),
    ("trailing-space", "foo.txt%20"),
]

# Legitimate filenames that MUST still work. The last five are exactly what the
# client-side encoding fix exists to deliver -- raw interpolation truncated them
# at '#'/'?' and silently read and wrote the wrong sidecar at HTTP 200.
ACCEPT_CASES = [
    ("nested", "sub/nested.png.txt"),
    ("deep", "deep/a/b/c.png.txt"),
    ("hash", "hash%23tag.png.txt"),
    ("question", "query%3Fx=1.png.txt"),
    ("percent", "percent100%25.png.txt"),
    ("space", "has%20space.png.txt"),
    ("unicode", "unicode-%C3%A4%C3%B6%C3%BC.png.txt"),
]


def _escaped(root: Path, canary: Path) -> list[str]:
    """Any file that appeared outside the dataset root, or a mutated canary."""
    problems = []
    if canary.read_text(encoding="utf-8") != CANARY:
        problems.append(f"canary CONTENT CHANGED: {canary}")
    for p in canary.parent.rglob("*"):
        if p.is_file() and not p.is_relative_to(root) and p != canary:
            problems.append(f"file written outside root: {p}")
    return problems


@pytest.mark.asyncio
@pytest.mark.parametrize("label,wire", REJECT_CASES, ids=[c[0] for c in REJECT_CASES])
async def test_read_never_escapes_the_dataset_root(env, label, wire):
    """A hostile filename must never return content from outside the root.

    Asserted on the OBSERVABLE outcome rather than on a status code: a 200 whose
    body carries the canary is the real failure, and a guard that returns empty
    content for an escaped path would pass a status-only assertion.
    """
    async with _client() as client:
        r = await client.get(f"/api/datasets/ds/captions/{wire}")

    assert CANARY not in r.text, (
        f"{label}: content from OUTSIDE the dataset root was returned "
        f"(status {r.status_code}). Wire form: {wire!r}"
    )
    assert r.status_code != 500, (
        f"{label}: hostile input produced a 500 rather than a handled refusal. "
        f"An unhandled exception is not a guard. Body: {r.text[:200]}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("label,wire", REJECT_CASES, ids=[c[0] for c in REJECT_CASES])
async def test_write_never_escapes_the_dataset_root(env, label, wire):
    """A hostile filename must never create or modify a file outside the root."""
    async with _client() as client:
        r = await client.put(
            f"/api/datasets/ds/captions/{wire}", json={"content": "PWNED"}
        )

    assert r.status_code != 500, (
        f"{label}: hostile write produced a 500 rather than a handled refusal. "
        f"Body: {r.text[:200]}"
    )
    problems = _escaped(env["root"], env["canary"])
    assert not problems, f"{label}: wrote outside the dataset root -> {problems}"


@pytest.mark.asyncio
async def test_double_encoding_is_not_decoded_twice(env):
    """`%252E%252E%252F` must be a literal filename, never traversal.

    Decoding twice anywhere in the stack would turn this into `../`. Measured:
    the handler receives `%2E%2E%2F...` -- decoded exactly once. Pinned because
    a future "helpfully" added unquote() would silently reopen traversal.
    """
    async with _client() as client:
        r = await client.put(
            "/api/datasets/ds/captions/%252E%252E%252Fcanary.txt",
            json={"content": "literal"},
        )
    assert r.status_code != 500
    assert env["canary"].read_text(encoding="utf-8") == CANARY
    assert not _escaped(env["root"], env["canary"])


# The decoded filename each wire form MUST produce. Truncation at '#' or '?'
# is the live defect the client-side encoding fix addresses: the backend
# answered 200 for a DIFFERENT file, so the caption editor read and wrote the
# wrong sidecar with no error anywhere.
ACCEPT_DECODED = {
    "nested": "sub/nested.png.txt",
    "deep": "deep/a/b/c.png.txt",
    "hash": "hash#tag.png.txt",
    "question": "query?x=1.png.txt",
    "percent": "percent100%.png.txt",
    "space": "has space.png.txt",
    "unicode": "unicode-äöü.png.txt",
}

# Characters Windows forbids in a filename. A name containing one cannot be
# stored on this platform at all, so a round-trip is impossible there -- but the
# name must still ARRIVE intact, which is the property under test.
_WINDOWS_ILLEGAL = set('<>:"|?*')


@pytest.mark.asyncio
@pytest.mark.parametrize("label,wire", ACCEPT_CASES, ids=[c[0] for c in ACCEPT_CASES])
async def test_legitimate_filenames_arrive_untruncated(env, label, wire, monkeypatch):
    """The accept half: the handler must receive the WHOLE filename.

    Asserted on what the handler actually received rather than on a round-trip,
    because two accept cases cannot round-trip for reasons that have nothing to
    do with this guard: '?' is illegal in a Windows filename, and nested paths
    need a directory that may not exist. Truncation is the defect; storage is a
    separate concern.

    Without this half the reject half could pass simply because every request
    fails for an unrelated reason, which would prove nothing.
    """
    seen: list[str] = []
    real = env["mgr"].read_caption

    def spy(name, filename):
        seen.append(filename)
        return real(name, filename)

    monkeypatch.setattr(env["mgr"], "read_caption", spy)

    async with _client() as client:
        await client.get(f"/api/datasets/ds/captions/{wire}")

    assert seen, f"{label}: the route never reached the handler at all"
    assert seen[0] == ACCEPT_DECODED[label], (
        f"{label}: handler received {seen[0]!r}, expected "
        f"{ACCEPT_DECODED[label]!r} -- the filename was altered in transit. "
        f"Truncation here means reading and writing the WRONG file at HTTP 200."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("label,wire", ACCEPT_CASES, ids=[c[0] for c in ACCEPT_CASES])
async def test_legitimate_filenames_round_trip_where_the_os_allows(env, label, wire):
    """End-to-end write-then-read for every name this platform can store.

    Proves the routes still reach real files inside the root, so a green reject
    half is meaningful rather than vacuous.
    """
    decoded = ACCEPT_DECODED[label]
    if any(ch in _WINDOWS_ILLEGAL for ch in Path(decoded).name) and os.name == "nt":
        pytest.skip(f"{decoded!r} is not a legal filename on Windows")

    # Captions live beside their image, so a nested caption implies a nested
    # image directory. Create it: this test is about containment and
    # round-tripping, not about whether save_caption creates parents.
    (env["root"] / decoded).parent.mkdir(parents=True, exist_ok=True)

    body = f"caption for {label}"
    async with _client() as client:
        w = await client.put(
            f"/api/datasets/ds/captions/{wire}", json={"content": body}
        )
        assert w.status_code == 200, f"{label}: write failed: {w.status_code} {w.text[:200]}"

        r = await client.get(f"/api/datasets/ds/captions/{wire}")
        assert r.status_code == 200, f"{label}: read failed: {r.status_code}"
        assert r.json()["content"] == body, (
            f"{label}: round-trip returned different content -- the request "
            f"resolved to the WRONG file, which is a 200-shaped data-loss bug."
        )

    written = [p for p in env["root"].rglob("*") if p.is_file()]
    assert written, f"{label}: reported success but wrote nothing"
    assert all(p.is_relative_to(env["root"]) for p in written)
    assert not _escaped(env["root"], env["canary"])


@pytest.mark.asyncio
async def test_the_canary_is_actually_reachable_on_disk(env):
    """Prove the negative: the traversal target really exists and is readable.

    If the canary were missing or empty, every reject assertion above would pass
    vacuously and this module would be worthless while looking green.
    """
    assert env["canary"].is_file()
    assert env["canary"].read_text(encoding="utf-8") == CANARY
    assert not env["canary"].is_relative_to(env["root"]), (
        "the canary must live OUTSIDE the dataset root or it tests nothing"
    )
    # And the escape really is only one level up, i.e. '../canary.txt' is the
    # correct attack string for this layout.
    assert env["canary"] == Path(env["root"]).parent / "canary.txt"
    assert os.path.exists(env["canary"])
