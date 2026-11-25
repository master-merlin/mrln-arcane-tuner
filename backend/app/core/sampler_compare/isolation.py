"""The comparison's own stores: a database, settings, a server log and a dataset copy, all under
its work folder, so its backend and trainer never write into the user's.

Shared and only read: the code checkout, the Python environment, the model weights. Every file
written here goes through a fresh name and ``os.replace``, so a link in the target is replaced,
never written through.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from app.core.sampler_compare.plan import Comparison

# Every path seam the backend reads, and where a comparison's children find it under the work
# folder. The database is copied afresh for each run into the folder named here: a killed
# backend can leave a write-ahead log beside its file, and a new copy under the old name would
# meet it.
ISOLATED_PATHS: dict[str, str] = {
    "MRLN_DB_PATH": "data/db",
    "MRLN_SETTINGS_PATH": "data/settings.json",
    "MRLN_SERVER_LOG_PATH": "data/server.log",
}
DATASET_FOLDER = "data/dataset"
MANIFEST_VERSION = 1
_LEFT_BEHIND = frozenset({".cache", ".thumbnails"})  # rebuilt in the copy, never copied
_ACTIVE = ("pending", "running", "paused")


def write_bytes(path: Path, data: bytes) -> None:
    """Write ``path`` through a temp file beside it and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.replace(temp, path)
    except BaseException:
        if os.path.exists(temp):
            os.remove(temp)
        raise


def write_json(path: Path, content: Any) -> None:
    write_bytes(path, (json.dumps(content, indent=2, default=str) + "\n").encode("utf-8"))


def child_env(c: Comparison, db_path: Path) -> dict[str, str]:
    """The environment of the comparison's backend, trainer and fact child."""
    env = dict(os.environ)
    for name, relative in ISOLATED_PATHS.items():
        env[name] = str(c.work_dir / relative)
    env["MRLN_DB_PATH"] = str(db_path)
    env.update(
        PORT=str(c.port),
        MRLN_RESTART="1",  # no frontend start, no browser
        MRLN_GIT_REMOTE="",  # no self-update
        PYTHONDONTWRITEBYTECODE="1",  # the checkout is shared and only read
    )
    if c.allow_downloads:
        env.pop("HF_HUB_OFFLINE", None)
    else:
        env["HF_HUB_OFFLINE"] = "1"  # a missing weight fails the run instead of writing the shared cache
    if c.cpu_affinity:
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(len(c.cpu_affinity))
    return env


def dataset_digest(manifest: dict[str, str]) -> str:
    lines = "\n".join(f"{name}:{digest}" for name, digest in sorted(manifest.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _dataset_files(source: Path) -> list[str]:
    names = []
    for folder, dirs, files in os.walk(source):
        dirs[:] = sorted(d for d in dirs if d not in _LEFT_BEHIND)
        for name in sorted(files):
            names.append((Path(folder) / name).relative_to(source).as_posix())
    return names


def source_manifest(source: Path) -> dict[str, str]:
    """Relative name -> sha256 of every file the copy takes from ``source``."""
    return {name: _file_digest(source / name) for name in _dataset_files(source)}


def manifest_path(target: Path) -> Path:
    return target.parent / "manifest.json"


def copy_dataset(source: Path, target: Path) -> dict[str, str]:
    """Copy the dataset without its caches; -> its manifest. A copy made before is kept, but only
    while the source still holds the same content."""
    source, target = Path(source), Path(target)
    if not source.is_dir():
        raise RuntimeError(f"the dataset folder {source} does not exist")
    recorded_path = manifest_path(target)
    fresh = source_manifest(source)
    if recorded_path.exists():
        recorded = json.loads(recorded_path.read_text(encoding="utf-8"))["files"]
        if recorded != fresh:
            raise RuntimeError(
                f"the dataset changed since this comparison began: {source} no longer holds what "
                f"{recorded_path} recorded; start a comparison of another name"
            )
        return recorded
    for name in fresh:
        write_bytes(target / name, (source / name).read_bytes())
    write_json(recorded_path, {"version": MANIFEST_VERSION, "source": str(source), "files": fresh})
    return fresh


def read_manifest(target: Path) -> dict[str, str]:
    return json.loads(manifest_path(target).read_text(encoding="utf-8"))["files"]


def copy_database(source: Path, target: Path, dataset: str, dataset_path: Path) -> None:
    """A copy of the user's database (opened read-only) that knows only the comparison's dataset,
    at the copy's path, and has no job that could start or resume."""
    target.parent.mkdir(parents=True, exist_ok=True)
    reader = sqlite3.connect(f"{Path(source).resolve().as_uri()}?mode=ro", uri=True)
    temp = target.with_name(f".{target.name}.tmp")
    try:
        writer = sqlite3.connect(temp)
        try:
            reader.backup(writer)
            marks = ",".join("?" * len(_ACTIVE))
            writer.execute(f"update job_history set status='stopped' where status in ({marks})", _ACTIVE)
            writer.execute("delete from datasets where name != ?", (dataset,))
            found = writer.execute("update datasets set path=? where name=?", (str(dataset_path), dataset)).rowcount
            writer.commit()
        finally:
            writer.close()
        if found != 1:
            raise RuntimeError(f"the database {source} has no dataset named {dataset!r}")
        os.replace(temp, target)
    finally:
        reader.close()
        if temp.exists():
            temp.unlink()


def copy_settings(source: Path, target: Path) -> None:
    """A copy of the user's settings (the HF token, model overrides); none on a first launch."""
    if not Path(source).exists():
        return
    write_bytes(target, Path(source).read_bytes())
