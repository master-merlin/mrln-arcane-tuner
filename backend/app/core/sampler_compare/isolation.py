"""The comparison's own stores: a database, settings, a server log and a dataset copy, all under
its work folder, so its backend and trainer never write into the user's.

Shared and only read: the code checkout, the Python environment, the model weights. Every file
written here goes through a fresh name and ``os.replace`` (``fresh_write``), so a link in the
target is replaced, never written through.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

from app.core.dataset.fresh_write import replace_with_fresh_file
from app.core.sampler_compare.plan import Comparison, ConfigError

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
_CHUNK_BYTES = 1 << 20
# Windows refuses a rename while a process holds the file (access denied 5, sharing violation 32):
# the bound ``safe_save.retry_file_lock`` waits out, which imports torch and so can't be used here.
_LOCK_WINERRORS = frozenset({5, 32})
_LOCK_BACKOFF_S = (0.2, 0.5, 1.0, 2.0, 4.0, 8.0)


def _write_fresh(path: Path, fill: Callable[[BinaryIO], None]) -> None:
    """``fill`` a fresh file beside ``path`` and move it onto ``path`` (``_retrying``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def write() -> None:
        with replace_with_fresh_file(path) as out:
            fill(out)

    _retrying(write)


def write_bytes(path: Path, data: bytes) -> None:
    _write_fresh(path, lambda out: out.write(data))


def write_json(path: Path, content: Any) -> None:
    write_bytes(path, (json.dumps(content, indent=2, default=str) + "\n").encode("utf-8"))


def read_only(db: Path) -> sqlite3.Connection:
    """The user's database at ``db``, opened read-only."""
    if not Path(db).exists():
        raise ConfigError(f"the database {db} does not exist")
    return sqlite3.connect(f"{Path(db).resolve().as_uri()}?mode=ro", uri=True)


def child_env(c: Comparison, db_path: Path, root: Path | None = None) -> dict[str, str]:
    """The environment of the comparison's backend, trainer and fact child; its stores sit under
    ``root`` (the work folder unless given)."""
    env = dict(os.environ)
    for name, relative in ISOLATED_PATHS.items():
        env[name] = str((root or c.work_dir) / relative)
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


def _copy_hashed(source: Path, target: Path) -> str:
    """Copy ``source`` to ``target`` and hash it in the same pass; -> its sha256."""
    digest = ""

    def fill(out: BinaryIO) -> None:
        nonlocal digest
        hashed = hashlib.sha256()
        with open(source, "rb") as reader:
            while chunk := reader.read(_CHUNK_BYTES):
                hashed.update(chunk)
                out.write(chunk)
        digest = hashed.hexdigest()

    _write_fresh(target, fill)
    return digest


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


def _real(path: Path) -> Path:
    return Path(os.path.realpath(path))


def _overlap(a: Path, b: Path) -> bool:
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def _is_link(path: Path) -> bool:
    return path.is_symlink() or os.path.isjunction(path)


def dataset_target(source: Path, work_dir: Path) -> Path:
    """The work folder's dataset copy for ``source``, once it is proven safe to write and clean:
    it resolves to the work folder's own dataset folder (no link on the way), holds no link, and
    neither the copy nor the work folder overlaps the source. Raises otherwise; writes nothing."""
    target = Path(work_dir) / DATASET_FOLDER
    real_source, real_work, real_target = _real(source), _real(work_dir), _real(target)
    if _overlap(real_source, real_work):
        raise RuntimeError(f"refusing the dataset copy: the dataset {source} and the work folder {work_dir} "
                           "overlap; give the comparison a work folder outside the dataset")
    if real_target != real_work / DATASET_FOLDER:
        raise RuntimeError(f"refusing the dataset copy: {target} resolves to {real_target}, outside the work "
                           "folder (a link on the way)")
    if target.exists():
        for folder, dirs, files in os.walk(target):
            for name in (*dirs, *files):
                if _is_link(Path(folder) / name):
                    raise RuntimeError(f"refusing the dataset copy: {Path(folder) / name} is a link; the copy "
                                       "holds plain files only")
    return target


def copy_dataset(source: Path, work_dir: Path, copy: bool = True) -> dict[str, str]:
    """Copy the dataset without its caches into the work folder (``dataset_target``); -> its
    manifest. A copy made before is kept, but only while the source still holds the same content.
    Without ``copy`` nothing is written: the manifest is the earlier copy's (checked the same way)
    or the source's own."""
    source = Path(source)
    if not source.is_dir():
        raise RuntimeError(f"the dataset folder {source} does not exist")
    target = dataset_target(source, work_dir)
    recorded_path = manifest_path(target)
    if recorded_path.exists():
        recorded = read_manifest(target)
        if source_manifest(source) != recorded:
            raise RuntimeError(
                f"the dataset changed since this comparison began: {source} no longer holds what "
                f"{recorded_path} recorded; start a comparison of another name"
            )
        return recorded
    if not copy:
        return source_manifest(source)
    fresh = {name: _copy_hashed(source / name, target / name) for name in _dataset_files(source)}
    _remove_all_but(target, set(fresh))  # an interrupted copy may have left files the source lacks
    write_json(recorded_path, {"version": MANIFEST_VERSION, "source": str(source), "files": fresh})
    return fresh


def _retrying(action: Callable[[], None]) -> None:
    """``action``, a Windows file lock waited out within ``_LOCK_BACKOFF_S``; any other error raises at once."""
    for delay in (*_LOCK_BACKOFF_S, None):
        try:
            action()
            return
        except PermissionError as exc:
            if delay is None or getattr(exc, "winerror", None) not in _LOCK_WINERRORS:
                raise
            time.sleep(delay)


def _remove_all_but(target: Path, keep: set[str]) -> None:
    """Delete every file under ``target`` not named in ``keep`` (caches included), and the folders
    that leaves empty. ``dataset_target`` proved the tree holds no link; each path is checked again
    to lie inside ``target`` before it goes."""
    real_target = _real(target)

    def remove(path: Path, action: Callable[[], None]) -> None:
        if _is_link(path) or not _real(path).is_relative_to(real_target) or _real(path) == real_target:
            raise RuntimeError(f"refusing to delete {path}: it is not a plain entry of the dataset copy {target}")
        _retrying(action)

    for folder, dirs, files in os.walk(target, topdown=False):
        here = Path(folder)
        for name in files:
            path = here / name
            if path.relative_to(target).as_posix() not in keep:
                remove(path, path.unlink)
        for name in dirs:
            path = here / name
            if not any(path.iterdir()):
                remove(path, lambda p=path: os.rmdir(p))


def read_manifest(target: Path) -> dict[str, str]:
    return json.loads(manifest_path(target).read_text(encoding="utf-8"))["files"]


def copy_database(source: Path, target: Path, dataset: str, dataset_path: Path) -> None:
    """A copy of the user's database (opened read-only) that knows only the comparison's dataset,
    at the copy's path, and has no job that could start or resume."""
    target.parent.mkdir(parents=True, exist_ok=True)
    reader = read_only(source)
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


def read_settings(source: Path) -> bytes | None:
    """The user's settings file as it is now (the HF token, model overrides); None on a first launch."""
    return Path(source).read_bytes() if Path(source).exists() else None


def write_settings(content: bytes | None, target: Path) -> None:
    """The comparison's settings copy holds ``content``; a first launch's settings are empty."""
    write_bytes(target, b"{}\n" if content is None else content)


def copy_settings(source: Path, target: Path) -> None:
    """A copy of the user's settings; none on a first launch."""
    content = read_settings(source)
    if content is not None:
        write_settings(content, target)


def _weights_fingerprint(folder: Path) -> str:
    """The sha256 over every file's relative name, size and modification time under ``folder``:
    weights swapped or rewritten in place change it, without reading gigabytes."""
    folder = Path(folder)
    if not folder.is_dir():
        return f"missing: {folder}"
    lines = []
    for here, dirs, files in os.walk(folder, followlinks=True):
        dirs.sort()
        for name in sorted(files):
            path = Path(here) / name
            stat = path.stat()
            lines.append(f"{path.relative_to(folder).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def model_source(settings: bytes | None, definition_id: str) -> dict[str, Any]:
    """Where the trainer loads ``definition_id``'s weights from under ``settings`` (the bytes of a
    settings file): ``ModelOverrideManager.resolve_effective_source``'s answer (a test pins the
    two agreeing; that module's settings are the process-wide file), plus a fingerprint of a local
    weight folder."""
    from app.core.schemas.model_overrides import ModelSettings, ModelSourceType

    try:  # a pydantic refusal is a ValueError too
        raw = json.loads(settings.decode("utf-8")) if settings else {}
        models = ModelSettings.model_validate((raw.get("models") if isinstance(raw, dict) else None) or {})
    except ValueError as exc:
        raise ConfigError(f"the settings file's model sources can't be read: {exc}") from None
    override = models.overrides.get(definition_id)
    if not override or override.source_type == ModelSourceType.HF_HUB:
        offline = models.global_offline_mode or (override.skip_update if override else False)
        return {"source_type": ModelSourceType.HF_HUB.value, "local_path": None, "local_files_only": offline,
                "weights": None}
    return {"source_type": override.source_type.value, "local_path": override.local_path,
            "local_files_only": False, "weights": _weights_fingerprint(Path(override.local_path))}
