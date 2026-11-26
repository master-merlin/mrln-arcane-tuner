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
import re
import secrets
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

from app.core.dataset.fresh_write import replace_with_fresh_file
from app.core.dataset.media_types import IMAGE_EXTENSION_PREFERENCE
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
METADATA_SNAPSHOT = "data/metadata.db"
OWNER_RECORD = "data/dataset-owner.json"  # which folder the comparison made its dataset copy in
MANIFEST_VERSION = 1
_LEFT_BEHIND = frozenset({".cache", ".thumbnails"})  # rebuilt in the copy, never copied
_RECREATED_FOLDER = "masked"  # copied, but a job may recreate its pictures in the copy (``_recreated``)
_MASKS_FOLDER = "masks"
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
        if not _owned(work_dir, target) and any(target.iterdir()):
            raise RuntimeError(f"refusing the dataset copy: {target} holds files and was not made by this "
                               "comparison; give the comparison a work folder of its own")
    return target


def _folder_id(path: Path) -> list[int]:
    """What tells one folder from another put in its place: its device and file number."""
    stat = os.stat(path)
    return [stat.st_dev, stat.st_ino]


def _owned(work_dir: Path, target: Path) -> bool:
    """Whether ``target`` is the dataset folder this comparison made (``_claim``)."""
    path = Path(work_dir) / OWNER_RECORD
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(record, dict) and record.get("folder") == _folder_id(target)


def _claim(work_dir: Path, target: Path) -> None:
    """Make the dataset folder (or take an empty one a crash left) and record it as this
    comparison's own; only an owned folder is ever written into or cleaned."""
    try:
        target.mkdir(parents=True)  # exclusive: an existing folder is taken only while empty
    except FileExistsError:
        if any(target.iterdir()):
            raise RuntimeError(f"refusing the dataset copy: {target} holds files and was not made by this "
                               "comparison") from None
    write_json(Path(work_dir) / OWNER_RECORD,
               {"version": MANIFEST_VERSION, "folder": _folder_id(target), "token": secrets.token_hex(16)})


def copy_dataset(source: Path, work_dir: Path, copy: bool = True, recreates_masks: bool = False) -> dict[str, str]:
    """Copy the dataset without its caches into the work folder (``dataset_target``); -> its
    manifest. Only a folder this comparison made is written into or cleaned (``_claim``). A copy
    made before is kept, but only while the source still holds the same content. Without ``copy``
    nothing is written: the manifest is the earlier copy's (checked the same way) or the source's
    own."""
    source = Path(source)
    if not source.is_dir():
        raise RuntimeError(f"the dataset folder {source} does not exist")
    target = dataset_target(source, work_dir)
    if copy and not (target.exists() and _owned(work_dir, target)):
        _claim(work_dir, target)
    recorded_path = manifest_path(target)
    if recorded_path.exists():
        recorded = read_manifest(target)
        problem = copy_problem(work_dir, recorded, recreates_masks)
        if problem:
            raise RuntimeError(problem)
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


def _recreated(name: str, manifest: dict[str, str]) -> bool:
    """A picture a job with ``recreate_masks`` writes into the dataset as it trains, and so every
    arm makes again the same way from checked inputs: ``MaskingService.mass_apply`` writes
    ``masked/<stem>.jpg`` for each ``masks/<stem>.png`` that has a picture ``<stem>.<ext>``."""
    prefix = f"{_RECREATED_FOLDER}/"
    if not name.startswith(prefix) or not name.endswith(".jpg") or "/" in name[len(prefix):]:
        return False
    stem = name[len(prefix):-len(".jpg")]
    has_mask = any(n.startswith(f"{_MASKS_FOLDER}/") and "/" not in n[len(_MASKS_FOLDER) + 1:]
                   and n.lower().endswith(".png") and n[len(_MASKS_FOLDER) + 1:-len(".png")] == stem
                   for n in manifest)
    return has_mask and any(f"{stem}{ext}" in manifest for ext in IMAGE_EXTENSION_PREFERENCE)


def copy_problem(work_dir: Path, manifest: dict[str, str], recreates_masks: bool = False) -> str | None:
    """Why the work folder's dataset copy no longer holds exactly ``manifest`` (a file changed,
    missing or added; the caches the runs build aside and, for a job that ``recreates_masks``
    (``plan.recreates_masks``), the masked pictures it makes again (``_recreated``), though one may
    not go missing), or None. Nothing is written or deleted:
    arms that trained earlier read the copy as it was, so a changed copy is refused, never redone."""
    target = Path(work_dir) / DATASET_FOLDER
    if not target.is_dir():
        return f"the dataset copy {target} is gone"
    actual = source_manifest(target)
    def remade(name: str) -> bool:
        return recreates_masks and _recreated(name, manifest)

    differences = ([f"changed {n}" for n in sorted(manifest)
                    if n in actual and actual[n] != manifest[n] and not remade(n)]
                   + [f"missing {n}" for n in sorted(manifest) if n not in actual]
                   + [f"extra {n}" for n in sorted(actual) if n not in manifest and not remade(n)])
    if not differences:
        return None
    shown = ", ".join(differences[:10]) + (f" and {len(differences) - 10} more" if len(differences) > 10 else "")
    return (f"the dataset copy {target} changed since it was made: {shown}; "
            "start a comparison of another name")


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


def snapshot_database(source: Path, work_dir: Path, dataset: str) -> Path:
    """The comparison's one copy of the user's database (``copy_database``), taken at its first
    launch and never replaced: every run's database is copied from it, so the live backend's later
    changes (a picture switched off, a caption edited) reach no arm. -> its path."""
    path = Path(work_dir) / METADATA_SNAPSHOT
    if not path.exists():
        copy_database(source, path, dataset, Path(work_dir) / DATASET_FOLDER)
    return path


def dataset_metadata_digest(db: Path, dataset: str) -> str:
    """The sha256 of ``dataset``'s rows in ``db`` (its ``datasets`` row without its path, and every
    ``media_items`` row of it without its row id): what the backend serves the trainer about the
    dataset beside its files."""
    conn = read_only(db)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute("select * from datasets where name=?", (dataset,)).fetchone()
        if row is None:
            raise RuntimeError(f"the database {db} has no dataset named {dataset!r}")
        content: dict[str, Any] = {"dataset": {k: row[k] for k in row.keys() if k != "path"}}
        tables = {name for (name,) in conn.execute("select name from sqlite_master where type='table'")}
        if "media_items" in tables:
            items = conn.execute("select * from media_items where dataset_id=? order by rel_path", (row["id"],))
            content["media_items"] = [{k: item[k] for k in item.keys() if k != "id"} for item in items]
    finally:
        conn.close()
    return hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode("utf-8")).hexdigest()


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


def hub_cache() -> Path:
    """The Hugging Face cache the comparison's children load from (they inherit this environment)."""
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _hub_revision(path: str, cache: Path) -> str:
    """The commit of the cached snapshot a ``huggingface:<repo>[@<revision>][:<file>]`` path loads
    offline (``ModelPathResolver._resolve_hf``'s parsing): a pinned commit, else what
    ``refs/<revision or main>`` names, in the hub's cache layout; read from disk, never the network."""
    repo = path[len("huggingface:"):].split(":")[0]
    repo, revision = repo.split("@", 1) if "@" in repo else (repo, None)
    folder = cache / f"models--{repo.replace('/', '--')}"
    commit = revision if revision and _COMMIT.match(revision) else None
    if commit is None:
        ref = folder / "refs" / (revision or "main")
        try:
            commit = ref.read_text(encoding="utf-8").strip()
        except OSError:
            commit = None
    if not commit or not (folder / "snapshots" / commit).is_dir():
        return f"not cached: {repo}@{revision or 'main'}"
    return commit


def model_source(settings: bytes | None, definition_id: str, hub_paths: list[str] | tuple = (),
                 cache: Path | None = None) -> dict[str, Any]:
    """Where the trainer loads ``definition_id``'s weights from under ``settings`` (the bytes of a
    settings file): ``ModelOverrideManager.resolve_effective_source``'s answer (a test pins the
    two agreeing; that module's settings are the process-wide file), a fingerprint of a local
    weight folder, and the cached snapshot commit of every hub component (``hub_paths``, from the
    product's facts) in ``cache`` (``hub_cache``)."""
    from app.core.schemas.model_overrides import ModelSettings, ModelSourceType

    try:  # a pydantic refusal is a ValueError too
        raw = json.loads(settings.decode("utf-8")) if settings else {}
        models = ModelSettings.model_validate((raw.get("models") if isinstance(raw, dict) else None) or {})
    except ValueError as exc:
        raise ConfigError(f"the settings file's model sources can't be read: {exc}") from None
    cache = hub_cache() if cache is None else Path(cache)
    hub = {"hub_paths": list(hub_paths), "revision": {path: _hub_revision(path, cache) for path in hub_paths}}
    override = models.overrides.get(definition_id)
    if not override or override.source_type == ModelSourceType.HF_HUB:
        offline = models.global_offline_mode or (override.skip_update if override else False)
        return {"source_type": ModelSourceType.HF_HUB.value, "local_path": None, "local_files_only": offline,
                "weights": None, **hub}
    return {"source_type": override.source_type.value, "local_path": override.local_path,
            "local_files_only": False, "weights": _weights_fingerprint(Path(override.local_path)), **hub}


def model_source_problem(settings_path: Path, definition_id: str, recorded: dict[str, Any]) -> str | None:
    """Why the weights a run would load now are not the ``recorded`` model source, or None."""
    now = json.loads(json.dumps(model_source(read_settings(settings_path), definition_id,
                                             recorded.get("hub_paths") or ())))
    changed = [f"model_source.{key}" for key in sorted(set(now) | set(recorded)) if now.get(key) != recorded.get(key)]
    return f"the model's weights changed since the comparison began: {', '.join(changed)}" if changed else None
