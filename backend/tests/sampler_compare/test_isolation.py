"""The comparison's own database, settings, server log and dataset copy."""
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.sampler_compare import isolation, plan


def _tree(folder):
    """Every file under ``folder``: relative name -> (bytes digest, mtime)."""
    out = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            out[path.relative_to(folder).as_posix()] = (hashlib.sha256(path.read_bytes()).hexdigest(),
                                                        path.stat().st_mtime_ns)
    return out


def _dataset(folder):
    folder.mkdir(parents=True)
    for i in range(3):
        (folder / f"img{i}.png").write_bytes(bytes([i]) * 100)
        (folder / f"img{i}.txt").write_text(f"caption {i}", encoding="utf-8")
    (folder / ".cache" / "model" / "v1").mkdir(parents=True)
    (folder / ".cache" / "x").write_bytes(b"latent")
    (folder / ".thumbnails").mkdir()
    (folder / ".thumbnails" / "y").write_bytes(b"thumb")
    return folder


def _database(path, dataset_path):
    db = sqlite3.connect(path)
    db.execute("create table datasets (id text primary key, name text not null unique, path text not null)")
    db.execute("create table job_history (id text primary key, status text, config text, definition_id text)")
    db.execute("insert into datasets values ('d1', 'Cars', ?)", (str(dataset_path),))
    db.execute("insert into datasets values ('d2', 'Other', 'D:/elsewhere')")
    db.execute("insert into job_history values ('j1', 'running', '{}', 'moss-image-base')")
    db.execute("insert into job_history values ('j2', 'completed', '{}', 'moss-image-base')")
    db.commit()
    db.close()
    return path


def test_every_path_seam_the_backend_reads_is_set(make_comparison, tmp_path):
    seams = set()
    for path in (plan.BACKEND / "app").rglob("*.py"):
        if "tests" in path.relative_to(plan.BACKEND).parts:
            continue
        seams |= set(re.findall(r"MRLN_[A-Z_]+_PATH", path.read_text(encoding="utf-8-sig")))
    assert seams == {"MRLN_DB_PATH", "MRLN_SETTINGS_PATH", "MRLN_SERVER_LOG_PATH"}
    env = isolation.child_env(make_comparison(), tmp_path / "run.db")
    assert seams <= set(env)
    assert set(isolation.ISOLATED_PATHS) == seams


def test_a_child_resolves_every_store_inside_the_work_folder(make_comparison, tmp_path):
    c = make_comparison()
    env = isolation.child_env(c, c.work_dir / "data" / "db" / "run.db")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "from app.core.settings_manager import SettingsManager\n"
        "from app.core.db.engine import DatabaseEngine\n"
        "from app.core import logger\n"
        "print(SettingsManager().storage_file); print(DatabaseEngine().db_path); print(logger.SERVER_LOG_PATH)\n"
    )
    done = subprocess.run([sys.executable, "-c", code, str(plan.BACKEND)], env=env, cwd=tmp_path,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.strip().splitlines()[-3:]
    assert len(lines) == 3
    for line in lines:
        assert os.path.realpath(line).startswith(os.path.realpath(tmp_path)), line


def test_the_database_copy_knows_only_the_comparisons_dataset(tmp_path):
    source = _database(tmp_path / "user.db", tmp_path / "user-datasets" / "Cars")
    before = source.read_bytes()
    target = tmp_path / "work" / "data" / "db" / "run.db"
    copied = tmp_path / "work" / "data" / "dataset"
    isolation.copy_database(source, target, "Cars", copied)
    db = sqlite3.connect(target)
    try:
        assert db.execute("select name, path from datasets").fetchall() == [("Cars", str(copied))]
        active = db.execute("select count(*) from job_history where status in ('pending', 'running', 'paused')")
        assert active.fetchone()[0] == 0
    finally:
        db.close()
    assert source.read_bytes() == before


def test_a_database_without_the_comparisons_dataset_is_refused(tmp_path):
    source = _database(tmp_path / "user.db", tmp_path / "Cars")
    with pytest.raises(RuntimeError, match="Missing"):
        isolation.copy_database(source, tmp_path / "run.db", "Missing", tmp_path / "copy")


def test_the_dataset_copy_leaves_caches_and_thumbnails_behind(tmp_path):
    source = _dataset(tmp_path / "Cars")
    target = tmp_path / "work" / "data" / "dataset"
    manifest = isolation.copy_dataset(source, target.parents[1])
    assert not (target / ".cache").exists()
    assert not (target / ".thumbnails").exists()
    assert sorted(manifest) == ["img0.png", "img0.txt", "img1.png", "img1.txt", "img2.png", "img2.txt"]
    for name, digest in manifest.items():
        assert hashlib.sha256((target / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
    assert (target.parent / "manifest.json").exists()


def test_the_dataset_copy_never_writes_into_the_source(tmp_path):
    source = _dataset(tmp_path / "Cars")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"linked picture")
    linked = True
    try:
        os.symlink(outside, source / "linked.png")
    except (OSError, NotImplementedError):
        linked = False  # the platform refuses links to this user: the content case is not testable here
    before, outside_before = _tree(source), outside.read_bytes()
    target = tmp_path / "work" / "data" / "dataset"
    manifest = isolation.copy_dataset(source, target.parents[1])
    assert _tree(source) == before
    assert outside.read_bytes() == outside_before
    if linked:
        assert not (target / "linked.png").is_symlink()
        assert (target / "linked.png").read_bytes() == b"linked picture"
        assert manifest["linked.png"] == hashlib.sha256(b"linked picture").hexdigest()


def test_a_changed_source_dataset_is_refused_on_resume(tmp_path):
    source = _dataset(tmp_path / "Cars")
    target = tmp_path / "work" / "data" / "dataset"
    first = isolation.copy_dataset(source, target.parents[1])
    assert isolation.copy_dataset(source, target.parents[1]) == first
    (source / "img1.png").write_bytes(b"another picture")
    with pytest.raises(RuntimeError, match="the dataset changed since this comparison began"):
        isolation.copy_dataset(source, target.parents[1])


@pytest.mark.parametrize(("change", "named"), [
    ("edited", "changed img1.txt"), ("missing", "missing img1.txt"), ("extra", "extra new.txt"),
    ("extra cache-like", "extra sub/.cache2"),
])
@pytest.mark.parametrize("copy", [True, False])
def test_a_changed_copy_is_refused_on_resume(tmp_path, change, named, copy):
    source = _dataset(tmp_path / "Cars")
    work = tmp_path / "work"
    isolation.copy_dataset(source, work)
    target = work / isolation.DATASET_FOLDER
    if change == "edited":
        (target / "img1.txt").write_text("an edited caption", encoding="utf-8")
    elif change == "missing":
        (target / "img1.txt").unlink()
    elif change == "extra":
        (target / "new.txt").write_text("a caption the source lacks", encoding="utf-8")
    else:
        (target / "sub").mkdir()
        (target / "sub" / ".cache2").write_bytes(b"x")
    (target / ".cache").mkdir(exist_ok=True)
    (target / ".cache" / "latent").write_bytes(b"a cache the runs build is no change")
    before = _tree(target)
    with pytest.raises(RuntimeError, match=f"the dataset copy .* changed since it was made: {named}"):
        isolation.copy_dataset(source, work, copy=copy)
    assert _tree(target) == before


def test_a_copy_with_only_its_caches_added_is_kept(tmp_path):
    source = _dataset(tmp_path / "Cars")
    work = tmp_path / "work"
    first = isolation.copy_dataset(source, work)
    (work / isolation.DATASET_FOLDER / ".cache" / "m").mkdir(parents=True)
    (work / isolation.DATASET_FOLDER / ".cache" / "m" / "latent").write_bytes(b"latent")
    assert isolation.copy_dataset(source, work) == first
    assert isolation.copy_problem(work, first) is None


def test_a_copy_over_an_interrupted_one_holds_exactly_the_manifests_files(tmp_path):
    source = _dataset(tmp_path / "Cars")
    target = tmp_path / "work" / "data" / "dataset"
    (target / "gone" / ".cache").mkdir(parents=True)
    (target / "stale.png").write_bytes(b"left by an interrupted copy")
    (target / "gone" / "old.txt").write_text("an earlier caption", encoding="utf-8")
    (target / "gone" / ".cache" / "latent").write_bytes(b"latent")
    (target / "img0.png").write_bytes(b"half a picture")
    manifest = isolation.copy_dataset(source, target.parents[1])
    files = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
    assert files == sorted(manifest)
    assert not (target / "gone").exists()
    assert hashlib.sha256((target / "img0.png").read_bytes()).hexdigest() == manifest["img0.png"]
    assert isolation.read_manifest(target) == manifest


def test_a_missing_settings_file_is_a_first_launch_not_an_error(tmp_path):
    target = tmp_path / "work" / "data" / "settings.json"
    isolation.copy_settings(tmp_path / "missing.json", target)
    assert not target.exists()
    source = tmp_path / "settings.json"
    source.write_text('{"application": {"backend_port": 8000}}', encoding="utf-8")
    isolation.copy_settings(source, target)
    assert target.read_bytes() == source.read_bytes()


def test_downloads_are_off_unless_the_config_allows_them(make_comparison, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert isolation.child_env(make_comparison(), tmp_path / "run.db")["HF_HUB_OFFLINE"] == "1"
    assert "HF_HUB_OFFLINE" not in isolation.child_env(make_comparison(allow_downloads=True), tmp_path / "run.db")


def test_the_children_get_the_port_threads_and_no_bytecode(make_comparison, tmp_path):
    cpus = list(range(min(4, os.cpu_count() or 1)))  # CPUs every host has
    env = isolation.child_env(make_comparison(cpu_affinity=cpus), tmp_path / "run.db")
    assert (env["PORT"], env["MRLN_RESTART"], env["MRLN_GIT_REMOTE"]) == ("8001", "1", "")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert (env["OMP_NUM_THREADS"], env["MKL_NUM_THREADS"]) == (str(len(cpus)), str(len(cpus)))
    assert env["MRLN_DB_PATH"] == str(tmp_path / "run.db")


def test_the_stores_can_sit_outside_the_work_folder(make_comparison, tmp_path):
    c = make_comparison()
    env = isolation.child_env(c, tmp_path / "scratch" / "facts.db", root=tmp_path / "scratch")
    for name in isolation.ISOLATED_PATHS:
        assert Path(env[name]).is_relative_to(tmp_path / "scratch")
        assert not Path(env[name]).is_relative_to(c.work_dir)


def _locked() -> PermissionError:
    error = PermissionError(13, "locked")
    error.winerror = 5
    return error


def test_a_write_waits_out_a_windows_file_lock(tmp_path, monkeypatch):
    real_replace, errors, sleeps = os.replace, [_locked()], []

    def replace(src, dst):
        if errors:
            raise errors.pop()
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(isolation.time, "sleep", sleeps.append)
    isolation.write_bytes(tmp_path / "state.json", b"{}")
    assert (tmp_path / "state.json").read_bytes() == b"{}"
    assert sleeps == [0.2]
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_a_write_that_fails_otherwise_raises_at_once(tmp_path, monkeypatch):
    def replace(src, dst):
        raise PermissionError(13, "denied")

    sleeps = []
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(isolation.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError):
        isolation.write_bytes(tmp_path / "state.json", b"{}")
    assert sleeps == [] and list(tmp_path.iterdir()) == []


def _folder_link(link, target):
    """A folder link at ``link`` to ``target`` (a junction on Windows), or the test is skipped."""
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
        else:
            os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"the platform refuses a folder link here: {exc}")


@pytest.mark.parametrize("where", ["data/dataset", "data/dataset/Cars", "data", "runs/Cars", "."])
@pytest.mark.parametrize("copy", [True, False])
def test_a_source_inside_the_work_folder_is_refused_and_left_whole(tmp_path, where, copy):
    work = tmp_path / "work"
    source = _dataset(work / where)
    before = _tree(source)
    with pytest.raises(RuntimeError, match="refusing the dataset copy"):
        isolation.copy_dataset(source, work, copy=copy)
    assert _tree(source) == before


@pytest.mark.parametrize("copy", [True, False])
def test_a_work_folder_inside_the_source_is_refused_and_the_source_left_whole(tmp_path, copy):
    source = _dataset(tmp_path / "Cars")
    before = _tree(source)
    with pytest.raises(RuntimeError, match="refusing the dataset copy"):
        isolation.copy_dataset(source, source / "work", copy=copy)
    assert _tree(source) == before


def test_a_dataset_folder_linked_to_the_source_is_refused_and_the_source_left_whole(tmp_path):
    source = _dataset(tmp_path / "Cars")
    work = tmp_path / "work"
    _folder_link(work / isolation.DATASET_FOLDER, source)
    before = _tree(source)
    with pytest.raises(RuntimeError, match="refusing the dataset copy"):
        isolation.copy_dataset(source, work)
    assert _tree(source) == before


def test_a_data_folder_linked_out_of_the_work_folder_is_refused(tmp_path):
    user = tmp_path / "user"
    source = _dataset(user / "Cars")
    (user / "dataset").mkdir()
    (user / "dataset" / "sibling.png").write_bytes(b"a picture of the user's")
    work = tmp_path / "work"
    _folder_link(work / "data", user)
    before, sibling = _tree(source), _tree(user / "dataset")
    with pytest.raises(RuntimeError, match="refusing the dataset copy"):
        isolation.copy_dataset(source, work)
    assert _tree(source) == before and _tree(user / "dataset") == sibling


def test_a_link_inside_the_dataset_copy_is_refused_and_never_followed(tmp_path):
    source = _dataset(tmp_path / "Cars")
    work = tmp_path / "work"
    (work / isolation.DATASET_FOLDER).mkdir(parents=True)
    _folder_link(work / isolation.DATASET_FOLDER / "gone", source)
    before = _tree(source)
    with pytest.raises(RuntimeError, match="refusing the dataset copy"):
        isolation.copy_dataset(source, work)
    assert _tree(source) == before


@pytest.mark.parametrize("models", [
    None,
    {},
    {"global_offline_mode": True},
    {"overrides": {"moss-image-base": {"source_type": "hf_hub", "skip_update": True}}},
    {"overrides": {"moss-image-base": {"source_type": "local_diffusers", "local_path": "LOCAL"}}},
    {"overrides": {"moss-image-base": {"source_type": "local_safetensors", "local_path": "LOCAL"}}},
    {"overrides": {"other-model": {"source_type": "local_diffusers", "local_path": "LOCAL"}}},
])
def test_the_model_source_is_the_products_own_resolution(tmp_path, monkeypatch, models):
    from app.core.schemas.model_overrides import ModelSettings
    from app.engine.utils.model_override_manager import ModelOverrideManager

    local = tmp_path / "weights"
    local.mkdir()
    (local / "model.safetensors").write_bytes(b"weights")
    models = json.loads(json.dumps(models).replace("LOCAL", local.as_posix())) if models is not None else None
    settings = None if models is None else json.dumps({"models": models}).encode("utf-8")
    monkeypatch.setattr(ModelOverrideManager, "_load", staticmethod(lambda: ModelSettings.model_validate(models or {})))
    source_type, local_path, local_files_only = ModelOverrideManager.resolve_effective_source("moss-image-base")
    resolved = isolation.model_source(settings, "moss-image-base")
    assert (resolved["source_type"], resolved["local_path"], resolved["local_files_only"]) == (
        source_type.value, local_path, local_files_only)
    assert (resolved["weights"] is None) == (local_path is None)


def test_the_model_source_follows_the_weights_in_a_local_folder(tmp_path):
    local = tmp_path / "weights"
    (local / "transformer").mkdir(parents=True)
    (local / "transformer" / "model.safetensors").write_bytes(b"weights A")
    settings = json.dumps({"models": {"overrides": {"m": {"source_type": "local_diffusers",
                                                           "local_path": str(local)}}}}).encode("utf-8")
    first = isolation.model_source(settings, "m")
    assert isolation.model_source(settings, "m") == first
    (local / "transformer" / "model.safetensors").write_bytes(b"weights B, longer")
    assert isolation.model_source(settings, "m")["weights"] != first["weights"]


def test_the_dataset_digest_follows_content_not_order():
    a = {"a.png": "1" * 64, "b.png": "2" * 64}
    b = {"b.png": "2" * 64, "a.png": "1" * 64}
    assert isolation.dataset_digest(a) == isolation.dataset_digest(b)
    assert isolation.dataset_digest(a) != isolation.dataset_digest({**a, "b.png": "3" * 64})
