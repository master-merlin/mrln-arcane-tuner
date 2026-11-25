"""The comparison's own database, settings, server log and dataset copy."""
import hashlib
import os
import re
import sqlite3
import subprocess
import sys

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
    manifest = isolation.copy_dataset(source, target)
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
    manifest = isolation.copy_dataset(source, target)
    assert _tree(source) == before
    assert outside.read_bytes() == outside_before
    if linked:
        assert not (target / "linked.png").is_symlink()
        assert (target / "linked.png").read_bytes() == b"linked picture"
        assert manifest["linked.png"] == hashlib.sha256(b"linked picture").hexdigest()


def test_a_changed_source_dataset_is_refused_on_resume(tmp_path):
    source = _dataset(tmp_path / "Cars")
    target = tmp_path / "work" / "data" / "dataset"
    first = isolation.copy_dataset(source, target)
    assert isolation.copy_dataset(source, target) == first
    (source / "img1.png").write_bytes(b"another picture")
    with pytest.raises(RuntimeError, match="the dataset changed since this comparison began"):
        isolation.copy_dataset(source, target)


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
    env = isolation.child_env(make_comparison(cpu_affinity=[4, 5, 6, 7]), tmp_path / "run.db")
    assert (env["PORT"], env["MRLN_RESTART"], env["MRLN_GIT_REMOTE"]) == ("8001", "1", "")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert (env["OMP_NUM_THREADS"], env["MKL_NUM_THREADS"]) == ("4", "4")
    assert env["MRLN_DB_PATH"] == str(tmp_path / "run.db")


def test_the_dataset_digest_follows_content_not_order():
    a = {"a.png": "1" * 64, "b.png": "2" * 64}
    b = {"b.png": "2" * 64, "a.png": "1" * 64}
    assert isolation.dataset_digest(a) == isolation.dataset_digest(b)
    assert isolation.dataset_digest(a) != isolation.dataset_digest({**a, "b.png": "3" * 64})
