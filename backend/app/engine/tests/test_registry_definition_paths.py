"""The registry's definition directories are anchored on the package.

A CWD-relative definitions path resolves only when the process happens to be
launched from ``backend\\``. Under a service wrapper, a different launcher or a
test runner it silently resolves to a directory that does not exist, the scan
finds nothing, and no error is raised — the registry just comes up short.
"""

import os
from pathlib import Path

import pytest

from app.engine.models import registry as registry_module
from app.engine.models.registry import ModelRegistry


@pytest.fixture
def registry_state():
    """Snapshot/restore the class-level registry so these tests cannot leak."""
    definitions = dict(ModelRegistry._definitions)
    paths = dict(ModelRegistry._paths)
    loaded = ModelRegistry._definitions_loaded
    yield ModelRegistry
    ModelRegistry._definitions = definitions
    ModelRegistry._paths = paths
    ModelRegistry._definitions_loaded = loaded


def test_central_definitions_dir_sits_next_to_the_registry_module():
    expected = Path(registry_module.__file__).resolve().parent / "definitions"
    assert Path(ModelRegistry._central_definitions_dir()).resolve() == expected


def test_a_definitions_dir_under_the_cwd_is_never_scanned(registry_state, tmp_path, monkeypatch):
    """Proves the negative: a decoy tree under the CWD must be ignored.

    This is the exact shape the old code fell for — it joined ``os.getcwd()``
    with the in-package relative path, so whatever happened to sit at that
    location under the working directory got loaded as if it were ours.
    """
    scanned: list[str] = []
    monkeypatch.setattr(
        ModelRegistry,
        "_scan_dir_for_definitions",
        classmethod(lambda cls, d: scanned.append(str(d))),
    )

    decoy = tmp_path / "app" / "engine" / "models" / "definitions"
    decoy.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    ModelRegistry._definitions_loaded = False
    ModelRegistry.load_definitions("")  # falsy → exercises the default branch

    assert not any(str(tmp_path) in s for s in scanned), (
        f"scanned a CWD-relative definitions dir: {scanned}"
    )
    # And the family definition dirs (the ones that actually hold our YAML)
    # were reached, so the assertion above is not passing on an empty scan.
    families = os.path.join("models", "families")
    assert any(families in s for s in scanned)
