"""LANE-135: container image floors (setuptools, pip, npm) and the CVE register.

Reads Dockerfile, backend/requirements.txt, the dated Docker Scout snapshot and
``_harness/data/container-cve-dispositions.json`` from the repo root (anchored on
``__file__``), offline. Every failing assertion starts with ``LANE-135: ``; a
missing file is an assertion failure, never a collection error.

Register shape: ``{"findings": [{"cve", "purl", "disposition", "reason", "revisit"}]}``
with one row per (cve, purl) of the snapshot; disposition is one of
REBUILD/PIN/PURGE/HELD/DISMISS; HELD and DISMISS rows need non-empty
``reason`` and ``revisit`` (a lane id or a condition).
"""

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO / "Dockerfile"
REQ = REPO / "backend" / "requirements.txt"
SNAPSHOT = REPO / ".agent" / "workdir" / "lane-deps" / "evidence" / "scout-cu128-2026-10-04.findings.csv"
REGISTER = REPO / "_harness" / "data" / "container-cve-dispositions.json"
DISPOSITIONS = {"REBUILD", "PIN", "PURGE", "HELD", "DISMISS"}


def _read(path: Path) -> str:
    assert path.is_file(), f"LANE-135: {path.relative_to(REPO)} is missing"
    return path.read_text(encoding="utf-8")


def _ver(text: str) -> tuple:
    return tuple(int(p) for p in text.split("."))


def _dockerfile_code() -> str:
    """Dockerfile with comment-only lines dropped (a comment is not an install)."""
    return "\n".join(
        ln for ln in _read(DOCKERFILE).splitlines() if not ln.lstrip().startswith("#")
    )


def test_setuptools_pin_is_81_in_both_places():
    df = re.search(r"setuptools==(\d+(?:\.\d+)*)", _dockerfile_code())
    assert df, "LANE-135: Dockerfile has no setuptools== pin"
    assert df.group(1) == "81.0.0", (
        f"LANE-135: setuptools is pinned {df.group(1)} in Dockerfile; "
        "the floor is 81.0.0 (torch ceiling <82)"
    )
    req = re.search(r"^setuptools==(\d+(?:\.\d+)*)", _read(REQ), re.M)
    assert req, "LANE-135: backend/requirements.txt has no setuptools== pin"
    assert req.group(1) == "81.0.0", (
        f"LANE-135: setuptools is pinned {req.group(1)} in requirements.txt; "
        "the floor is 81.0.0 (torch ceiling <82)"
    )
    assert _ver(req.group(1)) < (82,), "LANE-135: setuptools must stay < 82 (torch)"


def test_runtime_pip_floor():
    code = _dockerfile_code()
    runtime = code.find(" AS runtime")
    apt = code.find("apt-get upgrade")
    m = re.search(r"\bpip==(\d+(?:\.\d+)*)", code)
    assert m, "LANE-135: the Dockerfile installs no pip==<version> in the runtime stage"
    assert _ver(m.group(1)) >= (26, 2, 0), (
        f"LANE-135: runtime pip is {m.group(1)}; the floor is 26.2.0 (pin 26.2.1)"
    )
    assert m.group(1) == "26.2.1", f"LANE-135: runtime pip is {m.group(1)}; the pin is 26.2.1"
    assert runtime != -1 and apt != -1 and m.start() > runtime and m.start() > apt, (
        "LANE-135: pip==26.2.1 must be installed in the runtime stage, after the apt layer"
    )


def test_runtime_npm_floor():
    code = _dockerfile_code()
    node = code.find("nodesource")
    m = re.search(r"\bnpm@(\d+(?:\.\d+)*)", code)
    assert m, "LANE-135: the Dockerfile installs no npm@<version> after the nodesource layer"
    assert _ver(m.group(1)) >= (11, 21, 0), (
        f"LANE-135: npm is {m.group(1)}; the floor is 11.21.0"
    )
    assert m.group(1) == "11.21.0", f"LANE-135: npm is {m.group(1)}; the pin is 11.21.0"
    assert node != -1 and m.start() > node, (
        "LANE-135: npm@11.21.0 must be installed after the nodesource layer"
    )


def test_every_scout_finding_has_one_disposition():
    assert SNAPSHOT.is_file(), f"LANE-135: Scout snapshot {SNAPSHOT.name} is missing"
    with SNAPSHOT.open(encoding="utf-8", newline="") as fh:
        expected = {(r["cve"], r["purl"]) for r in csv.DictReader(fh)}
    assert expected, "LANE-135: the Scout snapshot has no findings"
    assert REGISTER.is_file(), (
        "LANE-135: the register _harness/data/container-cve-dispositions.json is missing; "
        f"it must list all {len(expected)} CVE x package findings of the 2026-10-04 snapshot"
    )
    rows = json.loads(REGISTER.read_text(encoding="utf-8")).get("findings", [])
    keys = [(r.get("cve"), r.get("purl")) for r in rows]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, f"LANE-135: findings listed more than once: {dupes[:3]}"
    missing = sorted(expected - set(keys))
    extra = sorted(set(keys) - expected)
    assert not missing, f"LANE-135: {len(missing)} findings have no disposition, e.g. {missing[:3]}"
    assert not extra, f"LANE-135: {len(extra)} register rows are not in the snapshot, e.g. {extra[:3]}"
    for r in rows:
        key = (r["cve"], r["purl"])
        assert r.get("disposition") in DISPOSITIONS, (
            f"LANE-135: {key} has disposition {r.get('disposition')!r}, not one of {sorted(DISPOSITIONS)}"
        )
        if r["disposition"] in {"HELD", "DISMISS"}:
            for field in ("reason", "revisit"):
                assert str(r.get(field, "")).strip(), (
                    f"LANE-135: {key} is {r['disposition']} with an empty {field}"
                )


def test_cuda_base_minor_unchanged():
    text = _read(DOCKERFILE)
    base = re.search(r"^ARG CUDA_BASE=(\S+)", text, re.M)
    assert base and base.group(1).startswith("12.8."), "LANE-135: ARG CUDA_BASE must stay 12.8.x (LANE-140 owns CUDA 13)"
    alt = re.search(r"--build-arg CUDA_BASE=(\S+)", text)
    assert alt and alt.group(1).startswith("12.6."), "LANE-135: the cu126 build-arg must stay 12.6.x"


def test_torch_trio_unchanged():
    text = _dockerfile_code()
    for pin in ("torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0", "torchao==0.17.0"):
        assert re.search(rf"(?<![\w-]){re.escape(pin)}\b", text), f"LANE-135: Dockerfile must keep {pin}"
