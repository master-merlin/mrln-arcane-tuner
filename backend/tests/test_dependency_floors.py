"""LANE-134: the Dependabot bumps, pinned as floors, plus guards for what is held.

Reads the manifests from the repo root (anchored on ``__file__``), offline.
Every failing assertion starts with ``LANE-134: `` so the lane is named in
the output. A missing file is an assertion failure, never a collection error.
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REQ = "backend/requirements.txt"

# Spec order: accelerate first, so it is the first failure on main.
BACKEND_TAKE = [
    ("accelerate", "1.15.0"),
    ("tzdata", "2026.3"),
    ("anyio", "4.15.1"),
    ("idna", "3.20"),
    ("lazy_loader", "0.6"),
    ("lxml", "6.1.3"),
    ("multidict", "6.9.1"),
    ("networkx", "3.7"),
    ("nvidia-ml-py", "13.615.71"),
    ("onnxruntime-gpu", "1.30.0"),
    ("pandas", "3.0.6"),
    ("platformdirs", "4.11.14"),
    ("portalocker", "4.4.0"),
    ("propcache", "0.5.4"),
    ("PyMatting", "1.1.16"),
    ("regex", "2026.9.10"),
    ("rembg", "2.0.85"),
    ("ruff", "0.16.9"),
    ("starlette", "1.7.0"),
    ("tifffile", "2026.9.20"),
    ("timm", "1.0.30"),
    ("urllib3", "2.8.0"),
    ("uvicorn", "0.54.0"),
    ("wcwidth", "0.9.1"),
    ("yarl", "1.25.1"),
]

NG = "22.2.0"
FRONTEND_MANIFEST = {
    "@angular/common": NG,
    "@angular/compiler": NG,
    "@angular/core": NG,
    "@angular/forms": NG,
    "@angular/platform-browser": NG,
    "@angular/router": NG,
    "@angular/build": NG,
    "@angular/cli": NG,
    "@angular/compiler-cli": NG,
    "@codemirror/commands": "^6.11.1",
    "@codemirror/state": "^6.7.6",
    "@codemirror/view": "^6.43.13",
    "@lucide/angular": "1.48.0",
    "@playwright/test": "^1.63.0",
    "jsdom": "^30.1.1",
}

# Lock targets: the manifest packages (range prefix stripped) plus the lock-only ones.
FRONTEND_LOCK = {k: v.lstrip("^~") for k, v in FRONTEND_MANIFEST.items()}
FRONTEND_LOCK.update({"postcss": "8.5.28", "undici": "8.11.2", "ip-address": "10.7.2"})

TRUFFLEHOG_SHA = "4dd8831c5f12599465d4d45c3c447b4018a34c85"

# Hard-coded from main at 05404929.
HELD_KEEP = {
    "fsspec": "2025.10.0",
    "setuptools": "78.1.1",
    "torch": "2.12.1",
    "torchvision": "0.27.1",
    "torchaudio": "2.11.0",
    "torchao": "0.17.0",
    "diffusers": "0.40.0",
    "transformers": "5.14.1",
    "peft": "0.20.0",
}
HELD_COMMENTED = {
    "dill": "0.4.0",
    "mpmath": "1.3.0",
    "multiprocess": "0.70.18",
    "numpy": "2.3.5",
    "pydantic_core": "2.46.5",
    "tokenizers": "0.22.2",
    "tqdm": "4.67.1",
    "typing_extensions": "4.15.0",
}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _read(rel: str) -> str:
    path = REPO / rel
    assert path.is_file(), f"LANE-134: {rel} is missing at {path}"
    return path.read_text(encoding="utf-8")


def _json(rel: str) -> dict:
    try:
        return json.loads(_read(rel))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"LANE-134: {rel} is not valid JSON: {exc}") from exc


def _pins() -> dict[str, tuple[str, str]]:
    """normalised name -> (version, full line) for every ``name==version`` line."""
    out: dict[str, tuple[str, str]] = {}
    for raw in _read(REQ).splitlines():
        code = raw.split("#", 1)[0].split(";", 1)[0].strip()
        m = re.match(
            r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*==\s*(\S+)$", code
        )
        if m:
            out[_norm(m.group(1))] = (m.group(2), raw)
    return out


def _ver(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", text.split("-", 1)[0]))


def _lock_versions(name: str) -> list[str]:
    pkgs = _json("frontend/package-lock.json").get("packages", {})
    suffix = f"node_modules/{name}"
    return [
        str(info.get("version"))
        for path, info in pkgs.items()
        if path == suffix or path.endswith("/" + suffix)
    ]


def test_backend_pins_meet_lane134_floors():
    pins = _pins()
    bad = []
    for name, want in BACKEND_TAKE:
        have = pins.get(_norm(name), ("not pinned ==", ""))[0]
        if have != want:
            bad.append(
                f"LANE-134: {name} is pinned {have} in {REQ}; the floor is {want}"
            )
    assert not bad, bad[0] + (
        "\nall mismatches:\n" + "\n".join(bad) if len(bad) > 1 else ""
    )


def test_frontend_manifest_matches_lane134_targets():
    pkg = _json("frontend/package.json")
    have_all = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    bad = [
        f"LANE-134: {n} is {have_all.get(n, 'absent')} in frontend/package.json; the target is {want}"
        for n, want in FRONTEND_MANIFEST.items()
        if have_all.get(n) != want
    ]
    assert not bad, bad[0] + (
        "\nall mismatches:\n" + "\n".join(bad) if len(bad) > 1 else ""
    )


def test_frontend_lock_matches_lane134_targets():
    bad = []
    for name, want in FRONTEND_LOCK.items():
        have = _lock_versions(name)
        if not have or any(v != want for v in have):
            bad.append(
                f"LANE-134: {name} resolves {have or 'nothing'} in frontend/package-lock.json; the target is {want}"
            )
    assert not bad, bad[0] + (
        "\nall mismatches:\n" + "\n".join(bad) if len(bad) > 1 else ""
    )


def test_frontend_lock_meets_security_floors():
    bad = []
    for name, floor in (("undici", "8.10.2"), ("ip-address", "10.7.1")):
        have = _lock_versions(name)
        if not have:
            bad.append(f"LANE-134: {name} has no entry in frontend/package-lock.json")
        for v in have:
            if _ver(v) < _ver(floor):
                bad.append(
                    f"LANE-134: {name} {v} in frontend/package-lock.json is below the security floor {floor}"
                )
    assert not bad, "\n".join(bad)


def test_trufflehog_action_is_v3_97_9():
    text = _read(".github/workflows/gate.yml")
    want = f"trufflesecurity/trufflehog@{TRUFFLEHOG_SHA}"
    assert want in text, (
        f"LANE-134: .github/workflows/gate.yml does not reference {want} (v3.97.9)"
    )


def test_held_pins_stay_put():
    pins = _pins()
    bad = []
    for name, want in {**HELD_KEEP, **HELD_COMMENTED}.items():
        have, line = pins.get(_norm(name), ("absent", ""))
        if have != want:
            bad.append(f"LANE-134: held pin {name} is {have}; it must stay {want}")
        elif name in HELD_COMMENTED and "# HELD" not in line:
            bad.append(f"LANE-134: {name} lost its '# HELD' comment")
    assert not bad, "\n".join(bad)


def test_majors_not_taken():
    pkg = _json("frontend/package.json")
    dev = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    assert str(dev.get("typescript", "")).startswith("~6."), (
        f"LANE-134: typescript is {dev.get('typescript')}; it stays ~6.x (majors are a separate lane)"
    )
    assert str(dev.get("vitest", "")).startswith("^4."), (
        f"LANE-134: vitest is {dev.get('vitest')}; it stays ^4.x (majors are a separate lane)"
    )
