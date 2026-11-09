"""Choose, install and verify the torch stack this machine can run.

Every installer calls this file; the versions live only in ``torch-stack.env`` next to it.
Standard library only: it runs before any dependency is installed.

    python torch_stack.py select                  # print the chosen set's name
    python torch_stack.py install --python <py>   # pip-install the chosen set into that interpreter
    python torch_stack.py filter <req> <out> [--also NAME ...]
                                                  # requirements minus the stack (and NAMEs)
    python torch_stack.py verify --python <py>    # the installed stack equals the chosen set

The choice:
- ``MRLN_TORCH_INDEX`` names a set (or ``cpu``). It chooses the set but never skips its checks.
- Otherwise the highest set in ``SETS`` that fits. A set fits when the driver meets its
  ``DRIVER_MIN`` and EVERY GPU ``nvidia-smi`` lists has a compatible architecture in its
  ``ARCHS``: one venv serves every GPU on the machine.
- No ``nvidia-smi`` at all: the CPU set, said loudly.
- Nothing fits: exit 2 with the table, before anything is installed.

``MRLN_TORCH_NVIDIA_SMI`` names a file holding canned ``nvidia-smi
--query-gpu=driver_version,compute_cap --format=csv,noheader`` output, used instead of
running it (tests, and simulating another machine).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
STACK_FILE = HERE / "torch-stack.cfg"
INDEX_URL = "https://download.pytorch.org/whl/{name}"
CPU = "cpu"

#: The distributions this file owns. No installer's bulk requirements step may install them.
STACK_DISTRIBUTIONS = ("torch", "torchvision", "torchaudio", "torchao", "triton", "triton-windows")

REFUSED = 2


class Refused(Exception):
    """No set can be installed here; the message says why and shows the table."""


@dataclass(frozen=True)
class StackSet:
    name: str
    torch: str
    torchvision: str
    torchaudio: str
    torchao: str
    triton: str
    triton_windows: str
    driver_min: int | None  # None: the CPU set
    archs: tuple[tuple[int, int], ...]

    def pins(self, platform: str = sys.platform) -> dict[str, str]:
        """distribution -> version, as installed on ``platform`` (triton is per-OS)."""
        pins = {
            "torch": self.torch,
            "torchvision": self.torchvision,
            "torchaudio": self.torchaudio,
            "torchao": self.torchao,
        }
        if platform == "win32":
            pins["triton-windows"] = self.triton_windows
        elif platform.startswith("linux"):
            pins["triton"] = self.triton
        return pins

    def fits_arch(self, cc: tuple[int, int]) -> bool:
        major, minor = cc
        return any(a_major == major and a_minor <= minor for a_major, a_minor in self.archs)


@dataclass(frozen=True)
class Gpu:
    driver: str
    cc: tuple[int, int]

    @property
    def driver_major(self) -> int:
        return int(self.driver.split(".")[0])


def read_stack(path: Path = STACK_FILE) -> tuple[str, dict[str, StackSet]]:
    """(default set name, {name: set}) from ``torch-stack.env``; the CPU set included."""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"{path.name}: not KEY=VALUE: {raw!r}")
        values[key.strip()] = value.strip()

    def arch(token: str) -> tuple[int, int]:
        number = int(token)
        return number // 10, number % 10

    sets: dict[str, StackSet] = {}
    for name in values["SETS"].split():
        p = name.upper() + "_"
        sets[name] = StackSet(
            name=name,
            torch=values[p + "TORCH"],
            torchvision=values[p + "TORCHVISION"],
            torchaudio=values[p + "TORCHAUDIO"],
            torchao=values[p + "TORCHAO"],
            triton=values[p + "TRITON"],
            triton_windows=values[p + "TRITON_WINDOWS"],
            driver_min=int(values[p + "DRIVER_MIN"]),
            archs=tuple(arch(t) for t in values[p + "ARCHS"].split()),
        )
    default = values["DEFAULT"]
    if default not in sets:
        raise ValueError(f"{path.name}: DEFAULT={default} is not in SETS")
    base = sets[default]
    sets[CPU] = StackSet(CPU, base.torch, base.torchvision, base.torchaudio, base.torchao, base.triton,
                         base.triton_windows, driver_min=None, archs=())
    return default, sets


def parse_gpus(text: str) -> list[Gpu]:
    gpus = []
    for line in text.splitlines():
        if not line.strip():
            continue
        driver, _, cc = (part.strip() for part in line.partition(","))
        major, _, minor = cc.partition(".")
        gpus.append(Gpu(driver=driver, cc=(int(major), int(minor or 0))))
    return gpus


def detect_gpus() -> list[Gpu] | None:
    """The GPUs ``nvidia-smi`` lists; None when there is no ``nvidia-smi``."""
    canned = os.environ.get("MRLN_TORCH_NVIDIA_SMI")
    if canned:
        return parse_gpus(Path(canned).read_text(encoding="utf-8"))
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
    except FileNotFoundError:
        return None
    if done.returncode != 0:
        raise Refused(
            f"nvidia-smi failed (exit {done.returncode}): {done.stderr.strip() or done.stdout.strip()}\n"
            f"Fix the NVIDIA driver, or set MRLN_TORCH_INDEX=cpu to install the CPU build."
        )
    return parse_gpus(done.stdout)


def why_not(stack: StackSet, gpus: list[Gpu]) -> str | None:
    """None when ``stack`` fits every GPU, else the reason it does not."""
    if stack.driver_min is None:
        return None
    driver = min(g.driver_major for g in gpus)
    if driver < stack.driver_min:
        return f"driver {driver} is below R{stack.driver_min}"
    missing = [g for g in gpus if not stack.fits_arch(g.cc)]
    if missing:
        return "no kernels for compute capability " + ", ".join(f"{g.cc[0]}.{g.cc[1]}" for g in missing)
    return None


def table(sets: dict[str, StackSet], gpus: list[Gpu]) -> str:
    rows = ["set     torch    driver   architectures (compute capability)"]
    for s in sets.values():
        if s.driver_min is None:
            continue
        archs = " ".join(f"{a}.{b}" for a, b in s.archs)
        rows.append(f"{s.name:<7} {s.torch:<8} R{s.driver_min}+   {archs}")
    found = ", ".join(f"cc {g.cc[0]}.{g.cc[1]} (driver {g.driver})" for g in gpus)
    rows.append(f"this machine: {found}")
    return "\n".join(rows)


def choose(sets: dict[str, StackSet], gpus: list[Gpu] | None, override: str | None) -> tuple[StackSet, str]:
    """(the set, how it was chosen), or Refused."""
    if override:
        if override not in sets:
            raise Refused(f"MRLN_TORCH_INDEX={override} is not a set; the sets are: {', '.join(sets)}")
        chosen = sets[override]
        if chosen.driver_min is None:
            return chosen, f"MRLN_TORCH_INDEX={override}"
        if gpus is None:
            raise Refused(f"MRLN_TORCH_INDEX={override} needs an NVIDIA GPU, and there is no nvidia-smi")
        reason = why_not(chosen, gpus)
        if reason:
            raise Refused(f"MRLN_TORCH_INDEX={override} cannot run here: {reason}\n{table(sets, gpus)}")
        return chosen, f"MRLN_TORCH_INDEX={override}"
    if gpus is None:
        return sets[CPU], "no nvidia-smi found: installing the CPU build, without GPU support"
    if not gpus:
        raise Refused("nvidia-smi lists no GPU. Set MRLN_TORCH_INDEX=cpu to install the CPU build.")
    reasons = []
    for s in sets.values():
        if s.driver_min is None:
            continue
        reason = why_not(s, gpus)
        if reason is None:
            return s, "fits the driver and every GPU"
        reasons.append(f"{s.name}: {reason}")
    raise Refused("no torch build fits this machine:\n  " + "\n  ".join(reasons) + "\n" + table(sets, gpus))


def install_commands(stack: StackSet, python: str, platform: str = sys.platform) -> list[list[str]]:
    """The pip commands that install ``stack`` into ``python``."""
    index = INDEX_URL.format(name=stack.name)
    pip = [python, "-m", "pip", "install"]
    commands = [
        pip + [f"torch=={stack.torch}", f"torchvision=={stack.torchvision}", "--index-url", index],
        # --no-deps: torch is already in place; it keeps a torchaudio pin from ever moving torch.
        pip + ["--no-deps", f"torchaudio=={stack.torchaudio}", "--index-url", index],
        pip + ["--no-deps", f"torchao=={stack.torchao}"] + _torchao_index(stack, platform),
    ]
    pins = stack.pins(platform)
    for name in ("triton-windows", "triton"):
        if name in pins:
            commands.append(pip + [f"{name}=={pins[name]}"])
    return commands


def _torchao_from_index(platform: str) -> bool:
    """Linux takes torchao from the set's own index; elsewhere PyPI's pure-python wheel.

    PyPI's Linux torchao wheel carries extensions built for ONE CUDA (13 for 0.18), so on a
    cu126 stack they would not load, and ``import torchao`` would still succeed. The torch
    indexes publish a build per CUDA for Linux, and none for Windows.
    """
    return platform.startswith("linux")


def _torchao_index(stack: StackSet, platform: str) -> list[str]:
    if _torchao_from_index(platform):
        return ["--index-url", INDEX_URL.format(name=stack.name)]
    return []


_STACK_LINE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")


def filter_requirements(lines: list[str], also: tuple[str, ...] = ()) -> list[str]:
    """``lines`` minus every requirement naming a stack distribution (or one of ``also``)."""
    drop = {_canonical(n) for n in (*STACK_DISTRIBUTIONS, *also)}
    kept = []
    for line in lines:
        match = _STACK_LINE.match(line)
        if match and not line.lstrip().startswith("#") and _canonical(match.group(1)) in drop:
            continue
        kept.append(line)
    return kept


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_versions(python: str) -> dict[str, str | None]:
    code = (
        "import importlib.metadata as m, json\n"
        f"names = {list(STACK_DISTRIBUTIONS)!r}\n"
        "out = {}\n"
        "for n in names:\n"
        "    try: out[n] = m.version(n)\n"
        "    except m.PackageNotFoundError: out[n] = None\n"
        "print(json.dumps(out))\n"
    )
    done = subprocess.run([python, "-c", code], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def differences(stack: StackSet, installed: dict[str, str | None], platform: str = sys.platform) -> list[str]:
    """Each stack distribution whose installed version is not the set's, as a line."""
    out = []
    for name, want in stack.pins(platform).items():
        have = installed.get(name)
        # torch / torchvision / torchaudio carry their index as a local version (2.15.0+cu132,
        # 2.15.0+cpu); a CPU wheel from PyPI has none. So does torchao on Linux, where it
        # comes from the set's index (a bare version there is PyPI's build, for another
        # CUDA). triton never does.
        from_index = ("torch", "torchvision", "torchaudio") + (("torchao",) if _torchao_from_index(platform) else ())
        if name in from_index:
            expected = {f"{want}+{stack.name}"} | ({want} if stack.driver_min is None else set())
        else:
            expected = {want}
        if have not in expected:
            out.append(f"{name}: installed {have or 'nothing'}, the {stack.name} set needs {want}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("select")
    for name in ("install", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--python", required=True)
        if name == "install":
            p.add_argument("--dry-run", action="store_true")
    f = sub.add_parser("filter")
    f.add_argument("requirements")
    f.add_argument("out")
    f.add_argument("--also", nargs="*", default=[])
    args = parser.parse_args(argv)

    if args.command == "filter":
        lines = Path(args.requirements).read_text(encoding="utf-8").splitlines()
        Path(args.out).write_text("\n".join(filter_requirements(lines, tuple(args.also))) + "\n", encoding="utf-8")
        return 0

    _, sets = read_stack()
    try:
        stack, how = choose(sets, detect_gpus(), os.environ.get("MRLN_TORCH_INDEX") or None)
    except Refused as refused:
        print(f"[torch] REFUSED: {refused}", file=sys.stderr)
        return REFUSED
    print(f"[torch] {stack.name}: torch {stack.torch} ({how})", file=sys.stderr)

    if args.command == "select":
        print(stack.name)
        return 0
    if args.command == "install":
        for command in install_commands(stack, args.python):
            print("[torch] " + " ".join(command[1:]), file=sys.stderr)
            if not args.dry_run:
                subprocess.run(command, check=True)
        return 0
    installed = installed_versions(args.python)
    for name in STACK_DISTRIBUTIONS:
        if name in stack.pins():
            print(f"[torch] {name} {installed.get(name) or 'not installed'}", file=sys.stderr)
    wrong = differences(stack, installed)
    if wrong:
        print("[torch] WRONG STACK:\n  " + "\n  ".join(wrong), file=sys.stderr)
        return 1
    print(f"[torch] the {stack.name} set is installed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
