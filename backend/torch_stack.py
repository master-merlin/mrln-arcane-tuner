"""Choose, install and verify the torch stack this machine can run.

Every installer calls this file; the versions live only in ``torch-stack.env`` next to it.
Standard library only: it runs before any dependency is installed.

    python torch_stack.py select                  # print the chosen set's name
    python torch_stack.py install --python <py>   # pip-install the chosen set into that interpreter
    python torch_stack.py filter <req> <out> [--also NAME ...]
                                                  # requirements minus the stack (and NAMEs)
    python torch_stack.py verify --python <py>    # the installed stack equals the chosen set

The image build and the container's start name their set instead, and never choose:

    python torch_stack.py install --set <name> [--platform linux] --python <py>
                                                  # that set, whatever GPU this machine has (or none)
    python torch_stack.py check --set <name>      # does that set run on the GPUs CUDA can see
                                                  # (CUDA_VISIBLE_DEVICES)? 0 fits, 3 refused (with
                                                  # the images that fit), 4 no GPU information (no
                                                  # or a failing nvidia-smi, no GPU visible, or a
                                                  # visible set it cannot pin down)
    python torch_stack.py constraints --python <py> <out>
                                                  # the installed stack as pip constraints;
                                                  # 0 written, 5 confirmed no torch, else a fault
    python torch_stack.py verify --set <name> [--platform linux] --python <py>
                                                  # the installed stack equals that set

The choice:
- ``MRLN_TORCH_INDEX`` names a set (or ``cpu``). It chooses the set but never skips its checks.
- Otherwise the highest set in ``SETS`` that fits. A set fits when the driver meets its
  ``DRIVER_MIN`` and EVERY GPU ``nvidia-smi`` lists has a compatible architecture in its
  ``ARCHS``: one venv serves every GPU on the machine.
- No ``nvidia-smi`` at all: the CPU set, said loudly.
- Nothing fits: exit 2 with the table, before anything is installed.

``MRLN_TORCH_NVIDIA_SMI`` names a file holding canned ``nvidia-smi
--query-gpu=index,uuid,driver_version,compute_cap --format=csv,noheader`` output (or the older
two columns, ``driver_version,compute_cap``), used instead of running it (tests, and simulating
another machine).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
STACK_FILE = HERE / "torch-stack.cfg"
INDEX_URL = "https://download.pytorch.org/whl/{name}"
CPU = "cpu"

#: The distributions this file owns. No installer's bulk requirements step may install them.
STACK_DISTRIBUTIONS = ("torch", "torchvision", "torchaudio", "torchao", "triton", "triton-windows")

REFUSED = 2
CHECK_REFUSED = 3
CHECK_NO_GPU_INFO = 4
#: ``constraints``: confirmed that no torch is installed. Never 1, which any unhandled exception
#: exits with, nor 2 (argparse's): a probe that crashed must never read as "nothing to keep".
CONSTRAINTS_NO_TORCH = 5

#: The published container images. The default set's image takes the unsuffixed ``:latest``.
IMAGE = "mastermerlin/mrln-arcane-tuner"


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
    index: int | None = None  # nvidia-smi's index: the PCI bus order
    uuid: str | None = None  # "GPU-...", as nvidia-smi and CUDA_VISIBLE_DEVICES spell it

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
    """``index, uuid, driver_version, compute_cap`` lines, or the older ``driver_version,
    compute_cap`` ones (each then indexed by its position, with no UUID)."""
    gpus = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            index, uuid, driver, cc = int(parts[0]), parts[1], parts[2], parts[3]
        else:
            (driver, cc), index, uuid = parts, len(gpus), None
        major, _, minor = cc.partition(".")
        gpus.append(Gpu(driver=driver, cc=(int(major), int(minor or 0)), index=index, uuid=uuid))
    return gpus


def detect_gpus() -> list[Gpu] | None:
    """The GPUs ``nvidia-smi`` lists; None when there is no ``nvidia-smi``."""
    canned = os.environ.get("MRLN_TORCH_NVIDIA_SMI")
    if canned:
        return parse_gpus(Path(canned).read_text(encoding="utf-8"))
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,driver_version,compute_cap", "--format=csv,noheader"],
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


#: The cu128 image (torch 2.11.0), frozen: never rebuilt, its tags never moved. It is the one image
#: for Blackwell on R570-R579, where cu132 is below its driver floor and cu126 has no sm_100/sm_120.
#: Its architectures are what its README row claims: Ampere and newer, Blackwell included.
FROZEN_CU128 = StackSet("cu128", torch="2.11.0", torchvision="", torchaudio="", torchao="", triton="",
                        triton_windows="", driver_min=570, archs=((8, 0), (8, 6), (9, 0), (10, 0), (12, 0)))


def image_name(name: str, default: str) -> str:
    """The rolling tag of the image that carries set ``name``."""
    return f"{IMAGE}:latest" if name == default else f"{IMAGE}:latest-{name}"


def images_that_fit(sets: dict[str, StackSet], default: str, gpus: list[Gpu]) -> list[str]:
    """The images whose set runs on ``gpus``, each with what it carries; the frozen cu128 one only
    when no current image fits."""
    fit = [f"{image_name(s.name, default)} ({s.name}: torch {s.torch}, R{s.driver_min}+)"
           for s in sets.values() if s.driver_min is not None and why_not(s, gpus) is None]
    if not fit and why_not(FROZEN_CU128, gpus) is None:
        fit.append(f"{IMAGE}:latest-cu128 (cu128: torch {FROZEN_CU128.torch}, R{FROZEN_CU128.driver_min}+;"
                   " frozen, it gets no further updates)")
    return fit


def visible_candidates(gpus: list[Gpu], env: Mapping[str, str]) -> list[list[Gpu]]:
    """Each set of ``gpus`` that CUDA may make visible under ``env``; one set unless it cannot tell.

    ``CUDA_VISIBLE_DEVICES`` unset: every GPU. Otherwise its comma-separated entries, read up
    to the first that names no device or repeats one (CUDA ignores the rest): an ordinal, a
    ``GPU-<uuid>`` (or a prefix matching one GPU alone), or a MIG instance, whose GPU is known
    from ``MIG-GPU-<uuid>/<gi>/<ci>`` and not from ``MIG-<uuid>``. Empty, or ``-1``, selects none.
    An ordinal is nvidia-smi's index (the PCI bus order) only under
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID``; under the default FASTEST_FIRST, and for a MIG UUID,
    the device could be any GPU not already named, so every such choice is a candidate.
    """
    raw = env.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return [list(gpus)]
    pci_order = env.get("CUDA_DEVICE_ORDER", "").strip().upper() == "PCI_BUS_ID"
    by_index = sorted(gpus, key=lambda g: g.index if g.index is not None else 0)

    def by_uuid(prefix: str) -> Gpu | None:
        matches = [g for g in gpus if g.uuid and g.uuid.lower().startswith(prefix.lower())]
        return matches[0] if len(matches) == 1 and len(prefix) > len("GPU-") else None

    known: list[Gpu] = []
    ordinals: set[int] = set()
    unknown_ordinals = unknown_migs = 0
    for token in (t.strip() for t in raw.split(",")):
        if token.isdigit():
            i = int(token)
            if i >= len(gpus) or i in ordinals:
                break
            ordinals.add(i)
            if pci_order:
                if by_index[i] in known:
                    break
                known.append(by_index[i])
            else:
                unknown_ordinals += 1
        elif token.upper().startswith("GPU-"):
            g = by_uuid(token)
            if g is None or g in known:
                break
            known.append(g)
        elif token.upper().startswith("MIG-GPU-"):
            g = by_uuid(token[len("MIG-"):].split("/")[0])
            if g is None:
                break
            if g not in known:  # several instances of one GPU are one GPU to check
                known.append(g)
        elif token.upper().startswith("MIG-") and len(token) > len("MIG-"):
            unknown_migs += 1
        else:
            break
    if not unknown_ordinals and not unknown_migs:
        return [known]
    rest = [g for g in gpus if g not in known]
    # An unplaced ordinal is a GPU of its own unless it is one already named; a MIG UUID may be
    # an instance of a named GPU, adding none.
    low = max(0, unknown_ordinals - len(known))
    high = min(unknown_ordinals + unknown_migs, len(rest))
    out = []
    for size in range(low, high + 1):
        for extra in itertools.combinations(rest, size):
            if known or extra:
                out.append(known + list(extra))
    return out or [known]


def _described(gpus: list[Gpu]) -> str:
    return ", ".join(f"cc {g.cc[0]}.{g.cc[1]}" + (f" (index {g.index})" if g.index is not None else "")
                     for g in gpus)


def check(sets: dict[str, StackSet], default: str, name: str) -> int:
    """Whether set ``name`` runs on this machine's GPUs, said on stderr; the exit code.

    Only the GPUs CUDA can see are checked (``visible_candidates``): one that
    ``CUDA_VISIBLE_DEVICES`` hides is never touched by torch, so it neither refuses nor passes
    the set.
    """
    try:
        gpus = detect_gpus()
    except Refused as failed:
        print(f"[torch] cannot check the {name} set: {failed}", file=sys.stderr)
        return CHECK_NO_GPU_INFO
    if not gpus:
        said = "no nvidia-smi" if gpus is None else "nvidia-smi lists no GPU"
        print(f"[torch] cannot check the {name} set: {said}, so there is no GPU to check", file=sys.stderr)
        return CHECK_NO_GPU_INFO
    print(f"[torch] {machine(gpus)}", file=sys.stderr)
    candidates = visible_candidates(gpus, os.environ)
    shown = os.environ.get("CUDA_VISIBLE_DEVICES")
    if all(not c for c in candidates):
        print(f"[torch] cannot check the {name} set: CUDA_VISIBLE_DEVICES={shown!r} leaves no GPU visible,"
              " so there is no GPU to check", file=sys.stderr)
        return CHECK_NO_GPU_INFO
    reasons = [why_not(sets[name], c) for c in candidates]
    if any(r is None for r in reasons) and any(r is not None for r in reasons):
        print(f"[torch] cannot check the {name} set: CUDA_VISIBLE_DEVICES={shown!r} could select any of"
              f" {' / '.join(_described(c) for c in candidates)}, and the set fits some of them only."
              " Name GPUs by UUID, or set CUDA_DEVICE_ORDER=PCI_BUS_ID so an ordinal is nvidia-smi's index.",
              file=sys.stderr)
        return CHECK_NO_GPU_INFO
    if shown is not None:
        print(f"[torch] CUDA_VISIBLE_DEVICES={shown!r}: checking {' / '.join(_described(c) for c in candidates)}",
              file=sys.stderr)
    # Every candidate agrees. Their union is what the refusal speaks for: an image that fits it
    # fits each of them.
    gpus = [g for g in gpus if any(g in c for c in candidates)]
    reason = reasons[0]
    if reason is None:
        print(f"[torch] the {name} set (torch {sets[name].torch}) fits this machine", file=sys.stderr)
        return 0
    fit = images_that_fit(sets, default, gpus)
    instead = ("use instead:\n  " + "\n  ".join(fit)) if fit else "no image fits this machine"
    print(f"[torch] REFUSED: the {name} set (torch {sets[name].torch}) cannot run here: {reason}\n"
          f"{instead}\n{table(sets, gpus)}", file=sys.stderr)
    return CHECK_REFUSED


def table(sets: dict[str, StackSet], gpus: list[Gpu]) -> str:
    rows = ["set     torch    driver   architectures (compute capability)"]
    for s in sets.values():
        if s.driver_min is None:
            continue
        archs = " ".join(f"{a}.{b}" for a, b in s.archs)
        rows.append(f"{s.name:<7} {s.torch:<8} R{s.driver_min}+   {archs}")
    rows.append(machine(gpus))
    return "\n".join(rows)


def machine(gpus: list[Gpu]) -> str:
    """What was detected, the way the installer prints it and the refusal table ends."""
    return "this machine: " + ", ".join(f"cc {g.cc[0]}.{g.cc[1]} (driver {g.driver})" for g in gpus)


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
    # Each build from the set's index is pinned WITH its local version (torch==2.15.0+cu132): a bare
    # ==2.15.0 is satisfied by 2.15.0 from any index (+cpu, +cu134), so pip would keep another build
    # already in a reused venv, and only verify would notice.
    local = f"+{stack.name}"
    torchao = f"torchao=={stack.torchao}" + (local if _torchao_from_index(platform) else "")
    commands = [
        pip + [f"torch=={stack.torch}{local}", f"torchvision=={stack.torchvision}{local}", "--index-url", index],
        # --no-deps: torch is already in place; it keeps a torchaudio pin from ever moving torch.
        pip + ["--no-deps", f"torchaudio=={stack.torchaudio}{local}", "--index-url", index],
        pip + ["--no-deps", torchao] + _torchao_index(stack, platform),
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


def constraint_lines(installed: dict[str, str | None]) -> list[str]:
    """The installed stack as pip constraints, each pinned with its local version (torch==2.15.0+cu132).

    A local version in a constraint admits only that build, while ``torch>=2`` or ``torch==2.15.0``
    in another package's metadata still matches it; a package that needs any other torch fails
    pip's resolution instead of replacing the stack.
    """
    return [f"{name}=={installed[name]}" for name in STACK_DISTRIBUTIONS if installed.get(name)]


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
        p.add_argument("--set", dest="set_name", help="this set, never chosen by this machine")
        p.add_argument("--platform", help="with --set: the OS to install or verify for (default: this one)")
        if name == "install":
            p.add_argument("--dry-run", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--set", dest="set_name", required=True)
    k = sub.add_parser("constraints")
    k.add_argument("--python", required=True)
    k.add_argument("out")
    f = sub.add_parser("filter")
    f.add_argument("requirements")
    f.add_argument("out")
    f.add_argument("--also", nargs="*", default=[])
    args = parser.parse_args(argv)
    if getattr(args, "platform", None) and not args.set_name:
        parser.error("--platform needs --set: a chosen set is always for this machine's OS")

    if args.command == "filter":
        lines = Path(args.requirements).read_text(encoding="utf-8").splitlines()
        Path(args.out).write_text("\n".join(filter_requirements(lines, tuple(args.also))) + "\n", encoding="utf-8")
        return 0
    if args.command == "constraints":
        pins = constraint_lines(installed_versions(args.python))
        if not any(p.startswith("torch==") for p in pins):
            print(f"[torch] no torch is installed in {args.python}: no constraints written", file=sys.stderr)
            return CONSTRAINTS_NO_TORCH
        Path(args.out).write_text("# The installed torch stack: torch_stack.py constraints\n"
                                  + "\n".join(pins) + "\n", encoding="utf-8")
        print(f"[torch] constraints: {' '.join(pins)}", file=sys.stderr)
        return 0

    default, sets = read_stack()
    if getattr(args, "set_name", None):
        if args.set_name not in sets:
            print(f"[torch] REFUSED: {args.set_name} is not a set; the sets are: {', '.join(sets)}", file=sys.stderr)
            return REFUSED
        if args.command == "check":
            return check(sets, default, args.set_name)
        stack, platform = sets[args.set_name], args.platform or sys.platform
        print(f"[torch] {stack.name}: torch {stack.torch} (--set {stack.name}, for {platform})", file=sys.stderr)
        return _install_or_verify(args, stack, platform)

    override = os.environ.get("MRLN_TORCH_INDEX") or None
    try:
        # The CPU set needs no GPU, so an override naming it never asks nvidia-smi: a broken
        # driver is exactly when the refusal tells the user to set MRLN_TORCH_INDEX=cpu.
        cpu_override = override in sets and sets[override].driver_min is None
        gpus = None if cpu_override else detect_gpus()
        if gpus:
            print(f"[torch] {machine(gpus)}", file=sys.stderr)
        stack, how = choose(sets, gpus, override)
    except Refused as refused:
        print(f"[torch] REFUSED: {refused}", file=sys.stderr)
        return REFUSED
    print(f"[torch] {stack.name}: torch {stack.torch} ({how})", file=sys.stderr)

    if args.command == "select":
        print(stack.name)
        return 0
    return _install_or_verify(args, stack, sys.platform)


def _install_or_verify(args: argparse.Namespace, stack: StackSet, platform: str) -> int:
    if args.command == "install":
        for command in install_commands(stack, args.python, platform):
            print("[torch] " + " ".join(command[1:]), file=sys.stderr)
            if not args.dry_run:
                subprocess.run(command, check=True)
        return 0
    installed = installed_versions(args.python)
    for name in STACK_DISTRIBUTIONS:
        if name in stack.pins(platform):
            print(f"[torch] {name} {installed.get(name) or 'not installed'}", file=sys.stderr)
    wrong = differences(stack, installed, platform)
    if wrong:
        print("[torch] WRONG STACK:\n  " + "\n  ".join(wrong), file=sys.stderr)
        return 1
    print(f"[torch] the {stack.name} set is installed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
