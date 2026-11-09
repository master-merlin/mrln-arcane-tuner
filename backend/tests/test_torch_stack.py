"""The torch-stack selector (backend/torch_stack.py): which set a machine gets, and what it refuses.

Every case feeds canned ``nvidia-smi --query-gpu=driver_version,compute_cap`` output, the way
``MRLN_TORCH_NVIDIA_SMI`` does for a real run, so the table is tested on machines this one is not.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("torch_stack", BACKEND / "torch_stack.py")
torch_stack = importlib.util.module_from_spec(_spec)
sys.modules["torch_stack"] = torch_stack
_spec.loader.exec_module(torch_stack)

DEFAULT, SETS = torch_stack.read_stack()


def smi(*gpus: tuple[str, str]) -> list:
    """GPUs as nvidia-smi lists them: (driver_version, compute_cap) per line."""
    return torch_stack.parse_gpus("\n".join(f"{driver}, {cc}" for driver, cc in gpus))


def chosen(gpus, override=None) -> str:
    return torch_stack.choose(SETS, gpus, override)[0].name


def refused(gpus, override=None) -> str:
    with pytest.raises(torch_stack.Refused) as caught:
        torch_stack.choose(SETS, gpus, override)
    return str(caught.value)


class TestTheStackFile:
    def test_the_default_set_is_cu132_and_the_fallback_cu126(self):
        assert DEFAULT == "cu132"
        assert list(SETS) == ["cu132", "cu126", "cpu"]

    def test_the_cpu_set_carries_the_default_versions(self):
        assert SETS["cpu"].pins("linux") == SETS[DEFAULT].pins("linux")
        assert SETS["cpu"].pins("win32") == SETS[DEFAULT].pins("win32")

    def test_triton_is_per_os(self):
        assert SETS["cu132"].pins("win32")["triton-windows"] == "3.9.0.post30"
        assert "triton" not in SETS["cu132"].pins("win32")
        assert SETS["cu132"].pins("linux")["triton"] == "3.9.0"
        assert "triton-windows" not in SETS["cu132"].pins("linux")


class TestTheMachineGetsTheHighestSetThatFits:
    @pytest.mark.parametrize("driver, cc", [
        ("616.92", "12.0"),  # this machine
        ("580.65", "8.6"),   # cu132's driver boundary
        ("580.65", "7.5"),   # cu132's lowest architecture
        ("590.10", "8.9"),   # Ada, through sm_86
        ("590.10", "10.3"),  # through sm_100
    ])
    def test_cu132(self, driver, cc):
        assert chosen(smi((driver, cc))) == "cu132"

    @pytest.mark.parametrize("driver, cc", [
        ("579.99", "8.6"),  # one below cu132's floor
        ("580.65", "7.0"),  # Volta: below cu132's architectures
        ("560.35", "6.1"),  # cu126's driver boundary
        ("565.57", "7.5"),
    ])
    def test_cu126(self, driver, cc):
        assert chosen(smi((driver, cc))) == "cu126"

    @pytest.mark.parametrize("driver, cc", [
        ("575.64", "12.0"),  # Blackwell below R580: cu126 has no sm_120
        ("559.99", "8.6"),   # below every floor
        ("600.00", "11.0"),  # no set lists sm_110
        ("600.00", "3.7"),
    ])
    def test_nothing_fits_and_the_table_is_shown(self, driver, cc):
        message = refused(smi((driver, cc)))
        assert "no torch build fits this machine" in message
        assert "cu132" in message and "cu126" in message
        assert f"cc {cc}" in message

    def test_no_nvidia_smi_is_the_cpu_build_said_loudly(self):
        stack, how = torch_stack.choose(SETS, None, None)
        assert stack.name == "cpu"
        assert "without GPU support" in how

    def test_an_nvidia_smi_listing_no_gpu_is_refused(self):
        assert "lists no GPU" in refused([])


class TestEveryGpuMustFit:
    def test_two_gpus_both_in_cu132(self):
        assert chosen(smi(("616.92", "8.6"), ("616.92", "12.0"))) == "cu132"

    def test_two_gpus_both_in_cu126(self):
        assert chosen(smi(("575.64", "7.0"), ("575.64", "8.6"))) == "cu126"

    def test_no_set_holds_both(self):
        gpus = smi(("616.92", "7.0"), ("616.92", "12.0"))
        message = refused(gpus)
        assert "cc 7.0" in message and "cc 12.0" in message
        assert "MRLN_TORCH_INDEX=cu132 cannot run here" in refused(gpus, "cu132")


class TestTheOverrideChoosesButNeverSkipsTheChecks:
    def test_it_picks_a_set_that_fits(self):
        assert chosen(smi(("575.64", "8.6")), "cu126") == "cu126"

    def test_it_is_refused_where_the_set_has_no_kernels(self):
        assert "no kernels for compute capability 12.0" in refused(smi(("616.92", "12.0")), "cu126")

    def test_it_is_refused_below_the_driver_floor(self):
        assert "below R580" in refused(smi(("575.64", "8.6")), "cu132")

    def test_cpu_always_passes(self):
        assert chosen(smi(("616.92", "12.0")), "cpu") == "cpu"
        assert chosen(None, "cpu") == "cpu"

    def test_an_unknown_name_is_refused_with_the_names(self):
        message = refused(smi(("616.92", "12.0")), "cu999")
        assert "cu132" in message and "cu126" in message and "cpu" in message


class TestTheArchitectureRule:
    def test_a_minor_counts_only_within_its_own_major(self):
        # 8.0 is in cu132; a 9.x GPU must not borrow it, nor a 7.x one an 8.x kernel.
        assert not SETS["cu132"].fits_arch((7, 0))
        assert SETS["cu132"].fits_arch((9, 0))
        assert not SETS["cu126"].fits_arch((12, 0))
        assert SETS["cu126"].fits_arch((5, 2))  # Maxwell 5.2 through sm_50


class TestInstall:
    def test_torch_and_torchaudio_come_from_the_same_index(self):
        commands = torch_stack.install_commands(SETS["cu132"], "py", platform="win32")
        index = [c[c.index("--index-url") + 1] for c in commands if "--index-url" in c]
        assert index == ["https://download.pytorch.org/whl/cu132"] * 2
        assert any("torchaudio==2.11.1" in c and "--no-deps" in c for c in commands)

    @pytest.mark.parametrize("name", ["cu132", "cu126", "cpu"])
    def test_on_linux_torchao_comes_from_the_sets_own_index(self, name):
        # PyPI's Linux torchao wheel carries CUDA-13 extensions: on a cu126 stack they would not load.
        commands = torch_stack.install_commands(SETS[name], "py", platform="linux")
        (torchao,) = [c for c in commands if any(a.startswith("torchao==") for a in c)]
        assert torchao[torchao.index("--index-url") + 1] == f"https://download.pytorch.org/whl/{name}"
        assert "--no-deps" in torchao

    def test_on_windows_torchao_comes_from_pypi(self):
        # The CUDA indexes publish torchao for Linux only; PyPI's Windows wheel is pure python.
        commands = torch_stack.install_commands(SETS["cu126"], "py", platform="win32")
        (torchao,) = [c for c in commands if any(a.startswith("torchao==") for a in c)]
        assert "--index-url" not in torchao

    def test_every_stack_distribution_of_the_os_is_installed_at_its_pin(self):
        for platform in ("win32", "linux"):
            stack = SETS["cu126"]
            flat = " ".join(" ".join(c) for c in torch_stack.install_commands(stack, "py", platform))
            for name, version in stack.pins(platform).items():
                assert f"{name}=={version}" in flat

    def test_a_dry_run_names_the_set_and_installs_nothing(self, tmp_path):
        canned = tmp_path / "smi.txt"
        canned.write_text("575.64, 8.6\n", encoding="utf-8")
        done = subprocess.run(
            [sys.executable, str(BACKEND / "torch_stack.py"), "install", "--python", "nonexistent-python", "--dry-run"],
            capture_output=True, text=True, env={**_env(), "MRLN_TORCH_NVIDIA_SMI": str(canned)},
        )
        assert done.returncode == 0, done.stderr
        assert "cu126: torch 2.14.1" in done.stderr
        assert "torch==2.14.1" in done.stderr

    def test_a_refusal_exits_non_zero_before_any_install(self, tmp_path):
        canned = tmp_path / "smi.txt"
        canned.write_text("575.64, 12.0\n", encoding="utf-8")
        done = subprocess.run(
            # --dry-run: should the refusal ever break, the test lists pip commands rather than running them.
            [sys.executable, str(BACKEND / "torch_stack.py"), "install", "--python", "nonexistent-python", "--dry-run"],
            capture_output=True, text=True, env={**_env(), "MRLN_TORCH_NVIDIA_SMI": str(canned)},
        )
        assert done.returncode == torch_stack.REFUSED
        assert "REFUSED" in done.stderr
        assert "pip install" not in done.stderr


class TestFilter:
    def test_the_bulk_step_never_sees_a_stack_distribution(self):
        lines = [
            "torch==2.15.0", "torchao==0.18.0", "torchaudio==2.11.1  # note", "torchvision==0.29.2",
            "triton==3.9.0; sys_platform == 'linux'", "triton-windows==3.9.0.post30; sys_platform == 'win32'",
            "Triton_Windows==1", "torchmetrics==1.0", "torchsde==0.2", "# torch==2.12.1 in a comment", "sam3==0.1.4",
        ]
        kept = torch_stack.filter_requirements(lines, also=("sam3",))
        assert kept == ["torchmetrics==1.0", "torchsde==0.2", "# torch==2.12.1 in a comment"]

    @pytest.mark.parametrize("name", torch_stack.STACK_DISTRIBUTIONS)
    def test_each_stack_distribution_is_dropped(self, name):
        assert torch_stack.filter_requirements([f"{name}==1.0"]) == []


class TestVerify:
    def test_the_exact_set_passes(self):
        stack = SETS["cu132"]
        installed = {"torch": "2.15.0+cu132", "torchvision": "0.29.2+cu132", "torchaudio": "2.11.1+cu132",
                     "torchao": "0.18.0", "triton-windows": "3.9.0.post30"}
        assert torch_stack.differences(stack, installed, "win32") == []

    def test_a_bulk_step_that_moved_triton_is_caught(self):
        stack = SETS["cu126"]
        installed = {"torch": "2.14.1+cu126", "torchvision": "0.29.1+cu126", "torchaudio": "2.11.0+cu126",
                     "torchao": "0.18.0", "triton-windows": "3.9.0.post30"}
        assert torch_stack.differences(stack, installed, "win32") == [
            "triton-windows: installed 3.9.0.post30, the cu126 set needs 3.8.0.post29"]

    def test_a_torch_from_another_index_is_caught(self):
        stack = SETS["cu132"]
        installed = {"torch": "2.15.0+cu134", "torchvision": "0.29.2+cu132", "torchaudio": "2.11.1+cu132",
                     "torchao": "0.18.0+cu132", "triton": "3.9.0"}
        assert torch_stack.differences(stack, installed, "linux") == [
            "torch: installed 2.15.0+cu134, the cu132 set needs 2.15.0"]

    def test_the_cpu_set_accepts_the_cpu_local_version_and_none(self):
        stack = SETS["cpu"]
        for local in ("+cpu", ""):
            installed = {"torch": f"2.15.0{local}", "torchvision": f"0.29.2{local}",
                         "torchaudio": f"2.11.1{local}", "torchao": f"0.18.0{local}", "triton": "3.9.0"}
            assert torch_stack.differences(stack, installed, "linux") == []

    def test_on_linux_a_pypi_torchao_on_a_cuda_set_is_caught(self):
        # A bare 0.18.0 on Linux is PyPI's CUDA-13 build, not the cu126 index's.
        stack = SETS["cu126"]
        installed = {"torch": "2.14.1+cu126", "torchvision": "0.29.1+cu126", "torchaudio": "2.11.0+cu126",
                     "torchao": "0.18.0", "triton": "3.8.0"}
        assert torch_stack.differences(stack, installed, "linux") == [
            "torchao: installed 0.18.0, the cu126 set needs 0.18.0"]

    def test_on_windows_a_torchao_from_a_cuda_index_is_not_expected(self):
        stack = SETS["cu132"]
        installed = {"torch": "2.15.0+cu132", "torchvision": "0.29.2+cu132", "torchaudio": "2.11.1+cu132",
                     "torchao": "0.18.0+cu132", "triton-windows": "3.9.0.post30"}
        assert torch_stack.differences(stack, installed, "win32") == [
            "torchao: installed 0.18.0+cu132, the cu132 set needs 0.18.0"]


INSTALLERS = ("install.ps1", "install.bat", "install.sh")


def _code(name: str) -> str:
    """The installer without its comment lines."""
    lines = (BACKEND / name).read_text(encoding="utf-8").splitlines()
    return "\n".join(ln for ln in lines if not ln.lstrip().upper().startswith(("#", "REM ", "::")))


class TestTheInstallers:
    @pytest.mark.parametrize("name", INSTALLERS)
    def test_the_stack_comes_from_the_selector_and_is_verified_at_the_end(self, name):
        code = _code(name)
        if name == "install.ps1":  # arguments as a PowerShell array
            install, bulk = code.find('"install", "--python"'), code.find('"pip", "install", "-r"')
        else:
            install, bulk = code.find("install --python"), code.find("pip install -r")
        verify = code.rfind("verify")
        assert -1 not in (install, bulk, verify), f"{name}: install / bulk / verify step missing"
        assert install < bulk < verify, f"{name}: the stack must be installed first and verified last"

    @pytest.mark.parametrize("name", INSTALLERS)
    def test_the_bulk_install_filters_through_the_selector(self, name):
        code = _code(name)
        assert "filter" in code and "--also" in code, f"{name}: the bulk requirements step is not filtered by torch_stack.py"
        assert "grep -iv" not in code and "findstr /V" not in code and "-notmatch" not in code, (
            f"{name}: a hand-written requirements filter is back; the selector owns the stack's names"
        )

    @pytest.mark.parametrize("name", INSTALLERS)
    def test_no_installer_names_a_torch_version(self, name):
        code = _code(name)
        assert not re.search(r"\b(torch|torchvision|torchaudio|torchao|triton(-windows)?)==", code), (
            f"{name} pins a torch-stack version itself; torch-stack.cfg is the only place"
        )
        assert "download.pytorch.org" not in code, f"{name} names a torch index itself"

    @pytest.mark.parametrize("name, activation", [
        ("install.ps1", "Activate.ps1"), ("install.bat", "activate.bat"), ("install.sh", "bin/activate"),
    ])
    def test_no_installer_activates_a_venv(self, name, activation):
        """An activated venv is how an install lands in the wrong one: they run the target's interpreter."""
        assert activation not in _code(name), f"{name} activates a venv instead of using its interpreter by path"

    @pytest.mark.parametrize("name, anchor", [
        ("install.ps1", "$PSScriptRoot"), ("install.bat", "%~dp0"), ("install.sh", 'dirname "$0"'),
    ])
    def test_paths_are_anchored_on_the_script(self, name, anchor):
        code = _code(name)
        assert anchor in code, f"{name} does not anchor its paths on its own directory"
        # requirements.txt and torch_stack.py are named once each, joined to that directory.
        for file in ("requirements.txt", "torch_stack.py"):
            uses = [ln for ln in code.splitlines() if file in ln]
            assert len(uses) == 1, f"{name} names {file} {len(uses)} times; expected one anchored definition"
            assert any(a in uses[0] for a in ("$Here", "%HERE%", "$HERE")), f"{name}: {file} is not joined to its directory"

    @pytest.mark.parametrize("name, flag", [
        ("install.ps1", "-Yes"), ("install.bat", "--yes"), ("install.sh", "--yes"),
    ])
    def test_each_takes_a_venv_target_and_a_no_prompt_switch(self, name, flag):
        text = (BACKEND / name).read_text(encoding="utf-8")
        assert flag.lstrip("-").lower() in text.lower()
        assert ("VenvDir" in text) if name == "install.ps1" else ("--venv-dir" in text)


def _env() -> dict:
    import os

    env = {k: v for k, v in os.environ.items() if k not in ("MRLN_TORCH_INDEX", "MRLN_TORCH_NVIDIA_SMI")}
    return env
