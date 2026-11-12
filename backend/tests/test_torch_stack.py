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
        assert any("torchaudio==2.11.1+cu132" in c and "--no-deps" in c for c in commands)

    @pytest.mark.parametrize("name", ["cu132", "cu126", "cpu"])
    def test_on_linux_torchao_comes_from_the_sets_own_index(self, name):
        # PyPI's Linux torchao wheel carries CUDA-13 extensions: on a cu126 stack they would not load.
        commands = torch_stack.install_commands(SETS[name], "py", platform="linux")
        (torchao,) = [c for c in commands if any(a.startswith("torchao==") for a in c)]
        assert torchao[torchao.index("--index-url") + 1] == f"https://download.pytorch.org/whl/{name}"
        assert "--no-deps" in torchao

    @pytest.mark.parametrize("name, platform, torchao", [
        ("cu132", "win32", "torchao==0.18.0"),
        ("cu126", "linux", "torchao==0.18.0+cu126"),
        ("cpu", "linux", "torchao==0.18.0+cpu"),
        ("cpu", "win32", "torchao==0.18.0"),
    ])
    def test_every_build_from_a_sets_index_is_pinned_with_its_local_version(self, name, platform, torchao):
        """A bare ``torch==2.15.0`` is satisfied by 2.15.0 from ANY index (+cpu, +cu134), so pip would
        keep another build in a reused venv; the local version makes the set's own build the only fit."""
        stack = SETS[name]
        requirements = [a for c in torch_stack.install_commands(stack, "py", platform) for a in c if "==" in a]
        for dist, version in (("torch", stack.torch), ("torchvision", stack.torchvision),
                              ("torchaudio", stack.torchaudio)):
            assert f"{dist}=={version}+{name}" in requirements
        assert torchao in requirements

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
        assert "this machine: cc 8.6 (driver 575.64)" in done.stderr, "the install must name what it detected"
        assert "cu126: torch 2.14.1" in done.stderr
        assert "torch==2.14.1" in done.stderr

    def test_the_cpu_override_never_probes_nvidia(self, monkeypatch, capsys):
        """The refusal tells a broken driver to set MRLN_TORCH_INDEX=cpu: that must then work."""
        def broken_driver():
            raise torch_stack.Refused("nvidia-smi failed (exit 9): driver/library version mismatch")

        monkeypatch.setattr(torch_stack, "detect_gpus", broken_driver)
        monkeypatch.setenv("MRLN_TORCH_INDEX", "cpu")
        assert torch_stack.main(["select"]) == 0
        assert capsys.readouterr().out.strip() == "cpu"

    def test_without_an_override_a_broken_driver_is_still_refused(self, monkeypatch):
        def broken_driver():
            raise torch_stack.Refused("nvidia-smi failed (exit 9)")

        monkeypatch.setattr(torch_stack, "detect_gpus", broken_driver)
        monkeypatch.delenv("MRLN_TORCH_INDEX", raising=False)
        assert torch_stack.main(["select"]) == torch_stack.REFUSED

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


def _never_asks_the_machine(monkeypatch):
    def asked():
        raise AssertionError("a named set must never ask this machine's nvidia-smi")

    monkeypatch.setattr(torch_stack, "detect_gpus", asked)
    monkeypatch.delenv("MRLN_TORCH_INDEX", raising=False)


@pytest.fixture(scope="module")
def bare_python(tmp_path_factory) -> str:
    """An interpreter that sees no installed distribution but the stub stack put on its PYTHONPATH."""
    venv = tmp_path_factory.mktemp("bare") / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    return str(venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))


def _stub_stack(root: Path, versions: dict[str, str]) -> dict:
    """A directory of bare dist-info metadata, put on PYTHONPATH: what the stub interpreter reports."""
    for name, version in versions.items():
        info = root / f"{name.replace('-', '_')}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")
    return {**_env(), "PYTHONPATH": str(root)}


CU126_ON_LINUX = {"torch": "2.14.1+cu126", "torchvision": "0.29.1+cu126", "torchaudio": "2.11.0+cu126",
                  "torchao": "0.18.0+cu126", "triton": "3.8.0"}


class TestInstallANamedSet:
    """The image build installs the set it names, whatever GPU the build machine has (or none)."""

    def test_it_installs_the_named_set_for_linux_without_asking_the_machine(self, monkeypatch, capsys):
        _never_asks_the_machine(monkeypatch)
        code = torch_stack.main(["install", "--set", "cu126", "--platform", "linux", "--python", "py", "--dry-run"])
        assert code == 0
        err = capsys.readouterr().err
        for pin in ("torch==2.14.1+cu126", "torchvision==0.29.1+cu126", "torchaudio==2.11.0+cu126",
                    "torchao==0.18.0+cu126", "triton==3.8.0"):
            assert pin in err
        assert "triton-windows" not in err
        assert "--index-url https://download.pytorch.org/whl/cu126" in err

    def test_an_unknown_set_is_refused_with_the_names(self, monkeypatch, capsys):
        _never_asks_the_machine(monkeypatch)
        assert torch_stack.main(["install", "--set", "cu999", "--python", "py", "--dry-run"]) == torch_stack.REFUSED
        err = capsys.readouterr().err
        assert "cu132" in err and "cu126" in err
        assert "pip install" not in err

    def test_the_platform_is_only_for_a_named_set(self, monkeypatch):
        _never_asks_the_machine(monkeypatch)
        with pytest.raises(SystemExit):
            torch_stack.main(["install", "--platform", "linux", "--python", "py", "--dry-run"])


class TestCheckANamedSet:
    """The container's start: does the set baked into the image run on this host?"""

    def _check(self, tmp_path, name, smi_text):
        canned = tmp_path / "smi.txt"
        canned.write_text(smi_text, encoding="utf-8")
        env = {**_env(), "MRLN_TORCH_NVIDIA_SMI": str(canned)}
        return subprocess.run([sys.executable, str(BACKEND / "torch_stack.py"), "check", "--set", name],
                              capture_output=True, text=True, env=env)

    def test_a_set_that_fits_passes(self, tmp_path):
        done = self._check(tmp_path, "cu132", "616.92, 12.0\n")
        assert done.returncode == 0, done.stderr
        assert "cu132" in done.stderr and "cc 12.0" in done.stderr

    def test_a_driver_below_the_floor_is_refused_naming_the_images_that_fit(self, tmp_path):
        done = self._check(tmp_path, "cu132", "575.64, 8.6\n")
        assert done.returncode == torch_stack.CHECK_REFUSED == 3
        assert "below R580" in done.stderr
        assert ":latest-cu126" in done.stderr
        assert "set     torch    driver" in done.stderr, "the refusal shows the table"

    def test_cu126_on_blackwell_is_refused_naming_latest(self, tmp_path):
        done = self._check(tmp_path, "cu126", "616.92, 12.0\n")
        assert done.returncode == 3
        assert "no kernels for compute capability 12.0" in done.stderr
        assert "mastermerlin/mrln-arcane-tuner:latest " in done.stderr
        assert ":latest-cu128" not in done.stderr, "a current image fits; the frozen one is not offered"

    def test_blackwell_below_r580_is_sent_to_the_frozen_cu128_image(self, tmp_path):
        done = self._check(tmp_path, "cu132", "575.64, 12.0\n")
        assert done.returncode == 3
        assert ":latest-cu128" in done.stderr and "frozen" in done.stderr
        assert ":latest-cu126" not in done.stderr

    def test_nothing_fits_says_so(self, tmp_path):
        done = self._check(tmp_path, "cu132", "559.99, 8.6\n")
        assert done.returncode == 3
        assert "no image fits this machine" in done.stderr

    def test_no_nvidia_smi_is_its_own_code(self, monkeypatch, capsys):
        # In process: Windows finds System32's nvidia-smi whatever PATH a subprocess gets.
        def no_such_program(*args, **kwargs):
            raise FileNotFoundError("nvidia-smi")

        monkeypatch.delenv("MRLN_TORCH_NVIDIA_SMI", raising=False)
        monkeypatch.setattr(torch_stack.subprocess, "run", no_such_program)
        assert torch_stack.main(["check", "--set", "cu132"]) == torch_stack.CHECK_NO_GPU_INFO == 4
        assert "no nvidia-smi" in capsys.readouterr().err

    def test_an_nvidia_smi_listing_no_gpu_is_no_gpu_info(self, tmp_path):
        assert self._check(tmp_path, "cu132", "").returncode == 4

    def test_a_failing_nvidia_smi_is_no_gpu_info_with_its_error(self, monkeypatch, capsys):
        def broken_driver():
            raise torch_stack.Refused("nvidia-smi failed (exit 9): driver/library version mismatch")

        monkeypatch.setattr(torch_stack, "detect_gpus", broken_driver)
        assert torch_stack.main(["check", "--set", "cu132"]) == 4
        assert "driver/library version mismatch" in capsys.readouterr().err

    def test_the_override_does_not_change_what_is_checked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRLN_TORCH_INDEX", "cpu")
        canned = tmp_path / "smi.txt"
        canned.write_text("616.92, 12.0\n", encoding="utf-8")
        monkeypatch.setenv("MRLN_TORCH_NVIDIA_SMI", str(canned))
        assert torch_stack.main(["check", "--set", "cu126"]) == 3

    def test_an_unknown_set_is_refused(self, monkeypatch):
        _never_asks_the_machine(monkeypatch)
        assert torch_stack.main(["check", "--set", "cu999"]) == torch_stack.REFUSED


AMPERE_UUID = "GPU-1a2b3c4d-0000-1111-2222-333344445555"
BLACKWELL_UUID = "GPU-9f8e7d6c-aaaa-bbbb-cccc-ddddeeeeffff"
# nvidia-smi --query-gpu=index,uuid,driver_version,compute_cap: an Ampere and a Blackwell card.
MIXED = f"0, {AMPERE_UUID}, 616.92, 8.6\n1, {BLACKWELL_UUID}, 616.92, 12.0\n"


def _verdict(monkeypatch, capsys, smi_text: str, name: str, env: dict[str, str]) -> tuple[int, str]:
    """``check --set name`` in process, on canned GPUs, with exactly ``env``'s CUDA variables."""
    monkeypatch.setattr(torch_stack, "detect_gpus", lambda: torch_stack.parse_gpus(smi_text))
    for var in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    code = torch_stack.main(["check", "--set", name])
    return code, capsys.readouterr().err


class TestCheckSeesOnlyTheVisibleGpus:
    """The README supports CUDA_VISIBLE_DEVICES; a GPU it hides is one torch never touches, so
    it must not refuse (or pass) the image's set."""

    def test_a_hidden_blackwell_does_not_refuse_cu126_on_the_selected_ampere(self, monkeypatch, capsys):
        code, err = _verdict(monkeypatch, capsys, MIXED, "cu126", {"CUDA_VISIBLE_DEVICES": AMPERE_UUID})
        assert code == 0, err

    def test_the_selected_blackwell_still_refuses_cu126(self, monkeypatch, capsys):
        code, err = _verdict(monkeypatch, capsys, MIXED, "cu126", {"CUDA_VISIBLE_DEVICES": BLACKWELL_UUID})
        assert code == torch_stack.CHECK_REFUSED, err
        assert "12.0" in err

    def test_unset_checks_every_gpu(self, monkeypatch, capsys):
        assert _verdict(monkeypatch, capsys, MIXED, "cu126", {})[0] == torch_stack.CHECK_REFUSED

    def test_a_unique_uuid_prefix_selects_its_gpu(self, monkeypatch, capsys):
        code, err = _verdict(monkeypatch, capsys, MIXED, "cu126", {"CUDA_VISIBLE_DEVICES": "GPU-1a2b"})
        assert code == 0, err

    def test_an_ordinal_in_pci_order_is_nvidia_smis_index(self, monkeypatch, capsys):
        env = {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        assert _verdict(monkeypatch, capsys, MIXED, "cu126", env)[0] == 0
        env["CUDA_VISIBLE_DEVICES"] = "1"
        assert _verdict(monkeypatch, capsys, MIXED, "cu126", env)[0] == torch_stack.CHECK_REFUSED

    def test_an_ordinal_in_fastest_first_order_that_could_be_either_gpu_is_not_guessed(self, monkeypatch, capsys):
        code, err = _verdict(monkeypatch, capsys, MIXED, "cu126", {"CUDA_VISIBLE_DEVICES": "0"})
        assert code == torch_stack.CHECK_NO_GPU_INFO, err
        assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in err

    def test_an_ordinal_on_identical_gpus_needs_no_order(self, monkeypatch, capsys):
        same = f"0, {AMPERE_UUID}, 616.92, 12.0\n1, {BLACKWELL_UUID}, 616.92, 12.0\n"
        assert _verdict(monkeypatch, capsys, same, "cu126", {"CUDA_VISIBLE_DEVICES": "1"})[0] == torch_stack.CHECK_REFUSED
        assert _verdict(monkeypatch, capsys, same, "cu132", {"CUDA_VISIBLE_DEVICES": "1"})[0] == 0

    @pytest.mark.parametrize("value", ["", "-1", " ", "junk", "7", "GPU-0000", "GPU-"])
    def test_a_list_that_selects_no_gpu_is_no_gpu_to_check(self, monkeypatch, capsys, value):
        code, err = _verdict(monkeypatch, capsys, MIXED, "cu126", {"CUDA_VISIBLE_DEVICES": value})
        assert code == torch_stack.CHECK_NO_GPU_INFO, err
        assert "CUDA_VISIBLE_DEVICES" in err

    def test_parsing_stops_at_the_first_invalid_entry(self, monkeypatch, capsys):
        env = {"CUDA_VISIBLE_DEVICES": f"{AMPERE_UUID},-1,{BLACKWELL_UUID}"}
        assert _verdict(monkeypatch, capsys, MIXED, "cu126", env)[0] == 0
        env = {"CUDA_VISIBLE_DEVICES": f"{AMPERE_UUID},{AMPERE_UUID},{BLACKWELL_UUID}"}
        assert _verdict(monkeypatch, capsys, MIXED, "cu126", env)[0] == 0, "a repeated device ends the list"

    def test_a_mig_instance_named_with_its_parent_gpu_checks_that_gpu(self, monkeypatch, capsys):
        env = {"CUDA_VISIBLE_DEVICES": f"MIG-{AMPERE_UUID}/1/0"}
        assert _verdict(monkeypatch, capsys, MIXED, "cu126", env)[0] == 0
        env = {"CUDA_VISIBLE_DEVICES": f"MIG-{BLACKWELL_UUID}/1/0"}
        assert _verdict(monkeypatch, capsys, MIXED, "cu126", env)[0] == torch_stack.CHECK_REFUSED

    def test_a_mig_uuid_whose_parent_is_unknown_is_not_guessed(self, monkeypatch, capsys):
        env = {"CUDA_VISIBLE_DEVICES": "MIG-5c1e9d2a-1111-2222-3333-444455556666"}
        assert _verdict(monkeypatch, capsys, MIXED, "cu126", env)[0] == torch_stack.CHECK_NO_GPU_INFO

    def test_the_old_two_column_listing_still_parses(self):
        gpus = torch_stack.parse_gpus("616.92, 12.0\n575.64, 8.6\n")
        assert [(g.index, g.driver, g.cc) for g in gpus] == [(0, "616.92", (12, 0)), (1, "575.64", (8, 6))]

    def test_nvidia_smi_is_asked_for_index_and_uuid(self, monkeypatch):
        asked = {}

        def run(cmd, **kwargs):
            asked["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout=MIXED, stderr="")

        monkeypatch.delenv("MRLN_TORCH_NVIDIA_SMI", raising=False)
        monkeypatch.setattr(torch_stack.subprocess, "run", run)
        gpus = torch_stack.detect_gpus()
        assert "--query-gpu=index,uuid,driver_version,compute_cap" in asked["cmd"]
        assert [(g.index, g.uuid, g.cc) for g in gpus] == [(0, AMPERE_UUID, (8, 6)), (1, BLACKWELL_UUID, (12, 0))]

    def test_hidden_devices_do_not_change_startup_verdict(self, monkeypatch, capsys):
        """Property, over seeded random machines: the verdict for a CUDA_VISIBLE_DEVICES that
        selects V equals the verdict on a machine holding only V, whatever the hidden GPUs are
        and whichever way V is spelled (UUID, unique UUID prefix, PCI-order ordinal, MIG parent,
        an invalid tail)."""
        import random

        rng = random.Random(20261111)
        archs = [(7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)]
        drivers = ["560.35", "575.64", "580.82", "616.92"]

        def listing(gpus):
            return "".join(f"{i}, {u}, {d}, {c[0]}.{c[1]}\n" for i, (u, d, c) in enumerate(gpus))

        for trial in range(150):
            n = rng.randint(1, 5)
            driver = rng.choice(drivers)
            uuids = [f"GPU-{rng.getrandbits(128):032x}" for _ in range(n)]
            machine = [(u, driver, rng.choice(archs)) for u in uuids]
            visible = rng.sample(range(n), rng.randint(1, n))
            pci = rng.random() < 0.5
            tokens = []
            for i in visible:
                form = rng.choice(["uuid", "prefix", "mig"] + (["ordinal"] if pci else []))
                u = uuids[i]
                if form == "uuid":
                    tokens.append(u)
                elif form == "prefix":
                    k = next(k for k in range(5, len(u) + 1)
                             if sum(o.startswith(u[:k]) for o in uuids) == 1)
                    tokens.append(u[:k])
                elif form == "mig":
                    tokens.append(f"MIG-{u}/{rng.randint(0, 6)}/0")
                else:
                    tokens.append(str(i))
            if rng.random() < 0.3:
                hidden = [uuids[i] for i in range(n) if i not in visible]
                tokens += [rng.choice(["-1", "junk", str(n + 3)]), *hidden]
            env = {"CUDA_VISIBLE_DEVICES": ",".join(tokens)}
            if pci:
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            # The hidden GPUs re-rolled: none of them may matter.
            rerolled = [m if i in visible else (m[0], m[1], rng.choice(archs)) for i, m in enumerate(machine)]
            only_v = listing([machine[i] for i in visible])
            for name in ("cu132", "cu126"):
                want, _ = _verdict(monkeypatch, capsys, only_v, name, {})
                for gpus in (machine, rerolled):
                    got, err = _verdict(monkeypatch, capsys, listing(gpus), name, env)
                    assert got == want, (trial, name, env, gpus, err)


class TestConstraints:
    def test_the_installed_stack_is_written_with_its_local_versions(self, tmp_path, bare_python):
        env = _stub_stack(tmp_path / "site", {"torch": "2.15.0+cu132", "torchvision": "0.29.2+cu132",
                                              "torchaudio": "2.11.1+cu132", "torchao": "0.18.0+cu132",
                                              "triton": "3.9.0"})
        out = tmp_path / "constraints.txt"
        done = subprocess.run([sys.executable, str(BACKEND / "torch_stack.py"), "constraints",
                               "--python", bare_python, str(out)], capture_output=True, text=True, env=env)
        assert done.returncode == 0, done.stderr
        pins = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
        assert pins == ["torch==2.15.0+cu132", "torchvision==0.29.2+cu132", "torchaudio==2.11.1+cu132",
                        "torchao==0.18.0+cu132", "triton==3.9.0"]

    def test_it_writes_what_is_installed_not_what_a_set_says(self, monkeypatch, tmp_path):
        _never_asks_the_machine(monkeypatch)
        monkeypatch.setattr(torch_stack, "installed_versions", lambda python: {
            "torch": "2.14.1+cu126", "torchvision": "0.29.1+cu126", "torchaudio": "2.11.0+cu126",
            "torchao": "0.18.0+cu126", "triton": "3.8.0", "triton-windows": None})
        out = tmp_path / "c.txt"
        assert torch_stack.main(["constraints", "--python", "py", str(out)]) == 0
        text = out.read_text(encoding="utf-8")
        assert "torch==2.14.1+cu126" in text and "triton==3.8.0" in text
        assert "triton-windows" not in text

    def test_no_torch_installed_writes_nothing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(torch_stack, "installed_versions", lambda python: dict.fromkeys(torch_stack.STACK_DISTRIBUTIONS))
        out = tmp_path / "c.txt"
        code = torch_stack.main(["constraints", "--python", "py", str(out)])
        assert code == torch_stack.CONSTRAINTS_NO_TORCH
        assert not out.exists()

    def test_confirmed_absence_has_a_code_no_crash_can_give(self):
        """An unhandled exception exits 1 and argparse exits 2: neither may read as "no torch"."""
        assert torch_stack.CONSTRAINTS_NO_TORCH not in (0, 1, 2)

    def test_a_failing_probe_is_not_absence(self, monkeypatch, tmp_path):
        def broken(python):
            raise subprocess.CalledProcessError(1, [python, "-c", "..."])

        monkeypatch.setattr(torch_stack, "installed_versions", broken)
        with pytest.raises(subprocess.CalledProcessError):
            torch_stack.main(["constraints", "--python", "py", str(tmp_path / "c.txt")])


class TestVerifyANamedSet:
    def test_a_cu126_stack_passes_with_no_nvidia_smi(self, tmp_path, bare_python):
        """The image's stack is verified by name: on this PC (cc 12.0) today's verify would refuse cu126."""
        env = _stub_stack(tmp_path / "site", CU126_ON_LINUX)
        # Asking the machine would read this missing file and crash: verify --set must never ask.
        env["MRLN_TORCH_NVIDIA_SMI"] = str(tmp_path / "no-nvidia-smi-here.txt")
        done = subprocess.run([sys.executable, str(BACKEND / "torch_stack.py"), "verify", "--set", "cu126",
                               "--platform", "linux", "--python", bare_python],
                              capture_output=True, text=True, env=env)
        assert done.returncode == 0, done.stderr
        assert "the cu126 set is installed" in done.stderr

    def test_another_stack_is_caught(self, monkeypatch, capsys):
        _never_asks_the_machine(monkeypatch)
        monkeypatch.setattr(torch_stack, "installed_versions", lambda python: {**CU126_ON_LINUX, "torch": "2.15.0+cu132"})
        code = torch_stack.main(["verify", "--set", "cu126", "--platform", "linux", "--python", "py"])
        assert code == 1
        assert "torch: installed 2.15.0+cu132, the cu126 set needs 2.14.1" in capsys.readouterr().err


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


REPO = BACKEND.parent
_STACK_PIN = re.compile(r"\b(torch|torchvision|torchaudio|torchao|triton(-windows)?)==")


def _requirement_pins() -> dict[str, str]:
    """The stack distributions requirements.txt pins, by canonical name."""
    stack = {torch_stack._canonical(n) for n in torch_stack.STACK_DISTRIBUTIONS}
    pins = {}
    for line in (BACKEND / "requirements.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s;#]+)", line)
        if match and torch_stack._canonical(match.group(1)) in stack:
            pins[torch_stack._canonical(match.group(1))] = match.group(2)
    return pins


def _without_comments(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def _readme_manual_install() -> str:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    start = text.index("### Manual Install")
    end = text.index("\n### ", start + 1)
    return text[start:end]


class TestTorchStackCfgIsTheOnePlace:
    def test_requirements_pins_are_the_default_set(self):
        """requirements.txt documents the stack a default install gets; it never decides it."""
        default = SETS[DEFAULT]
        want = {torch_stack._canonical(k): v for k, v in {**default.pins("linux"), **default.pins("win32")}.items()}
        assert _requirement_pins() == want

    def test_ci_names_no_torch_stack_version(self):
        code = _without_comments((REPO / ".github" / "workflows" / "gate.yml").read_text(encoding="utf-8"))
        assert not _STACK_PIN.search(code), "gate.yml pins a torch-stack version itself; torch-stack.cfg is the only place"
        assert "download.pytorch.org" not in code, "gate.yml names a torch index itself"
        assert "torch_stack.py install" in code and "torch_stack.py verify" in code

    def test_the_readme_install_block_names_no_torch_stack_version(self):
        block = _readme_manual_install()
        assert not _STACK_PIN.search(block), "the README's manual install pins a torch-stack version itself"
        assert "download.pytorch.org" not in block, "the README's manual install names a torch index itself"
        assert "torch_stack.py install" in block and "torch_stack.py verify" in block


class TestTheTorchaudioClaimIsItsMetadata:
    """requirements.txt says what torchaudio declares; the installed metadata decides.

    The old line said torchaudio 2.11.0 "declares torch==2.11.0". Its metadata
    declares no dependency at all, and the claim stood for months because nothing
    read it.
    """

    def test_the_comment_names_exactly_what_the_installed_torchaudio_declares(self):
        import importlib.metadata as metadata

        try:
            declared = metadata.requires("torchaudio") or []
            installed = metadata.version("torchaudio").split("+")[0]
        except metadata.PackageNotFoundError:
            pytest.skip("no torchaudio in this environment")
        line = next(ln for ln in (BACKEND / "requirements.txt").read_text(encoding="utf-8").splitlines()
                    if ln.startswith("torchaudio=="))
        if installed != line.split("==")[1].split()[0]:
            pytest.skip(f"this environment has torchaudio {installed}, not the pinned one")
        torch_specs = [re.sub(r"[\s()]", "", r.split(";")[0]) for r in declared
                       if re.match(r"torch(?![\w-])", r.strip())]
        claimed = re.findall(r"declares (torch[<>=!~][^\s,;)]+)", line)
        assert claimed == torch_specs, (
            f"requirements.txt says torchaudio declares {claimed or 'nothing'}; its metadata declares {torch_specs}"
        )


def _env() -> dict:
    import os

    env = {k: v for k, v in os.environ.items()
           if k not in ("MRLN_TORCH_INDEX", "MRLN_TORCH_NVIDIA_SMI", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}
    return env
