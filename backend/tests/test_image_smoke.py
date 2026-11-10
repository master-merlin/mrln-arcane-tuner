"""The image's GPU kernel smoke test (backend/image_smoke.py), proved on CPU.

Nothing here touches a GPU: CUDA's absence, the device capability and the compiler are faked,
and the steps that run for real run on ``device="cpu"``.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("image_smoke", BACKEND / "image_smoke.py")
image_smoke = importlib.util.module_from_spec(_spec)
sys.modules["image_smoke"] = image_smoke
_spec.loader.exec_module(image_smoke)

STEP_NAMES = ["bf16 matmul", "bf16 conv2d", "fp8 weight-only", "fp8 dynamic", "torch.compile", "torchaudio mel"]


def _step(name: str) -> image_smoke.Step:
    (step,) = [s for s in image_smoke.steps() if s.name == name]
    return step


def _lines(capsys) -> list[str]:
    return capsys.readouterr().out.splitlines()


@pytest.fixture
def capability(monkeypatch):
    def fake(cc: tuple[int, int]):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: cc)
    return fake


class TestTheSteps:
    def test_the_steps_are_the_kernels_the_image_ships(self):
        assert [s.name for s in image_smoke.steps()] == STEP_NAMES

    def test_only_the_fp8_steps_may_skip(self):
        assert [s.name for s in image_smoke.steps() if s.may_skip] == ["fp8 weight-only", "fp8 dynamic"]


class TestWithoutCuda:
    def test_every_step_fails_and_the_script_exits_non_zero(self, monkeypatch, capsys):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(image_smoke, "_driver_version", lambda: "unknown")
        assert image_smoke.main([]) != 0
        lines = _lines(capsys)
        for name in STEP_NAMES:
            assert any(ln.startswith("FAIL") and name in ln for ln in lines), f"{name} is not reported FAIL"
        assert not any(ln.startswith(("PASS", "SKIP")) for ln in lines)


class TestAStepThatRaises:
    def test_it_is_reported_by_name_and_fails_the_run(self, capsys):
        def boom(device):
            raise RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED")

        ok = image_smoke.run([image_smoke.Step("exploding kernel", boom)], device="cpu")
        assert not ok
        (line,) = [ln for ln in _lines(capsys) if "exploding kernel" in ln]
        assert line.startswith("FAIL") and "CUBLAS_STATUS_NOT_SUPPORTED" in line

    def test_the_other_steps_still_run(self, capsys):
        ran = []

        def boom(device):
            raise RuntimeError("boom")

        steps = [image_smoke.Step("first", boom), image_smoke.Step("second", lambda device: ran.append(device))]
        assert not image_smoke.run(steps, device="cpu")
        assert ran == ["cpu"]
        assert any(ln.startswith("PASS") and "second" in ln for ln in _lines(capsys))

    def test_a_result_that_is_not_finite_fails(self):
        with pytest.raises(AssertionError, match="not finite"):
            image_smoke.compare(torch.tensor([1.0, float("nan")]), torch.tensor([1.0, 1.0]), rtol=0.1)

    def test_a_result_far_from_the_cpu_fails(self):
        with pytest.raises(AssertionError, match="relative error"):
            image_smoke.compare(torch.tensor([2.0, 2.0]), torch.tensor([1.0, 1.0]), rtol=0.1)


class TestTheCompileStep:
    def test_it_fails_when_no_kernel_was_generated(self, monkeypatch, capsys):
        monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)  # runs eagerly: no kernel
        assert not image_smoke.run([_step("torch.compile")], device="cpu")
        (line,) = [ln for ln in _lines(capsys) if "torch.compile" in ln]
        assert line.startswith("FAIL") and "generated_kernel_count" in line


class TestTheFp8Steps:
    @pytest.fixture(autouse=True)
    def no_quantization(self, monkeypatch):
        monkeypatch.setattr(image_smoke, "_fp8_forward", lambda device, config: None)

    @pytest.mark.parametrize("cc", [(7, 5), (8, 0), (8, 6)])
    def test_below_sm89_they_skip_and_the_run_passes(self, capability, capsys, cc):
        capability(cc)
        steps = [_step("fp8 weight-only"), _step("fp8 dynamic")]
        assert image_smoke.run(steps, device="cpu")
        lines = [ln for ln in _lines(capsys) if "fp8" in ln]
        assert len(lines) == 2 and all(ln.startswith("SKIP") for ln in lines)

    @pytest.mark.parametrize("cc", [(8, 9), (9, 0), (10, 0), (12, 0)])
    def test_from_sm89_they_run(self, capability, capsys, cc):
        capability(cc)
        assert image_smoke.run([_step("fp8 weight-only"), _step("fp8 dynamic")], device="cpu")
        lines = [ln for ln in _lines(capsys) if "fp8" in ln]
        assert len(lines) == 2 and all(ln.startswith("PASS") for ln in lines)

    def test_a_skip_from_sm89_is_a_failure(self, capability, capsys):
        capability((9, 0))

        def skips(device):
            raise image_smoke.Skipped("says it cannot run")

        assert not image_smoke.run([image_smoke.Step("fp8 weight-only", skips, may_skip=True)], device="cpu")
        assert _lines(capsys)[-1].startswith("FAIL")

    def test_a_step_that_may_not_skip_fails_when_it_skips(self, capability, capsys):
        capability((7, 5))

        def skips(device):
            raise image_smoke.Skipped("no")

        assert not image_smoke.run([image_smoke.Step("bf16 matmul", skips)], device="cpu")
        assert _lines(capsys)[-1].startswith("FAIL")

    @pytest.mark.parametrize("cc", [(7, 5), (8, 6), (8, 9), (9, 0), (12, 0)])
    def test_the_gate_is_the_products(self, monkeypatch, cc):
        from app.engine.factories.quantizers.torchao import TorchAOBackend

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: cc)
        for scheme in ("fp8", "fp8_dynamic"):
            assert image_smoke.fp8_supported(cc) == TorchAOBackend.is_available(scheme)


class TestTheChecksPassOnAGoodResult:
    """On CPU against CPU the checks must pass, or a GPU's FAIL would mean nothing."""

    @pytest.mark.parametrize("name", ["bf16 matmul", "bf16 conv2d", "torchaudio mel"])
    def test_on_cpu(self, capsys, name):
        assert image_smoke.run([_step(name)], device="cpu"), capsys.readouterr().out


class TestTheHeaderAndTheTimeout:
    def test_the_header_names_driver_gpu_and_versions(self, monkeypatch, capability):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA GeForce RTX 4090")
        monkeypatch.setattr(image_smoke, "_driver_version", lambda: "580.65.06")
        capability((8, 9))
        header = image_smoke.header()
        for part in ("driver 580.65.06", "NVIDIA GeForce RTX 4090", "cc 8.9", f"torch {torch.__version__}",
                     f"CUDA {torch.version.cuda}", "torchao "):
            assert part in header

    def test_the_whole_run_has_a_15_minute_timeout(self):
        assert image_smoke.TIMEOUT_SECONDS == 15 * 60

    def test_the_watchdog_ends_a_hung_run(self, capsys):
        ended = threading.Event()
        codes = []

        def exit_(code):
            codes.append(code)
            ended.set()

        watchdog = image_smoke.start_watchdog(0.05, exit_)
        try:
            assert ended.wait(5)
        finally:
            watchdog.cancel()
        assert codes and codes[0] != 0
        assert "TIMEOUT" in capsys.readouterr().out
