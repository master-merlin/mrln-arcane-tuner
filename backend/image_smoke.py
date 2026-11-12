"""The container image's GPU kernel smoke test: does every CUDA library the image ships run here?

    python image_smoke.py

Each step runs one kind of kernel on the GPU, waits for it (``torch.cuda.synchronize()``), and
compares its result with the same op on the CPU: finite, and close. It prints PASS, FAIL or SKIP
per step, and exits 0 only when every step passed. SKIP is allowed for the two fp8 steps alone,
and only on a GPU below SM89, where the product offers no fp8 either; anywhere else a skip is a
FAIL. The whole run has a 15-minute timeout, since a hung kernel cannot be interrupted.

It needs no model and no dataset, so the same run proves an image on any host: the first line
names the driver, the GPU and the versions it ran with.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

import torch

TIMEOUT_SECONDS = 15 * 60
TIMED_OUT = 124

#: The product's fp8 gate (TorchAOBackend.is_available): Ada, Hopper and Blackwell, SM 8.9 and up.
FP8_MIN_CC = (8, 9)


class Skipped(Exception):
    """The step cannot run on this GPU; the runner decides whether that may pass."""


@dataclass(frozen=True)
class Step:
    name: str
    fn: Callable[[str], None]  # runs on the device it is given; raises on a failure
    may_skip: bool = False


def fp8_supported(cc: tuple[int, int]) -> bool:
    return tuple(cc) >= FP8_MIN_CC


def compare(got: torch.Tensor, want: torch.Tensor, rtol: float) -> None:
    """``got`` (any device) is finite and within ``rtol`` of the CPU's ``want``, by relative norm."""
    got = got.detach().float().cpu()
    want = want.detach().float().cpu()
    assert got.shape == want.shape, f"shape {tuple(got.shape)}, the CPU gives {tuple(want.shape)}"
    assert torch.isfinite(got).all(), "the result is not finite"
    error = ((got - want).norm() / want.norm().clamp_min(1e-12)).item()
    assert error <= rtol, f"relative error {error:.3g} against the CPU, above {rtol}"


def _done(device: str) -> None:
    """The kernels have finished (an asynchronous failure surfaces here, not after a PASS)."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _inputs(*shapes: tuple[int, ...]) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return [torch.randn(*shape, generator=generator) for shape in shapes]


def _bf16_matmul(device: str) -> None:  # cuBLAS
    a, b = _inputs((512, 512), (512, 512))
    got = a.to(device, torch.bfloat16) @ b.to(device, torch.bfloat16)
    _done(device)
    compare(got, a @ b, rtol=2e-2)


def _bf16_conv2d(device: str) -> None:  # cuDNN
    x, w = _inputs((4, 16, 32, 32), (32, 16, 3, 3))
    got = torch.nn.functional.conv2d(x.to(device, torch.bfloat16), w.to(device, torch.bfloat16), padding=1)
    _done(device)
    compare(got, torch.nn.functional.conv2d(x, w, padding=1), rtol=2e-2)


def _fp8_forward(device: str, config: str) -> None:
    """The product's fp8 scheme ``config`` on a Linear: quantized, then run forward."""
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Float8WeightOnlyConfig,
        quantize_,
    )

    configs = {"fp8": Float8WeightOnlyConfig, "fp8_dynamic": Float8DynamicActivationFloat8WeightConfig}
    weight, x = _inputs((256, 256), (32, 256))
    weight = weight / 16
    module = torch.nn.Sequential(torch.nn.Linear(256, 256, bias=False))
    with torch.no_grad():
        module[0].weight.copy_(weight)
    module = module.to(device=device, dtype=torch.bfloat16)
    quantize_(module, configs[config]())
    with torch.no_grad():
        got = module(x.to(device, torch.bfloat16))
    _done(device)
    compare(got, x @ weight.T, rtol=0.1)


def _fp8_step(config: str) -> Callable[[str], None]:
    def step(device: str) -> None:
        cc = tuple(torch.cuda.get_device_capability())
        if not fp8_supported(cc):
            raise Skipped(f"fp8 needs SM{FP8_MIN_CC[0]}{FP8_MIN_CC[1]}+, this GPU is cc {cc[0]}.{cc[1]}")
        _fp8_forward(device, config)
    return step


def _compiled(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) * 2 + torch.cos(x) ** 2


def _compile(device: str) -> None:  # inductor, through a generated triton kernel
    import torch._dynamo
    from torch._inductor import metrics

    (x,) = _inputs((1024, 1024))
    torch._dynamo.reset()
    metrics.reset()
    got = torch.compile(_compiled, backend="inductor", fullgraph=True)(x.to(device))
    _done(device)
    count = metrics.generated_kernel_count
    assert count > 0, f"torch.compile generated no kernel (torch._inductor.metrics.generated_kernel_count = {count})"
    compare(got, _compiled(x), rtol=1e-4)


def _mel(device: str) -> None:  # torchaudio on CUDA (cuFFT)
    import torchaudio

    (wave,) = _inputs((1, 16000))
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=64)
    got = mel.to(device)(wave.to(device))
    _done(device)
    compare(got, mel.cpu()(wave), rtol=1e-3)


def steps() -> list[Step]:
    return [
        Step("bf16 matmul", _bf16_matmul),
        Step("bf16 conv2d", _bf16_conv2d),
        Step("fp8 weight-only", _fp8_step("fp8"), may_skip=True),
        Step("fp8 dynamic", _fp8_step("fp8_dynamic"), may_skip=True),
        Step("torch.compile", _compile),
        Step("torchaudio mel", _mel),
    ]


def run(to_run: list[Step], device: str = "cuda", out: Callable[[str], None] = print) -> bool:
    """Every step in turn, each line PASS, FAIL or SKIP; True when none failed."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        for step in to_run:
            out(f"FAIL  {step.name}: CUDA is not available")
        return False
    ok = True
    for step in to_run:
        start = time.monotonic()
        try:
            step.fn(device)
        except Skipped as skipped:
            cc = tuple(torch.cuda.get_device_capability())
            if step.may_skip and not fp8_supported(cc):
                out(f"SKIP  {step.name}: {skipped}")
                continue
            ok = False
            out(f"FAIL  {step.name}: skipped where it must run (cc {cc[0]}.{cc[1]}): {skipped}")
        except Exception as error:  # noqa: BLE001 — every failure is a FAIL line, by name
            ok = False
            out(f"FAIL  {step.name}: {type(error).__name__}: {error}")
        else:
            out(f"PASS  {step.name} ({time.monotonic() - start:.2f} s)")
    return ok


def _driver_version() -> str:
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return "unknown (no nvidia-smi)"
    lines = done.stdout.split()
    return lines[0] if done.returncode == 0 and lines else f"unknown (nvidia-smi exit {done.returncode})"


def header() -> str:
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        gpu = f"{torch.cuda.get_device_name(0)} (cc {cc[0]}.{cc[1]})"
    else:
        gpu = "no CUDA device"
    try:
        import torchao

        torchao_version = torchao.__version__
    except Exception as error:  # noqa: BLE001 — the header reports it; the fp8 steps then fail
        torchao_version = f"not importable ({type(error).__name__})"
    return (f"image smoke: driver {_driver_version()} | GPU {gpu} | torch {torch.__version__}"
            f" | CUDA {torch.version.cuda} | torchao {torchao_version}")


def start_watchdog(seconds: float, exit_: Callable[[int], None] = os._exit) -> threading.Timer:
    """Ends the process after ``seconds``: a hung kernel blocks the main thread for good."""
    def fire() -> None:
        print(f"TIMEOUT: the smoke test did not finish in {seconds:g} s", flush=True)
        exit_(TIMED_OUT)

    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()
    return timer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    watchdog = start_watchdog(TIMEOUT_SECONDS)
    try:
        print(header(), flush=True)
        ok = run(steps())
        print("PASSED: every step" if ok else "FAILED: see the FAIL lines", flush=True)
        return 0 if ok else 1
    finally:
        watchdog.cancel()


if __name__ == "__main__":
    sys.exit(main())
