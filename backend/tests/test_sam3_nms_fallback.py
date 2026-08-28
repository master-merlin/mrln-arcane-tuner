"""SAM3 mask NMS: the "fallback" is triton on the GPU, and triton is required.

`sam3/perflib/nms.py` logs this at import, at DEBUG level, on every start:

    Falling back to triton or CPU mask NMS implementation -- please install
    `torch_generic_nms` via ... pip install git+https://github.com/...

It reads like a performance warning with a safe CPU floor. It is neither, and
both halves of that reading are wrong in ways worth pinning:

1. **The optional CUDA kernel is not worth installing here.** `torch_generic_nms`
   is a git-URL package that compiles against nvcc with a hard-coded
   `TORCH_CUDA_ARCH_LIST="8.0 9.0"`; on Windows it additionally needs MSVC. It
   would be a class C dependency that breaks the ordinary install for everyone
   who cannot compile CUDA, to replace one GPU kernel with another.

2. **There is no CPU fallback on the CUDA path.** Reading `generic_nms`:

       if ious.is_cuda:
           if GENERIC_NMS_AVAILABLE:  -> compiled CUDA kernel
           else:                      -> from sam3.perflib.triton.nms import nms_triton
       return generic_nms_cpu(...)    -> only when the tensors are NOT on cuda

   `generic_nms_cpu` is unreachable while masking runs on a GPU. So if triton
   were ever dropped from requirements.txt, that `else` branch raises
   ImportError **at inference**, inside `sam3_image.py`'s `nms_masks` call —
   masking would not get slower, it would break, and only for GPU users.

That import sits inside an else-branch, so nothing at startup touches it and no
import-time check can see it. This file is the check.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO / "backend" / "requirements.txt"


def test_triton_is_importable():
    """The precondition SAM3's GPU mask NMS depends on.

    `sam3/perflib/triton/nms.py` does a module-level `import triton`, so its
    absence is an ImportError at call time, not a slower path.
    """
    assert importlib.util.find_spec("triton") is not None, (
        "triton is not importable. SAM3 mask NMS on a GPU imports "
        "sam3.perflib.triton.nms inside a branch taken at inference time, so "
        "this breaks masking for GPU users at the moment they use it — with no "
        "CPU fallback, despite what sam3's own log line implies."
    )


def test_requirements_pins_triton_for_both_platforms():
    """Two markered lines, and dropping either breaks only that platform.

    A single-platform pin would leave the other's users with the ImportError
    above, and a Windows maintainer would never see a Linux regression (or the
    reverse) because the marker hides it from their own install.
    """
    text = REQUIREMENTS.read_text(encoding="utf-8")
    lines = [
        ln for ln in text.splitlines()
        if re.match(r"^\s*triton(-windows)?\s*==", ln, re.I)
    ]
    assert lines, "requirements.txt pins no triton at all"

    platforms = " ".join(lines).lower()
    assert "win32" in platforms, (
        "no triton pin carries a win32 marker; Windows GPU users would hit an "
        f"ImportError in SAM3 mask NMS. Lines found: {lines}"
    )
    assert "linux" in platforms, (
        "no triton pin carries a linux marker; Linux and the container would "
        f"hit an ImportError in SAM3 mask NMS. Lines found: {lines}"
    )


def test_the_cuda_path_still_has_no_cpu_fallback():
    """Pins the upstream shape this file's reasoning rests on.

    If a future sam3 gives the CUDA branch a real CPU fallback, triton stops
    being load-bearing and the two tests above become stricter than they need
    to be — which is worth knowing rather than carrying forever. If instead the
    structure holds, this documents *why* it holds against anyone reading the
    reassuring log line and concluding the fallback is safe.
    """
    spec = importlib.util.find_spec("sam3.perflib.nms")
    if spec is None or not spec.origin:
        pytest.skip("sam3 is not installed in this environment")

    source = Path(spec.origin).read_text(encoding="utf-8")
    match = re.search(r"def generic_nms\(.*?\n(?=\ndef |\Z)", source, re.S)
    assert match, "sam3.perflib.nms.generic_nms not found — its shape changed"
    body = match.group(0)

    cuda_branch = body.split("if ious.is_cuda:", 1)
    assert len(cuda_branch) == 2, (
        "generic_nms no longer branches on `ious.is_cuda`; re-read it before "
        "trusting this file's reasoning about triton being required"
    )
    after = cuda_branch[1]
    triton_import = after.find("nms_triton")
    cpu_call = after.find("generic_nms_cpu")
    assert triton_import != -1, "the CUDA branch no longer reaches nms_triton"
    assert cpu_call == -1 or cpu_call > triton_import, (
        "generic_nms_cpu now appears inside the CUDA branch — there may be a "
        "real CPU fallback, which would make triton optional. Re-check."
    )
