"""LANE-150 parity: hidream_o1's UniPC sigma grid on diffusers 0.41.0's stock scheduler.

``app/engine/tests/fixtures/hidream_o1/unipc_parity.npz`` holds the sigmas, the int64
timesteps and a 20-step fake-model trajectory the vendored ``FlowUniPCMultistepScheduler``
produced as ``vendor/pipeline.py::build_scheduler`` built it (the ``"default"`` branch), for
(shift 3.0, 20 steps), (3.0, 50) and (6.0, 20) -- shift 3.0 is the sampler's default
(pipeline.py ``generate_image(shift=3.0)``). The fixture is written from the vendored class
before its retirement (LANE-150) and is frozen; ``build_scheduler`` now builds the stock class.

Why the explicit grid: the vendored class shifts its 1000-entry sigma table once in
``__init__`` (fm_solvers_unipc.py:108-116) and the linspace drawn from it again in
``set_timesteps`` (:184-189); the stock ``UniPCMultistepScheduler(use_flow_sigmas=True)``
shifts an explicit table exactly once. Feeding it ``linspace(sigma_max, 0, n + 1)[:-1]``
with the PRE-shifted endpoint ``sigma_max = shift * s0 / (1 + (shift - 1) * s0)``,
``s0 = 1 - 1/1000`` (the vendored table's first entry before its shift) reproduces upstream's
grid; the stock default grid differs by ~2.4e-3 in sigma.

``lora_keys.json`` pins the key list ``HiDreamO1Saver`` writes for the tiny model of
test_hidream_o1_family.py (``test_saver_keys_use_diffusion_model_prefix_kohya_style``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

_FIXTURES = Path(__file__).resolve().parents[1] / "app" / "engine" / "tests" / "fixtures" / "hidream_o1"
PARITY_NPZ = _FIXTURES / "unipc_parity.npz"
LORA_KEYS = _FIXTURES / "lora_keys.json"
ATOL = 1e-6
NUM_TRAIN_TIMESTEPS = 1000
CASES = [(3.0, 20), (3.0, 50), (6.0, 20)]
TRAJ_STEPS = 20
LATENT_SHAPE = (1, 4, 8, 8)


def _tag(shift: float, n: int) -> str:
    return f"shift={shift},n={n}"


def _release_scheduler(shift: float, n: int):
    from diffusers import UniPCMultistepScheduler

    s0 = 1.0 - 1.0 / NUM_TRAIN_TIMESTEPS
    sigma_max = shift * s0 / (1.0 + (shift - 1.0) * s0)
    sched = UniPCMultistepScheduler(
        use_flow_sigmas=True, prediction_type="flow_prediction", flow_shift=shift
    )
    sched.set_timesteps(device="cpu", sigmas=np.linspace(sigma_max, 0.0, n + 1)[:-1])
    return sched


def _trajectory(sched) -> np.ndarray:
    """Run TRAJ_STEPS steps of a fake flow model ``v = -0.1 * x`` from a seeded start, in float64.

    float64 isolates the GRID: the two schedulers' sigmas agree to 1 float32 ulp (5.96e-8),
    which moves this trajectory by ~1.2e-7; in float32 the two step implementations' different
    operation order alone adds up to ~6 ulps of the |z| ~ 2.9 result (measured 1.67e-6), noise
    the 1e-6 tolerance would mistake for a grid change.
    """
    g = torch.Generator().manual_seed(150)
    z = torch.randn(LATENT_SHAPE, generator=g, dtype=torch.float32).to(torch.float64)
    for t in sched.timesteps[:TRAJ_STEPS]:
        model_output = -z * 0.1
        z = sched.step(model_output, t.to(torch.float32), z, return_dict=False)[0]
    return z.numpy()


def _outputs(make) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for shift, n in CASES:
        tag = _tag(shift, n)
        out[f"sigmas[{tag}]"] = make(shift, n).sigmas.float().numpy()
        ts = make(shift, n).timesteps
        assert ts.dtype == torch.int64, f"LANE-150 parity hidream_o1/timesteps[{tag}]: dtype {ts.dtype} != int64"
        out[f"timesteps[{tag}]"] = ts.numpy()
        out[f"trajectory[{tag}]"] = _trajectory(make(shift, n))
    return out


def _stored() -> dict[str, np.ndarray]:
    assert PARITY_NPZ.exists(), f"LANE-150 parity hidream_o1: frozen fixture missing: {PARITY_NPZ}"
    with np.load(PARITY_NPZ) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def release_outputs():
    return _outputs(_release_scheduler)


@pytest.mark.parametrize("kind", ["sigmas", "timesteps", "trajectory"])
@pytest.mark.parametrize("shift,n", CASES)
def test_hidream_o1_release_unipc_matches_vendored_fixture(kind, shift, n, release_outputs):
    name = f"{kind}[{_tag(shift, n)}]"
    ref = _stored()[name]
    out = release_outputs[name]
    assert out.shape == ref.shape, (
        f"LANE-150 parity hidream_o1/{name}: shape {tuple(out.shape)} != fixture shape {tuple(ref.shape)}"
    )
    diff = float(np.abs(out.astype(np.float64) - ref.astype(np.float64)).max())
    assert diff <= ATOL, f"LANE-150 parity hidream_o1/{name}: max|release - fixture| = {diff:.3e} > {ATOL:g}"


def test_hidream_o1_lora_keys_pinned(tmp_path):
    import torch.nn as nn
    from safetensors import safe_open

    from app.engine.models.families.hidream_o1.lora_wrapper import inject_lora_layers
    from app.engine.models.families.hidream_o1.saver import HiDreamO1Saver

    class Mini(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Sequential(nn.Linear(8, 8))

    torch.manual_seed(150)
    m = Mini()
    inject_lora_layers(m, rank=2, alpha=2.0)
    out_dir = tmp_path / "k"
    HiDreamO1Saver().save_lora(model=m, out_dir=str(out_dir), name="k", metadata={"rank": 2, "alpha": 2.0})
    with safe_open(str(out_dir / "k.safetensors"), framework="pt") as f:
        keys = sorted(f.keys())
    pinned = json.loads(LORA_KEYS.read_text(encoding="utf-8"))
    extra, missing = sorted(set(keys) - set(pinned)), sorted(set(pinned) - set(keys))
    assert keys == pinned, (
        f"LANE-150 lora keys hidream_o1: saved keys differ from fixtures/hidream_o1/lora_keys.json: "
        f"+{extra[:5]} -{missing[:5]} ({len(extra)} extra, {len(missing)} missing)"
    )


def test_hidream_o1_flash_scheduler_name_is_retired():
    from app.engine.models.families.hidream_o1.vendor.pipeline import build_scheduler

    with pytest.raises(ValueError, match=r"^Unknown scheduler_name='flash'"):
        build_scheduler(5, None, 3.0, "cpu", scheduler_name="flash")


def test_hidream_o1_generate_image_has_no_noise_schedule_parameters():
    import inspect

    from app.engine.models.families.hidream_o1.vendor.pipeline import generate_image

    params = set(inspect.signature(generate_image).parameters)
    retired = {"noise_clip_std", "noise_scale_start", "noise_scale_end"} & params
    assert not retired, f"LANE-150 hidream_o1: retired generate_image parameters still present: {sorted(retired)}"


@pytest.mark.parametrize("shift,n", CASES)
def test_hidream_o1_build_scheduler_matches_fixture(shift, n):
    from app.engine.models.families.hidream_o1.vendor.pipeline import build_scheduler

    sched = build_scheduler(n, None, shift, "cpu", scheduler_name="default")
    ref = _stored()[f"sigmas[{_tag(shift, n)}]"]
    diff = float(np.abs(sched.sigmas.float().numpy().astype(np.float64) - ref.astype(np.float64)).max())
    assert diff <= ATOL, f"LANE-150 parity hidream_o1/build_scheduler[{_tag(shift, n)}]: max diff {diff:.3e} > {ATOL:g}"
