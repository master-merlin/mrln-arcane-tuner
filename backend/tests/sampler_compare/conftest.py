"""Shared builders for the sampler comparison's tests."""
import json
from pathlib import Path

import pytest

from app.core.sampler_compare import plan

CONFIG = {
    "version": 1,
    "name": "cmp",
    "reference_job": "job-1",
    "dataset": "Cars",
    "arms": {
        "uniform": {"timestep_sampling": "uniform"},
        "radc": {"timestep_sampling": "radc", "radc_start": 0.9, "radc_end": 0.3},
    },
    "seeds": [42, 43],
    "steps": 2000,
    "validation": {"every": 500, "noise_draws": 4, "holdout_fraction": 0.3, "split_seed": 0},
    "port": 8001,
    "limits": {"temperature_c": 94, "vram_mib": 88000, "free_ram_gib": 1, "stall_minutes": 15,
               "run_minutes": 120, "idle_vram_mib": 20000, "idle_wait_minutes": 30, "power_limit_w": 500},
}


def write_config(folder: Path, **changes) -> Path:
    config = json.loads(json.dumps(CONFIG))
    config.setdefault("work_dir", str(folder / "work"))
    config.update(changes)
    path = folder / "comparison.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.fixture
def make_comparison(tmp_path):
    def make(**changes) -> plan.Comparison:
        return plan.load_comparison(write_config(tmp_path, **changes))

    return make
