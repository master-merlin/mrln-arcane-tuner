"""LANE-168 -- the help texts and the guide say which engine fields a family applies.

The in-app "?" source (`frontend/public/config_help.json`), the schema descriptions
(`models/base.py`) and the generated guide (`docs/training-guide.md`) must agree with
the visibility rules: min_snr_gamma is SDXL-only, noise_offset is not offered for
HiDream-O1 (and MiniMax-H3 refuses it), timestep_sampling has HiDream-O1 bounds.
The `guide` tests assert on the committed guide and pass once the guide is regenerated.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.engine.models.base import BaseTrainingConfig

REPO = Path(__file__).resolve().parents[2]
HELP = json.loads((REPO / "frontend" / "public" / "config_help.json").read_text(encoding="utf-8"))
GUIDE = (REPO / "docs" / "training-guide.md").read_text(encoding="utf-8")


def _desc(name: str) -> str:
    return BaseTrainingConfig.model_fields[name].description or ""


def _guide_block(heading: str) -> str:
    m = re.search(rf"^### {re.escape(heading)}\n(.*?)(?=^#{{2,3}} )", GUIDE, re.S | re.M)
    assert m, f"guide has no '### {heading}' block"
    return m.group(1)


def test_min_snr_help_says_sdxl_only():
    entry = HELP["min_snr_gamma"]
    assert "SDXL only" in entry["tip"]
    assert "SDXL only" in entry["detail"] and "hidden" in entry["detail"]
    assert "SDXL only" in _desc("min_snr_gamma")


def test_noise_offset_help_names_hidream_and_minimax():
    entry = HELP["noise_offset"]
    for text in (entry["detail"], _desc("noise_offset")):
        assert "HiDream-O1" in text and "MiniMax-H3" in text
    assert "patch-space" in entry["detail"]


def test_timestep_sampling_help_states_hidream_bounds():
    for text in (HELP["timestep_sampling"]["detail"], _desc("timestep_sampling")):
        assert "HiDream-O1" in text
        assert "flux_shift" in text and "model_shift" in text and "uniform" in text
        assert "radc_res_influence" in text or "radc_res_influence" in HELP["timestep_sampling"]["detail"]


def test_guide_engine_block_carries_min_snr_and_noise_offset_statements():
    block = _guide_block("Advanced Engine")
    snr = next(line for line in block.splitlines() if "`min_snr_gamma`" in line)
    off = next(line for line in block.splitlines() if "`noise_offset`" in line)
    assert "SDXL only" in snr
    assert "HiDream-O1" in off and "MiniMax-H3" in off


def test_guide_strategy_block_carries_hidream_bounds():
    block = _guide_block("Training Dynamics")
    line = next(line for line in block.splitlines() if "`timestep_sampling`" in line)
    assert "HiDream-O1" in line and "radc_res_influence" in line


def test_guide_radc_limits_no_longer_calls_bernini_progressless():
    m = re.search(r"\*\*Limits\.\*\*(.*?)(?=^## )", GUIDE, re.S | re.M)
    assert m, "guide lost the RADC Limits paragraph"
    limits = " ".join(m.group(1).split())
    assert "WAN 2.2" in limits
    assert "Bernini-R" not in limits.split("FLUX2")[0]
