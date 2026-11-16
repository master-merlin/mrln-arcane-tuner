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


def _radc_section() -> str:
    return _guide_block("Timestep sampling — Resolution-Aware Dynamic Curriculum (RADC)")


def test_radc_help_names_the_models_noise_scale():
    for name in ("radc_start", "radc_end"):
        assert "model's own noise scale" in _desc(name), name
        assert "noise scale" in HELP[name]["tip"] and "model's own noise scale" in HELP[name]["detail"], name


def test_config_help_radc_entries_name_the_noise_scale():
    entry = HELP["radc_uniform_prob"]
    assert entry["tip"].strip() and "uniform" in entry["tip"].lower()
    assert "noise scale" in entry["detail"]
    assert "radc_uniform_prob" in _desc("radc_uniform_prob") or "uniform" in _desc("radc_uniform_prob").lower()


def test_guide_radc_section_states_the_changed_focus():
    section = " ".join(_radc_section().split())
    assert "model's own noise scale" in section
    for family in ("FLUX.3", "Bernini-R", "MiniMax-H3", "Boogu-Image", "Ideogram 4", "FLUX.2"):
        assert family in section, family
    assert "0.98" in section and "0.75" in section, "the guide must give FLUX.3's old effective range"
    assert "no longer weights the loss" in section


def test_guide_lists_radc_uniform_prob():
    assert "`radc_uniform_prob`" in _radc_section()
    block = _guide_block("Training Dynamics")
    assert any("`radc_uniform_prob`" in line for line in block.splitlines()), "the field reference was not regenerated"


def test_guide_says_the_share_is_off_by_default_and_what_was_measured():
    assert BaseTrainingConfig.model_fields["radc_uniform_prob"].default == 0.0
    section = " ".join(_radc_section().split())
    assert "off by default because it was measured" in section
    assert "+0.00002" in section and "0.00047" in section, "the guide must give the share's measured gap and the floor"
    assert "not a measured result" not in section, "the guide still calls RADC's gain unmeasured"


def test_guide_lists_no_expert_family_as_progressless():
    # WAN 2.2 and Bernini-R both follow the curriculum, so the Limits paragraph is gone.
    assert "**Limits.**" not in GUIDE
    assert "**On WAN 2.2,** `radc` also decides how often each expert trains" in " ".join(GUIDE.split())
