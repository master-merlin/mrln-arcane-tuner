"""Bernini-R's 14B resumes its router and its experts as WAN 2.2 does.

The checks live in ``test_wan22_router_resume.py``; this module runs them on Bernini-R's own trainer
and driver, whose router estimates through its own warped draw.
"""

from __future__ import annotations

import pytest

from app.engine.models.families.bernini_r.driver import BerniniRDriver
from app.engine.models.families.bernini_r.trainer import BerniniRTrainer
from app.engine.models.families.wan22.expert_router import HIGH, LOW
from tests.engine.families.wan22.test_wan22_router_resume import (
    Family,
    check_decisions_continue,
    check_missing_file_warns_once,
    check_mismatch_leaves_the_built_router,
    check_only_a_unet_folder_loads_into_the_active_expert,
    check_own_weights,
    check_right_expert_is_resident,
    check_save_writes_the_router_and_readers_accept_it,
    check_today_layout_resumes,
)


class _BerniniDefn:
    id = "bernini_r_14b"
    architecture_params = {"mode": "t2v", "dual_expert": True, "switch_dit_boundary": 0.875}
    lora_targetable_modules: list[str] = []
    defaults: dict = {"num_frames": 81}


BERNINI = Family(BerniniRTrainer, BerniniRDriver, _BerniniDefn)


@pytest.mark.parametrize("swap", ["resident", "swap"])
@pytest.mark.parametrize(
    ("mode", "interval", "soft_stop"),
    [("mode", 1, False), ("radc", 1, False), ("mode", 3, False), ("radc", 1, True)],
    ids=["mode", "radc", "interval3", "radc-soft-stop"],
)
def test_a_resumed_run_takes_the_decisions_of_an_uninterrupted_one(tmp_path, swap, mode, interval, soft_stop):
    check_decisions_continue(BERNINI, tmp_path, mode=mode, swap=swap, interval=interval, soft_stop=soft_stop)


@pytest.mark.parametrize("swap", ["resident", "swap"])
def test_the_right_expert_is_active_and_resident_at_the_first_step(tmp_path, swap):
    check_right_expert_is_resident(BERNINI, tmp_path, swap=swap)


def test_the_save_path_writes_the_router_and_its_readers_accept_it(tmp_path):
    check_save_writes_the_router_and_readers_accept_it(BERNINI, tmp_path)


def test_a_checkpoint_without_the_file_warns_once(tmp_path):
    check_missing_file_warns_once(BERNINI, tmp_path)


@pytest.mark.parametrize(
    ("setting", "changed", "expert_mode"),
    [
        ("seed", {"seed": 8}, "both"),
        ("switch_interval", {"expert_switch_interval": 3}, "both"),
        ("pinned_expert", {}, "low"),
    ],
    ids=["seed", "interval", "pinned"],
)
def test_a_mismatched_setting_leaves_the_built_router(tmp_path, setting, changed, expert_mode):
    check_mismatch_leaves_the_built_router(BERNINI, tmp_path, setting=setting, changed=changed, expert_mode=expert_mode)


@pytest.mark.parametrize("router_file", [True, False], ids=["router-file", "no-router-file"])
@pytest.mark.parametrize("at_save", [HIGH, LOW])
@pytest.mark.parametrize("startup", [HIGH, LOW])
def test_each_expert_ends_with_its_own_weights(tmp_path, startup, at_save, router_file):
    check_own_weights(BERNINI, tmp_path, startup=startup, at_save=at_save, router_file=router_file)


@pytest.mark.parametrize("startup", [HIGH, LOW])
def test_only_a_unet_folder_loads_into_the_active_expert(tmp_path, startup):
    check_only_a_unet_folder_loads_into_the_active_expert(BERNINI, tmp_path, startup=startup)


def test_a_folder_in_todays_layout_resumes(tmp_path):
    check_today_layout_resumes(BERNINI, tmp_path)
