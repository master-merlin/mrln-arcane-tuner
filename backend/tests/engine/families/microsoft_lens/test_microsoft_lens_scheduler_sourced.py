"""W3-1: microsoft_lens sampling scheduler is definition-sourced.

The sampler used to hard-code the FlowMatchEulerDiscreteScheduler config
(shift 3.0, dynamic shifting, exponential time-shift, 1000 train timesteps,
base/max shift 0.5/1.15, base/max image_seq_len 256/4096). Those values live
in lens_base.yaml's ``scheduler.*`` architecture_params; the sampler now reads
them from there. Pure plumbing — the shipped values are byte-identical.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import torch

from app.engine.models.families.microsoft_lens.sampler import MicrosoftLensSampler


def _build_sampler(arch: dict):
    sampler = object.__new__(MicrosoftLensSampler)
    sampler._scheduler = None
    pipeline = MagicMock()
    pipeline.device = torch.device("cpu")
    defn = MagicMock()
    defn.architecture_params = dict(arch)
    pipeline.definition = defn
    sampler.pipeline = pipeline
    return sampler


def _shipped_arch() -> dict:
    import yaml

    path = (
        pathlib.Path(__file__).resolve().parents[4]  # .../backend
        / "app" / "engine" / "models" / "families" / "microsoft_lens"
        / "definitions" / "lens_base.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))["architecture_params"]


class TestSchedulerDefinitionSourced:
    def test_shipped_yaml_values_reach_the_scheduler(self):
        """The real lens_base.yaml scheduler.* keys build the exact scheduler
        the sampler used to hard-code (byte-identical plumbing)."""
        sampler = _build_sampler(_shipped_arch())
        sched = sampler._get_scheduler()
        c = sched.config
        assert c.num_train_timesteps == 1000
        assert c.shift == 3.0
        assert c.use_dynamic_shifting is True
        assert c.base_shift == 0.5
        assert c.max_shift == 1.15
        assert c.base_image_seq_len == 256
        assert c.max_image_seq_len == 4096
        assert c.time_shift_type == "exponential"

    def test_scheduler_is_definition_sourced_not_hardcoded(self):
        """Sentinel scheduler.* values must flow through to the scheduler,
        proving the config is read from the definition, not hard-coded."""
        sampler = _build_sampler({
            "scheduler.num_train_timesteps": 999,
            "scheduler.shift": 7.0,
            "scheduler.use_dynamic_shifting": False,
            "scheduler.base_shift": 0.25,
            "scheduler.max_shift": 2.5,
            "scheduler.base_image_seq_len": 128,
            "scheduler.max_image_seq_len": 2048,
            "scheduler.time_shift_type": "linear",
        })
        c = sampler._get_scheduler().config
        assert c.num_train_timesteps == 999
        assert c.shift == 7.0
        assert c.use_dynamic_shifting is False
        assert c.base_shift == 0.25
        assert c.max_shift == 2.5
        assert c.base_image_seq_len == 128
        assert c.max_image_seq_len == 2048
        assert c.time_shift_type == "linear"

    def test_shipped_yaml_carries_scheduler_config(self):
        arch = _shipped_arch()
        assert arch["scheduler.shift"] == 3.0
        assert arch["scheduler.time_shift_type"] == "exponential"
        assert arch["scheduler.base_shift"] == 0.5
        assert arch["scheduler.max_shift"] == 1.15
        assert arch["scheduler.base_image_seq_len"] == 256
        assert arch["scheduler.max_image_seq_len"] == 4096
        assert arch["scheduler.num_train_timesteps"] == 1000
        assert arch["scheduler.use_dynamic_shifting"] is True
