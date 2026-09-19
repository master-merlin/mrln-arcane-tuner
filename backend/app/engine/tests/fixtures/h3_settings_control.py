"""POSITIVE CONTROL for ``test_minimax_h3_settings.py::test_only_settings_reads_config_keys``.

This file is NOT production code and is never imported by the engine. It
deliberately reads the H3 run-config / ``architecture_params`` keys the way a
consumer must NOT (RULE-21: ``families/minimax_h3/settings.py`` is the ONE
resolver), so the source guard has an offender it is REQUIRED to flag. If the
guard ever stops naming this file, the guard is broken, not the tree.

Plan row 1.0 (``_harness/plans/2026-09-12-minimax-h3-pr1.md``).
"""

from __future__ import annotations

from typing import Any


def bad_reads(config: dict[str, Any], arch: dict[str, Any]) -> tuple[Any, ...]:
    # Each of these is a direct read a consumer must not make.
    return (
        config.get("audio_loss_weight"),
        config.get("sigma_shift_video"),
        config.get("cfg_augment_scale"),
        config.get("train_audio"),
        arch["video.sigma_shift"],
        arch["audio.loss_weight"],
        arch.get("cfg_augment.scale"),
    )
