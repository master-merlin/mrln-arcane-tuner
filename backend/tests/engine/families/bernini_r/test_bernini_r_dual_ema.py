"""Bernini-R dual-expert EMA tests (W3.T10; no GPU/weights).

Mirrors ``wan22/test_wan22_dual_ema.py``: ``BerniniRTrainer._ema_parameters()``
must return the union of BOTH experts' trainable params (prefixed ``high.``/
``low.``) on a 14B ``both``-mode run, and fall back to ``None`` (base
primary-model EMA behavior) for the 1.3B single-expert path (``driver.is_dual``
False) and for single-expert-mode 14B runs (``expert_mode`` high/low) — the
SAME two guards ``_configure_optimization``/``_apply_peft`` already use for
exactly this dual-expert reason.

Exercises ``_ema_parameters()`` directly against a lightweight fake driver
(plain ``nn.Module`` experts) rather than the full PEFT/router pipeline —
sufficient to pin the method's own branching + prefixing logic, which is all
this override adds on top of the already-tested ``_collect_expert_params``
machinery.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from app.engine.models.families.bernini_r.trainer import BerniniRTrainer


def _fake_expert(dim=4, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(dim, dim, bias=False)


def _bare_trainer(
    is_dual: bool, expert_mode: str, high=None, low=None
) -> BerniniRTrainer:
    t = object.__new__(BerniniRTrainer)
    t.driver = SimpleNamespace(
        is_dual=is_dual,
        transformer_high=high,
        transformer_low=low,
    )
    t.expert_mode = expert_mode
    return t


def test_returns_prefixed_union_for_dual_both_mode():
    high, low = _fake_expert(seed=1), _fake_expert(seed=2)
    t = _bare_trainer(is_dual=True, expert_mode="both", high=high, low=low)

    params = t._ema_parameters()

    assert params is not None
    assert set(params) == {"high.weight", "low.weight"}
    assert params["high.weight"] is high.weight
    assert params["low.weight"] is low.weight


def test_returns_none_for_non_dual_driver():
    """The 1.3B (non-MoE) path: driver.is_dual is False -> base behavior."""
    high = _fake_expert()
    t = _bare_trainer(is_dual=False, expert_mode="both", high=high, low=None)
    assert t._ema_parameters() is None


def test_returns_none_for_single_expert_mode_on_dual_driver():
    """14B dual driver but expert_mode pinned to one expert -> base behavior
    (mirrors the SAME guard _configure_optimization/_apply_peft use)."""
    high = _fake_expert()
    t = _bare_trainer(is_dual=True, expert_mode="high", high=high, low=None)
    assert t._ema_parameters() is None


def test_skips_missing_expert_model():
    """If one expert slot is None (shouldn't happen in a real both-mode run,
    but the method must not crash), only the present expert is returned."""
    high = _fake_expert()
    t = _bare_trainer(is_dual=True, expert_mode="both", high=high, low=None)
    params = t._ema_parameters()
    assert params is not None
    assert set(params) == {"high.weight"}


def test_frozen_params_excluded():
    high, low = _fake_expert(seed=1), _fake_expert(seed=2)
    high.weight.requires_grad = False
    t = _bare_trainer(is_dual=True, expert_mode="both", high=high, low=low)
    params = t._ema_parameters()
    assert params is not None
    assert set(params) == {"low.weight"}
