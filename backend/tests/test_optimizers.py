"""
Tests for OptimizerFactory and LRSchedulerFactory.

Covers: AdamW, AdamW8bit fallback, Prodigy fallback, custom betas,
is_adaptive detection, and LR scheduler creation.
"""
import torch
import torch.nn as nn

from app.engine.factories.optimizer import OptimizerFactory, LRSchedulerFactory


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_params():
    """Create a small parameter list for optimizer tests."""
    return [nn.Parameter(torch.randn(4, 4))]


# ── OptimizerFactory ─────────────────────────────────────────────────────


class TestOptimizerFactory:
    """Tests for OptimizerFactory.create()."""

    def test_create_adamw(self):
        """'AdamW' should create a standard PyTorch AdamW."""
        opt = OptimizerFactory.create("AdamW", _make_params(), lr=1e-4)
        assert isinstance(opt, torch.optim.AdamW)

    def test_create_adamw_custom_betas(self):
        """Custom betas should be passed through."""
        opt = OptimizerFactory.create(
            "AdamW", _make_params(), lr=1e-4, betas=(0.95, 0.98)
        )
        assert opt.defaults["betas"] == (0.95, 0.98)

    def test_create_adamw_custom_weight_decay(self):
        """Weight decay should be configurable."""
        opt = OptimizerFactory.create(
            "AdamW", _make_params(), lr=1e-4, weight_decay=0.05
        )
        assert opt.defaults["weight_decay"] == 0.05

    def test_unknown_type_falls_back_to_adamw(self):
        """Unknown optimizer type should fall back to AdamW."""
        opt = OptimizerFactory.create("NonExistentOptimizer", _make_params(), lr=1e-4)
        assert isinstance(opt, torch.optim.AdamW)

    def test_adamw8bit_fallback_without_bnb(self):
        """When bitsandbytes is missing, AdamW8bit should fall back to AdamW."""
        import unittest.mock as mock

        with mock.patch.dict("sys.modules", {"bitsandbytes": None}):
            # Force re-import failure by patching importlib
            with mock.patch("builtins.__import__", side_effect=lambda name, *a, **k: (
                (_ for _ in ()).throw(ImportError()) if name == "bitsandbytes" else __builtins__.__import__(name, *a, **k)
            )):
                opt = OptimizerFactory.create("AdamW8bit", _make_params(), lr=1e-4)
                assert isinstance(opt, torch.optim.AdamW)


class TestIsAdaptive:
    """Tests for OptimizerFactory.is_adaptive()."""

    def test_prodigy_is_adaptive(self):
        """Prodigy should be recognized as adaptive."""
        assert OptimizerFactory.is_adaptive("Prodigy") is True

    def test_prodigy_plus_sf_is_adaptive(self):
        """ProdigyPlusSF should be recognized as adaptive."""
        assert OptimizerFactory.is_adaptive("ProdigyPlusSF") is True

    def test_adamw_is_not_adaptive(self):
        """AdamW should NOT be adaptive."""
        assert OptimizerFactory.is_adaptive("AdamW") is False

    def test_adamw8bit_is_not_adaptive(self):
        """AdamW8bit should NOT be adaptive."""
        assert OptimizerFactory.is_adaptive("AdamW8bit") is False


class TestProdigyPlusSF:
    """Tests for ProdigyPlusScheduleFree optimizer creation."""

    def test_create_prodigy_plus_sf(self):
        """ProdigyPlusSF should create a ProdigyPlusScheduleFree instance."""
        opt = OptimizerFactory.create("ProdigyPlusSF", _make_params(), lr=1.0)
        # Should be ProdigyPlusScheduleFree (not AdamW fallback)
        assert type(opt).__name__ == "ProdigyPlusScheduleFree"

    def test_create_prodigy_plus_sf_custom_params(self):
        """Custom ppsf_ params should be accepted without error."""
        opt = OptimizerFactory.create(
            "ProdigyPlusSF", _make_params(), lr=1.0,
            config={
                "ppsf_d_coef": 0.5,
                "ppsf_factored": False,
                "ppsf_use_stableadamw": False,
                "ppsf_use_cautious": True,
                "ppsf_split_groups": False,
            },
        )
        assert type(opt).__name__ == "ProdigyPlusScheduleFree"

    def test_prodigy_plus_sf_has_eval_train(self):
        """ProdigyPlusSF should have eval() and train() for Schedule-Free mode."""
        opt = OptimizerFactory.create("ProdigyPlusSF", _make_params(), lr=1.0)
        assert hasattr(opt, "eval") and callable(opt.eval)
        assert hasattr(opt, "train") and callable(opt.train)

    def test_prodigy_plus_sf_fallback_without_package(self):
        """When prodigy-plus-schedule-free is missing, should fall back to AdamW."""
        import unittest.mock as mock

        with mock.patch.dict("sys.modules", {"prodigyplus": None}):
            with mock.patch("builtins.__import__", side_effect=lambda name, *a, **k: (
                (_ for _ in ()).throw(ImportError()) if name == "prodigyplus" else __builtins__.__import__(name, *a, **k)
            )):
                opt = OptimizerFactory.create("ProdigyPlusSF", _make_params(), lr=1.0)
                assert isinstance(opt, torch.optim.AdamW)


# ── LRSchedulerFactory ───────────────────────────────────────────────────


class TestLRSchedulerFactory:
    """Tests for LRSchedulerFactory.create()."""

    def _make_optimizer(self):
        return torch.optim.AdamW(_make_params(), lr=1e-4)

    def test_create_constant_scheduler(self):
        """'constant' type should create a valid scheduler."""
        sched = LRSchedulerFactory.create(
            "constant", self._make_optimizer(), num_warmup_steps=10, num_training_steps=100
        )
        assert sched is not None

    def test_create_cosine_scheduler(self):
        """'cosine' type should create a valid scheduler."""
        sched = LRSchedulerFactory.create(
            "cosine", self._make_optimizer(), num_warmup_steps=10, num_training_steps=100
        )
        assert sched is not None

    def test_create_linear_scheduler(self):
        """'linear' type should create a valid scheduler."""
        sched = LRSchedulerFactory.create(
            "linear", self._make_optimizer(), num_warmup_steps=10, num_training_steps=100
        )
        assert sched is not None

    def test_create_unknown_defaults_to_constant(self):
        """Unknown scheduler type should fall back to constant."""
        sched = LRSchedulerFactory.create(
            "mysterious", self._make_optimizer(), num_warmup_steps=5, num_training_steps=50
        )
        assert sched is not None
