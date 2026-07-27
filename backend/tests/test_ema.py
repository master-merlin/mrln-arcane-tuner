"""
Tests for EMAHandler — covers init, step, store_and_swap, restore, state_dict.
"""

import torch
import torch.nn as nn

import app.engine.strategies.ema as ema_module
from app.engine.strategies.ema import EMAHandler


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict]] = []
        self.debugs: list[tuple[str, dict]] = []

    def warning(self, event, **kw):
        self.warnings.append((event, kw))

    def debug(self, event, **kw):
        self.debugs.append((event, kw))

    def info(self, event, **kw):
        pass


class _LevelGatedLogger:
    """Fake structlog logger honoring ``isEnabledFor`` (mirrors
    ``structlog.stdlib.BoundLogger`` — the project's real wrapper class,
    which proxies straight through to the underlying stdlib ``Logger``)."""

    def __init__(self, debug_enabled: bool) -> None:
        self._debug_enabled = debug_enabled
        self.debug_calls: list[tuple[str, dict]] = []

    def isEnabledFor(self, level: int) -> bool:  # noqa: N802 - matches stdlib API
        return self._debug_enabled

    def debug(self, event, **kw):
        self.debug_calls.append((event, kw))

    def info(self, event, **kw):
        pass

    def warning(self, event, **kw):
        pass


def _make_model():
    """Simple linear model for EMA testing."""
    m = nn.Linear(4, 2, bias=False)
    nn.init.ones_(m.weight)
    return m


class TestEMAInit:
    def test_shadow_clones_trainable_params(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.99)
        assert len(ema.shadow) == 1  # weight only, no bias
        assert torch.allclose(ema.shadow["weight"], model.weight.data)

    def test_ignores_frozen_params(self):
        model = _make_model()
        model.weight.requires_grad = False
        ema = EMAHandler(model, decay=0.99)
        assert len(ema.shadow) == 0


class TestEMAStep:
    def test_step_updates_shadow(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.5)
        # Change model weights
        model.weight.data.fill_(0.0)
        ema.step()
        # shadow = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        assert torch.allclose(ema.shadow["weight"], torch.full_like(model.weight.data, 0.5))

    def test_multiple_steps_converge(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.9)
        model.weight.data.fill_(0.0)
        for _ in range(100):
            ema.step()
        # After many steps, shadow should converge toward model (0.0)
        assert ema.shadow["weight"].abs().max().item() < 0.01

    def test_step_increments_counter(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.99)
        ema.step()
        ema.step()
        assert ema._step_count == 2


class TestEMAStepEfficiency:
    """W5.T2: cached {name: param} mapping + in-place lerp_ + gated telemetry.

    ``step()`` used to walk ALL of ``_named_parameters()`` every call
    (name/dict-membership test per tensor) and allocate a fresh tensor per
    shadow param before ``copy_``. This pins: (1) the update is numerically
    equivalent to the original ``(1-d)*p + d*s`` formula, (2) step() no
    longer re-walks ``_named_parameters()`` at all (only the cached
    shadow-sized mapping), (3) the periodic norm telemetry is gated behind
    the logger's DEBUG level.
    """

    def test_step_matches_reference_formula(self):
        # Non-power-of-two decay/values so floating-point reordering
        # differences (if any) would actually show up, not cancel out.
        torch.manual_seed(0)
        model = _make_model()
        nn.init.normal_(model.weight)
        decay = 0.837
        ema = EMAHandler(model, decay=decay)
        original_shadow = ema.shadow["weight"].clone()

        # Simulate an optimizer step moving the live param.
        nn.init.normal_(model.weight, mean=5.0, std=2.0)
        reference = (1.0 - decay) * model.weight.data + decay * original_shadow

        ema.step()

        # torch's lerp_ kernel reorders the arithmetic (self + w*(end-self))
        # vs. the reference's (1-d)*p + d*s, so this is NOT bit-exact — but
        # the deviation is pure float32-ULP rounding on O(1)-O(5) magnitude
        # operands (empirically ~1e-7 absolute). Plain allclose's DEFAULT
        # rtol is relative to the OUTPUT, which fails near an output's
        # zero-crossing (rtol * ~0 ~= 0) even though the absolute error
        # there is the same tiny ULP-level rounding as everywhere else — so
        # pin with a fixed absolute tolerance sized to that rounding, not a
        # size that would tolerate any real algebraic change (a genuine
        # semantic bug would show O(0.1)+ deviation, ~1000x bigger).
        assert torch.allclose(ema.shadow["weight"], reference, atol=1e-5, rtol=0)
        # Confirm the assertion has teeth (not a degenerate always-true check).
        assert not torch.allclose(
            ema.shadow["weight"], reference + 1.0, atol=1e-5, rtol=0
        )

    def test_step_matches_reference_formula_dual_expert_prefixed(self):
        """Same numeric pin, but through the W3.T10 dict-bound (dual-expert
        prefixed-name) construction path — the trickiest path to keep
        correct while caching per-name param references."""
        torch.manual_seed(1)
        high = nn.Linear(4, 2, bias=False)
        low = nn.Linear(4, 2, bias=False)
        nn.init.normal_(high.weight)
        nn.init.normal_(low.weight)
        decay = 0.712
        ema = EMAHandler(
            {"high.weight": high.weight, "low.weight": low.weight}, decay=decay
        )
        orig_high = ema.shadow["high.weight"].clone()
        orig_low = ema.shadow["low.weight"].clone()

        nn.init.normal_(high.weight, mean=3.0, std=1.5)
        nn.init.normal_(low.weight, mean=-2.0, std=0.5)
        ref_high = (1.0 - decay) * high.weight.data + decay * orig_high
        ref_low = (1.0 - decay) * low.weight.data + decay * orig_low

        ema.step()

        assert torch.allclose(ema.shadow["high.weight"], ref_high, atol=1e-5, rtol=0)
        assert torch.allclose(ema.shadow["low.weight"], ref_low, atol=1e-5, rtol=0)

    def test_step_does_not_rewalk_named_parameters(self):
        """The whole-model walk only happens at construction (and inside
        load_state_dict); step() must use the cached mapping exclusively."""
        model = _make_model()
        ema = EMAHandler(model, decay=0.9)

        calls = {"n": 0}
        original = ema._named_parameters

        def counting():
            calls["n"] += 1
            return original()

        ema._named_parameters = counting

        for _ in range(50):
            ema.step()

        assert calls["n"] == 0

    def test_load_state_dict_refreshes_cache_after_rewalk(self):
        """load_state_dict's own re-walk must refresh the cached mapping —
        step() afterwards must still update the (possibly-rebound) params."""
        model = _make_model()
        ema = EMAHandler(model, decay=0.5)
        ema.load_state_dict({"weight": torch.full((2, 4), 3.0)})

        model.weight.data.fill_(0.0)
        ema.step()
        # shadow = 0.5*0.0 + 0.5*3.0 = 1.5 — proves the post-load_state_dict
        # cache still points at the live "weight" param, not a stale/empty one.
        assert torch.allclose(
            ema.shadow["weight"], torch.full_like(model.weight.data, 1.5)
        )

    def test_step_telemetry_suppressed_below_debug_level(self, monkeypatch):
        model = _make_model()
        ema = EMAHandler(model, decay=0.9)

        rec = _LevelGatedLogger(debug_enabled=False)
        monkeypatch.setattr(ema_module, "logger", rec)

        for _ in range(200):
            ema.step()

        assert rec.debug_calls == []

    def test_step_telemetry_emitted_when_debug_enabled(self, monkeypatch):
        model = _make_model()
        ema = EMAHandler(model, decay=0.9)

        rec = _LevelGatedLogger(debug_enabled=True)
        monkeypatch.setattr(ema_module, "logger", rec)

        for _ in range(100):
            ema.step()

        assert len(rec.debug_calls) == 1
        event, kw = rec.debug_calls[0]
        assert event == "ema_step"
        assert kw["step_count"] == 100


class TestStoreAndSwap:
    def test_swap_loads_shadow_into_model(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.5)
        model.weight.data.fill_(0.0)
        ema.step()  # shadow now 0.5

        ema.store_and_swap()
        # Model should now have shadow weights (0.5)
        assert torch.allclose(model.weight.data, ema.shadow["weight"])

    def test_backup_is_created(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.99)
        ema.store_and_swap()
        assert len(ema.backup) > 0

    def test_backup_skips_frozen_params(self):
        # Regression: store_and_swap used to clone every parameter,
        # including the frozen base model. On 20B-class transformers
        # that ~40 GB transient allocation pushes the sampling peak
        # past consumer-card VRAM ceilings and spills into WDDM shared
        # memory. Only the params we'll actually overwrite (shadow,
        # i.e. trainable LoRA weights) need backing up.
        model = nn.Sequential(
            nn.Linear(64, 64, bias=False),  # frozen "base"
            nn.Linear(64, 4, bias=False),   # trainable "adapter"
        )
        model[0].weight.requires_grad = False
        model[1].weight.requires_grad = True
        ema = EMAHandler(model, decay=0.99)

        assert len(ema.shadow) == 1
        ema.store_and_swap()
        assert set(ema.backup.keys()) == set(ema.shadow.keys()), (
            "backup must mirror shadow keys (trainable only), not the full model"
        )
        backup_numel = sum(t.numel() for t in ema.backup.values())
        frozen_numel = model[0].weight.numel()
        assert backup_numel < frozen_numel, (
            f"backup ({backup_numel} elts) must be smaller than the "
            f"frozen base ({frozen_numel} elts); it is cloning frozen weights"
        )

    def test_restore_after_trainable_only_backup(self):
        # Round-trip: after store_and_swap + restore, the trainable
        # param must match its original value. Validates that restore
        # correctly matches by name once backup is dict-shaped.
        model = nn.Sequential(
            nn.Linear(8, 8, bias=False),
            nn.Linear(8, 2, bias=False),
        )
        model[0].weight.requires_grad = False
        model[1].weight.requires_grad = True
        nn.init.ones_(model[1].weight)
        original = model[1].weight.data.clone()

        ema = EMAHandler(model, decay=0.99)
        # Mutate shadow so swap will visibly change the model
        ema.shadow["1.weight"] = torch.zeros_like(ema.shadow["1.weight"])

        ema.store_and_swap()
        assert torch.allclose(model[1].weight.data, torch.zeros_like(original))
        ema.restore()
        assert torch.allclose(model[1].weight.data, original)


class TestRestore:
    def test_restore_reverts_model(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.5)
        model.weight.data.fill_(0.0)
        ema.step()

        ema.store_and_swap()  # swap to shadow
        ema.restore()  # revert
        # Model weights should be zero (the state before swap)
        assert torch.allclose(model.weight.data, torch.zeros_like(model.weight.data))

    def test_restore_clears_backup(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.99)
        ema.store_and_swap()
        ema.restore()
        assert ema.backup == {}

    def test_restore_noop_without_backup(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.99)
        ema.restore()  # Should not raise


class TestStateDictRoundTrip:
    def test_state_dict_returns_shadow(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.99)
        sd = ema.state_dict()
        assert "weight" in sd

    def test_load_state_dict(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.99)
        new_shadow = {"weight": torch.zeros(2, 4)}
        ema.load_state_dict(new_shadow)
        assert torch.allclose(ema.shadow["weight"], torch.zeros(2, 4))

    def test_load_state_dict_rebinds_to_param_device(self):
        # Regression: checkpoints are saved with map_location="cpu", so on
        # resume the loaded shadow tensors must be moved to each parameter's
        # current device. Without this, EMA.step() mixes CPU shadow with
        # CUDA params and raises a device-mismatch RuntimeError.
        if not torch.cuda.is_available():
            import pytest
            pytest.skip("CUDA not available")
        model = _make_model().to("cuda")
        ema = EMAHandler(model, decay=0.99)
        cpu_shadow = {"weight": torch.zeros(2, 4, device="cpu")}
        ema.load_state_dict(cpu_shadow)
        assert ema.shadow["weight"].device.type == "cuda"
        # Real-world reproduction: step() must not raise device mismatch
        ema.step()


class TestDictParamsConstruction:
    """EMAHandler bound to an explicit ``{name: Parameter}`` mapping (W3.T10 —
    the dual-expert seam) instead of an ``nn.Module``. This is what
    ``Wan22Trainer``/``BerniniRTrainer``'s ``_ema_parameters()`` hands
    ``_configure_ema`` on a ``both``-mode run: the union of BOTH experts'
    trainable params, name-prefixed (``high.``/``low.``) so keys stay unique.
    Every method (init/step/swap/restore/state_dict round-trip) must behave
    identically to the ``nn.Module`` path, just sourced from the dict instead
    of re-querying a single model.
    """

    @staticmethod
    def _prefixed_params():
        """A dual-expert-shaped params dict: two DISTINCT Linear layers whose
        param names are IDENTICAL ("weight") except for the high./low. prefix
        — exactly the collision the prefixing exists to avoid."""
        high = nn.Linear(4, 2, bias=False)
        low = nn.Linear(4, 2, bias=False)
        nn.init.ones_(high.weight)
        nn.init.constant_(low.weight, 2.0)
        return (
            {
                "high.weight": high.weight,
                "low.weight": low.weight,
            },
            high,
            low,
        )

    def test_shadow_clones_both_prefixed_params(self):
        params, high, low = self._prefixed_params()
        ema = EMAHandler(params, decay=0.99)
        assert set(ema.shadow) == {"high.weight", "low.weight"}
        assert torch.allclose(ema.shadow["high.weight"], high.weight.data)
        assert torch.allclose(ema.shadow["low.weight"], low.weight.data)
        assert ema.model is None

    def test_step_updates_both_prefixed_shadows_independently(self):
        params, high, low = self._prefixed_params()
        ema = EMAHandler(params, decay=0.5)
        high.weight.data.fill_(0.0)
        low.weight.data.fill_(0.0)
        ema.step()
        # high shadow: 0.5*0 + 0.5*1 = 0.5 ; low shadow: 0.5*0 + 0.5*2 = 1.0
        assert torch.allclose(
            ema.shadow["high.weight"], torch.full_like(high.weight, 0.5)
        )
        assert torch.allclose(
            ema.shadow["low.weight"], torch.full_like(low.weight, 1.0)
        )

    def test_store_and_swap_and_restore_round_trip(self):
        params, high, low = self._prefixed_params()
        ema = EMAHandler(params, decay=0.5)
        high.weight.data.fill_(0.0)
        low.weight.data.fill_(0.0)
        ema.step()  # shadows now 0.5 / 1.0

        ema.store_and_swap()
        assert torch.allclose(high.weight.data, ema.shadow["high.weight"])
        assert torch.allclose(low.weight.data, ema.shadow["low.weight"])

        ema.restore()
        assert torch.allclose(high.weight.data, torch.zeros_like(high.weight))
        assert torch.allclose(low.weight.data, torch.zeros_like(low.weight))

    def test_zero_overlap_keeps_fresh_shadow_and_warns(self, monkeypatch):
        """Wave-3 regression: resuming a dual-expert (``both``) run whose
        live shadow keys are ``high.``/``low.``-prefixed (T10) from a
        checkpoint saved BEFORE that prefixing existed (plain un-prefixed
        keys) must NOT silently adopt the stale, zero-overlap dict — that
        would make step()/store_and_swap()/restore() no-op forever and both
        experts' saved LoRA would silently revert to raw weights (the exact
        defect T10 fixed, now doubled and silent)."""
        params, high, low = self._prefixed_params()
        ema = EMAHandler(params, decay=0.9)
        fresh_high = ema.shadow["high.weight"].clone()
        fresh_low = ema.shadow["low.weight"].clone()

        rec = _RecordingLogger()
        monkeypatch.setattr(ema_module, "logger", rec)

        stale_unprefixed = {
            "weight": torch.full((2, 4), 99.0),  # pre-wave, un-prefixed key
        }
        ema.load_state_dict(stale_unprefixed)

        assert torch.allclose(ema.shadow["high.weight"], fresh_high), (
            "zero-overlap checkpoint must NOT clobber the freshly-initialized shadow"
        )
        assert torch.allclose(ema.shadow["low.weight"], fresh_low)
        assert set(ema.shadow) == {"high.weight", "low.weight"}, (
            "the stale foreign key must not be adopted into the shadow at all"
        )

        events = [ev for ev, _ in rec.warnings]
        assert "ema_shadow_key_mismatch" in events
        kw = dict(rec.warnings)["ema_shadow_key_mismatch"]
        assert kw["reason"] == "no_overlap"
        assert kw["overlap"] == 0
        assert kw["live_params"] == 2
        assert kw["loaded_keys"] == 1

    def test_partial_overlap_adopts_covered_subset_and_warns(self, monkeypatch):
        """``expert_mode`` flipped ``high`` -> ``both`` between save and
        resume: the checkpoint only ever shadowed the high expert. The
        overlapping ``high.*`` history is real and worth keeping; the new
        ``low.*`` param (never in the checkpoint) must keep its own
        freshly-initialized shadow rather than vanish — and the partial
        coverage must be logged, not silent."""
        params, high, low = self._prefixed_params()
        ema = EMAHandler(params, decay=0.9)
        fresh_low = ema.shadow["low.weight"].clone()

        rec = _RecordingLogger()
        monkeypatch.setattr(ema_module, "logger", rec)

        high_only_checkpoint = {"high.weight": torch.full((2, 4), 42.0)}
        ema.load_state_dict(high_only_checkpoint)

        assert torch.allclose(ema.shadow["high.weight"], torch.full((2, 4), 42.0)), (
            "the covered high.* history must be adopted"
        )
        assert torch.allclose(ema.shadow["low.weight"], fresh_low), (
            "the uncovered low.* param must keep its freshly-initialized shadow"
        )

        events = [ev for ev, _ in rec.warnings]
        assert "ema_shadow_key_mismatch" in events
        kw = dict(rec.warnings)["ema_shadow_key_mismatch"]
        assert kw["reason"] == "partial_overlap"
        assert kw["overlap"] == 1
        assert kw["live_params"] == 2
        assert "low.weight" in kw["sample_missing_live"]

    def test_full_live_coverage_drops_dead_keys_silently(self, monkeypatch):
        """``expert_mode`` flipped ``both`` -> ``high``: the checkpoint still
        carries ``low.*`` keys for an expert that no longer exists in this
        run's live params. Every live param IS covered, so the dead
        ``low.*`` entry is just dropped — no warning (nothing was left
        uncovered)."""
        high = nn.Linear(4, 2, bias=False)
        nn.init.ones_(high.weight)
        ema = EMAHandler({"high.weight": high.weight}, decay=0.9)

        rec = _RecordingLogger()
        monkeypatch.setattr(ema_module, "logger", rec)

        both_checkpoint = {
            "high.weight": torch.full((2, 4), 7.0),
            "low.weight": torch.full((2, 4), 13.0),  # dead — no live low expert
        }
        ema.load_state_dict(both_checkpoint)

        assert torch.allclose(ema.shadow["high.weight"], torch.full((2, 4), 7.0))
        assert set(ema.shadow) == {"high.weight"}, (
            "the dead low.* key must not be adopted into the shadow"
        )
        assert rec.warnings == [], (
            "full live-parameter coverage must not warn, even with dropped dead keys"
        )

    def test_state_dict_round_trips_through_checkpoint_save_restore(self):
        """The EMA shadow must round-trip through checkpoint save/restore
        with the prefixed names — CheckpointManager just ``torch.save``s
        ``ema_handler.state_dict()`` and later ``ema_handler.load_state_dict``s
        it back (``app/engine/components/checkpoints.py``); this is
        source-agnostic (model vs. dict-bound), so a save/load cycle must
        preserve every prefixed key byte-for-byte."""
        params, high, low = self._prefixed_params()
        ema = EMAHandler(params, decay=0.9)
        high.weight.data.fill_(3.0)
        low.weight.data.fill_(4.0)
        ema.step()
        saved = {k: v.clone() for k, v in ema.state_dict().items()}

        # A FRESH handler (as on resume — new process, new Parameter objects,
        # SAME prefixed names) loads the saved shadow back.
        new_high = nn.Linear(4, 2, bias=False)
        new_low = nn.Linear(4, 2, bias=False)
        restored = EMAHandler(
            {"high.weight": new_high.weight, "low.weight": new_low.weight},
            decay=0.9,
        )
        restored.load_state_dict(saved)

        assert set(restored.shadow) == {"high.weight", "low.weight"}
        assert torch.allclose(restored.shadow["high.weight"], saved["high.weight"])
        assert torch.allclose(restored.shadow["low.weight"], saved["low.weight"])
        # Round-tripped shadow must still be independently steppable/swappable.
        restored.store_and_swap()
        assert torch.allclose(new_high.weight.data, saved["high.weight"])
        assert torch.allclose(new_low.weight.data, saved["low.weight"])
