"""Behavioural pins for the clean-room boogu_image Taylor feature cache.

Every expected value here is **hand-computed** from
``_harness/research/taylorseer-cache-behavioural-spec.md`` and written out in
the test that uses it. Nothing is compared against a previous implementation.

Chosen so the arithmetic is exact in fp32: every feature value and every
divisor is a small dyadic rational, so ``torch.equal`` is the right assertion
and a tolerance would only hide a real error.

**What these tests do NOT cover:** generated-image quality. A caching defect's
failure mode is a subtly worse image, not an exception. These pin the maths and
the schedule; a GPU sampling comparison against a pre-change checkpoint belongs
in the UAT pack.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from app.engine.models.families.boogu_image.vendor.taylor_cache import (
    cache_init,
    cal_type,
    derivative_approximation,
    derivative_approximation_4_double_stream,
    force_scheduler,
    taylor_cache_init,
    taylor_formula,
    taylor_formula_4_double_stream,
)

STREAM = "single_stream_layers"
LAYER = 0
MODULE = "total"


def _init(num_steps: int = 16, **config: Any) -> tuple[dict, dict]:
    """A cache pair with no pipeline; layer slots are created on demand."""
    cache_dic, current = cache_init(None, num_steps, **config)
    current["stream"] = STREAM
    current["layer"] = LAYER
    current["module"] = MODULE
    return cache_dic, current


def _terms(cache_dic: dict, current: dict) -> dict[int, tuple[torch.Tensor, ...]]:
    """Read the term store through the documented cache address (spec 2.2).

    Terms are held as fixed-arity tuples so the single- and double-stream maths
    is written once; a single-tensor feature is a 1-tuple.
    """
    return cache_dic["cache"][-1][current["stream"]][current["layer"]][
        current["module"]
    ]


def _seed(cache_dic: dict, current: dict, orders: dict[int, torch.Tensor]) -> None:
    """Place known single-tensor terms directly, to test evaluation in isolation."""
    cache_dic["cache"][-1].setdefault(current["stream"], {}).setdefault(
        current["layer"], {}
    )[current["module"]] = {order: (value,) for order, value in orders.items()}


def _run(
    cache_dic: dict,
    current: dict,
    num_steps: int,
    on_full=None,
    on_taylor=None,
) -> list[str]:
    """Drive the per-step loop exactly as transformer_boogu.py does.

    ``cal_type`` once at the top of the step, the layer work in between, and the
    step counter incremented at the end.
    """
    seen: list[str] = []
    for _ in range(num_steps):
        cal_type(cache_dic, current)
        seen.append(current["type"])
        if current["type"] == "full":
            taylor_cache_init(cache_dic, current)
            if on_full is not None:
                on_full(current["step"])
        elif on_taylor is not None:
            on_taylor(current["step"])
        current["step"] += 1
    return seen


class TestSeriesEvaluation:
    """Spec 3.3 / test 1: output = sum over i of c[i] * x**i / i!."""

    # c = {0: t, 1: 2t, 2: 6t} at x = 2 gives
    #   t*(2**0/0!) + 2t*(2**1/1!) + 6t*(2**2/2!)
    # = t*1 + 2t*2 + 6t*2
    # = t + 4t + 12t
    # = 17t
    BASE = torch.tensor([[1.0, 2.0], [4.0, 0.5]])
    EXPECTED_FACTOR = 17.0

    def _evaluate(self) -> torch.Tensor:
        cache_dic, current = _init()
        _seed(
            cache_dic,
            current,
            {0: self.BASE, 1: 2 * self.BASE, 2: 6 * self.BASE},
        )
        # Offset x = 2 from the last full compute.
        current["activated_steps"] = [0, 4]
        current["step"] = 6
        return taylor_formula(cache_dic, current)

    def _assert_factor(self, factor: float) -> None:
        """The single assertion under test; test 10 proves it can fail."""
        assert torch.equal(self._evaluate(), factor * self.BASE)

    def test_series_matches_hand_computed_17t(self):
        self._assert_factor(self.EXPECTED_FACTOR)

    def test_wrong_expected_value_fails(self):
        # Spec test 10 -- prove the negative. If this passes, the assertion in
        # test_series_matches_hand_computed_17t proves nothing.
        with pytest.raises(AssertionError):
            self._assert_factor(16.0)

    def test_zero_offset_returns_the_stored_feature(self):
        # x = 0 collapses every term above order 0 (0**i == 0 for i >= 1), so a
        # full step's own value must come back untouched.
        cache_dic, current = _init()
        _seed(cache_dic, current, {0: self.BASE, 1: 2 * self.BASE})
        current["activated_steps"] = [0, 4]
        current["step"] = 4
        assert torch.equal(taylor_formula(cache_dic, current), self.BASE)

    def test_predicting_without_a_full_step_raises(self):
        # Failure is never silent: an empty store must not quietly predict zeros.
        cache_dic, current = _init()
        current["activated_steps"] = [0]
        current["step"] = 1
        with pytest.raises(RuntimeError, match="nothing cached"):
            taylor_formula(cache_dic, current)


class TestDividedDifferences:
    """Spec 3.2 / test 2: new[i+1] = (new[i] - previous[i]) / h."""

    # first_enhance=1, fresh_threshold=2 puts full steps at 0, 2, 4 -- so h = 2
    # every time, and every division below is exact in fp32.
    F0 = torch.tensor([1.0, 2.0])
    F2 = torch.tensor([5.0, 4.0])
    F4 = torch.tensor([9.0, 10.0])

    def _run_three_full_steps(self) -> tuple[dict, dict]:
        cache_dic, current = _init(first_enhance=1, fresh_threshold=2)
        features = {0: self.F0, 2: self.F2, 4: self.F4}
        _run(
            cache_dic,
            current,
            5,
            on_full=lambda step: derivative_approximation(
                cache_dic, current, features[step]
            ),
        )
        return cache_dic, current

    def test_full_steps_land_where_the_hand_computation_assumes(self):
        # The h = 2 spacing the expected values below depend on.
        cache_dic, current = _init(first_enhance=1, fresh_threshold=2)
        seen = _run(cache_dic, current, 5)
        assert seen == ["full", "Taylor", "full", "Taylor", "full"]
        assert current["activated_steps"] == [0, 2, 4]

    def test_first_full_step_stores_order_zero_only(self):
        cache_dic, current = _init(first_enhance=1, fresh_threshold=2)
        _run(
            cache_dic,
            current,
            1,
            on_full=lambda step: derivative_approximation(
                cache_dic, current, self.F0
            ),
        )
        # One observation supports order 0 alone.
        assert sorted(_terms(cache_dic, current)) == [0]
        assert torch.equal(_terms(cache_dic, current)[0][0], self.F0)

    def test_order_one_is_the_first_divided_difference(self):
        cache_dic, current = _init(first_enhance=1, fresh_threshold=2)
        features = {0: self.F0, 2: self.F2}
        _run(
            cache_dic,
            current,
            3,
            on_full=lambda step: derivative_approximation(
                cache_dic, current, features[step]
            ),
        )
        terms = _terms(cache_dic, current)
        assert sorted(terms) == [0, 1]
        # (F2 - F0)/2 = ([5,4] - [1,2])/2 = [4,2]/2 = [2,1]
        assert torch.equal(terms[1][0], torch.tensor([2.0, 1.0]))

    def test_order_two_is_the_second_divided_difference(self):
        cache_dic, current = self._run_three_full_steps()
        terms = _terms(cache_dic, current)
        assert sorted(terms) == [0, 1, 2]
        assert torch.equal(terms[0][0], self.F4)
        # order 1 = (F4 - F2)/2 = ([9,10] - [5,4])/2 = [4,6]/2 = [2,3]
        assert torch.equal(terms[1][0], torch.tensor([2.0, 3.0]))
        # order 2 = (new order 1 - previous order 1)/2
        #         = ([2,3] - [2,1])/2 = [0,2]/2 = [0,1]
        assert torch.equal(terms[2][0], torch.tensor([0.0, 1.0]))

    def test_warmup_steps_do_not_build_higher_orders(self):
        # Spec 3.2 "guard the early steps": differences taken across the
        # densely-computed warm-up region degrade the image without raising.
        cache_dic, current = _init(first_enhance=4, fresh_threshold=2)
        _run(
            cache_dic,
            current,
            4,
            on_full=lambda step: derivative_approximation(
                cache_dic, current, self.F0 * (step + 1)
            ),
        )
        # Four consecutive full warm-up steps, still order 0 only.
        assert sorted(_terms(cache_dic, current)) == [0]


class TestOrderCeiling:
    """Spec test 3: with max_order = 2, no order-3 term is ever stored."""

    def test_no_order_above_max_order_however_many_full_steps_run(self):
        cache_dic, current = _init(
            num_steps=32, first_enhance=1, fresh_threshold=2, max_order=2
        )
        feature = torch.tensor([1.0, 3.0])
        seen_orders: set[int] = set()

        def record(step: int) -> None:
            derivative_approximation(cache_dic, current, feature * (step + 1))
            seen_orders.update(_terms(cache_dic, current))

        _run(cache_dic, current, 20, on_full=record)

        assert seen_orders == {0, 1, 2}
        assert max(_terms(cache_dic, current)) == 2

    def test_max_order_zero_keeps_only_the_reused_value(self):
        cache_dic, current = _init(first_enhance=1, fresh_threshold=2, max_order=0)
        feature = torch.tensor([2.0, 6.0])
        _run(
            cache_dic,
            current,
            9,
            on_full=lambda step: derivative_approximation(
                cache_dic, current, feature
            ),
        )
        assert sorted(_terms(cache_dic, current)) == [0]


class TestStepClassification:
    """Spec 3.1 / tests 4, 5, 6 -- the shipped configuration's schedule."""

    def test_every_warmup_step_is_full(self):
        # Spec test 4: while step < first_enhance the step is always full.
        cache_dic, current = _init()
        seen = _run(cache_dic, current, 5)
        assert seen == ["full"] * 5

    def test_steady_state_repeats_full_taylor_taylor(self):
        # Spec test 5. Shipped configuration: first_enhance=5, fresh_threshold=3.
        # Steps 0-4 are warm-up (full). Step 4 leaves cache_counter at 0, so
        # steps 5 and 6 are Taylor (counter 1, then 2) and step 7 is full
        # (counter has reached cal_threshold - 1 = 2). The cycle then repeats
        # with period 3, anchored on the last full step.
        cache_dic, current = _init()
        seen = _run(cache_dic, current, 13)
        assert seen == [
            "full", "full", "full", "full", "full",   # steps 0-4, warm-up
            "Taylor", "Taylor", "full",               # steps 5-7
            "Taylor", "Taylor", "full",               # steps 8-10
            "Taylor", "Taylor",                       # steps 11-12
        ]

    def test_activated_steps_ascending_one_entry_per_full_step(self):
        # Spec test 6.
        cache_dic, current = _init()
        seen = _run(cache_dic, current, 13)
        activated = current["activated_steps"]

        assert activated == [0, 1, 2, 3, 4, 7, 10]
        assert all(b > a for a, b in zip(activated, activated[1:]))
        assert len(activated) == seen.count("full")

    def test_step_zero_is_not_duplicated_in_activated_steps(self):
        # activated_steps is seeded with [0] at init because step 0 is always
        # full; the first full step must not append it a second time.
        cache_dic, current = _init()
        assert current["activated_steps"] == [0]
        _run(cache_dic, current, 1)
        assert current["activated_steps"] == [0]

    def test_cache_counter_resets_on_full_and_increments_on_taylor(self):
        cache_dic, current = _init()
        _run(cache_dic, current, 5)
        assert cache_dic["cache_counter"] == 0  # step 4 was full
        _run(cache_dic, current, 1)
        assert cache_dic["cache_counter"] == 1  # step 5 was Taylor
        _run(cache_dic, current, 1)
        assert cache_dic["cache_counter"] == 2  # step 6 was Taylor
        _run(cache_dic, current, 1)
        assert cache_dic["cache_counter"] == 0  # step 7 was full again

    def test_non_taylor_modes_fail_loudly_rather_than_guessing(self):
        # "ToCa"/"Delta-Cache" are unreachable in every shipped configuration
        # and were never exercised; mislabelling a cached step would silently
        # reuse a stale feature.
        cache_dic, current = _init(taylor_cache=False, first_enhance=1)
        _run(cache_dic, current, 1)  # step 0 is warm-up, still full
        with pytest.raises(NotImplementedError, match="only the Taylor path"):
            cal_type(cache_dic, current)


class TestDoubleStreamParity:
    """Spec test 7: a tuple in, a tuple of the same arity out, element-wise."""

    def test_each_element_matches_the_single_tensor_result(self):
        img = torch.tensor([[1.0, 2.0], [4.0, 0.5]])
        instruct = torch.tensor([[8.0, 1.0], [2.0, 16.0]])

        cache_dic, current = _init(first_enhance=1, fresh_threshold=2)

        def on_full(step: int) -> None:
            scale = float(step + 1)
            # Same address for the pair, two neighbouring addresses for the
            # single-tensor references, so all three share one schedule.
            current["layer"] = 0
            derivative_approximation_4_double_stream(
                cache_dic, current, (img * scale, instruct * scale)
            )
            current["layer"] = 1
            derivative_approximation(cache_dic, current, img * scale)
            current["layer"] = 2
            derivative_approximation(cache_dic, current, instruct * scale)

        def on_taylor(step: int) -> None:
            current["layer"] = 0
            pair = taylor_formula_4_double_stream(cache_dic, current)
            current["layer"] = 1
            first = taylor_formula(cache_dic, current)
            current["layer"] = 2
            second = taylor_formula(cache_dic, current)

            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert torch.equal(pair[0], first)
            assert torch.equal(pair[1], second)

        _run(cache_dic, current, 9, on_full=on_full, on_taylor=on_taylor)

    def test_arity_change_mid_run_is_rejected(self):
        # Failure is never silent: a pair becoming a triple would otherwise
        # zip short and drop a stream's history.
        cache_dic, current = _init(first_enhance=1, fresh_threshold=2)
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        _run(cache_dic, current, 1, on_full=lambda _: (
            derivative_approximation_4_double_stream(cache_dic, current, (a, b))
        ))
        cal_type(cache_dic, current)  # step 1: Taylor
        current["step"] += 1
        cal_type(cache_dic, current)  # step 2: full
        with pytest.raises(ValueError, match="arity changed"):
            derivative_approximation_4_double_stream(cache_dic, current, (a, b, a))


class TestScheduler:
    """Spec 3.4 / test 8."""

    def test_cal_threshold_equals_fresh_threshold_and_is_an_int(self):
        cache_dic, _ = _init()
        assert cache_dic["cal_threshold"] == cache_dic["fresh_threshold"]
        assert type(cache_dic["cal_threshold"]) is int
        assert not isinstance(cache_dic["cal_threshold"], torch.Tensor)

    def test_it_stays_an_int_after_every_full_step(self):
        cache_dic, current = _init()
        _run(cache_dic, current, 13)
        assert type(cache_dic["cal_threshold"]) is int
        assert cache_dic["cal_threshold"] == cache_dic["fresh_threshold"]

    def test_force_scheduler_is_exact_equality_for_any_configured_interval(self):
        for interval in (1, 2, 3, 7):
            cache_dic, current = _init(fresh_threshold=interval)
            force_scheduler(cache_dic, current)
            assert cache_dic["cal_threshold"] == interval
            assert type(cache_dic["cal_threshold"]) is int


class TestConfigValidation:
    """Spec 4.1 / test 9: bad configuration fails at init, naming the key."""

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("max_order", -1),
            ("fresh_threshold", 0),
            ("fresh_threshold", -3),
            ("first_enhance", -1),
        ],
    )
    def test_invalid_config_value_raises_naming_the_key(self, key, value):
        with pytest.raises(ValueError, match=key):
            cache_init(None, 16, **{key: value})

    @pytest.mark.parametrize("num_steps", [0, -1])
    def test_invalid_num_steps_raises(self, num_steps):
        with pytest.raises(ValueError, match="num_steps"):
            cache_init(None, num_steps)

    def test_shipped_defaults_are_accepted(self):
        cache_dic, current = cache_init(None, 20)
        assert cache_dic["taylor_cache"] is True
        assert cache_dic["max_order"] == 4
        assert cache_dic["first_enhance"] == 5
        assert cache_dic["fresh_threshold"] == 3
        assert cache_dic["cache_counter"] == 0
        assert current["step"] == 0
        assert current["num_steps"] == 20
        assert current["activated_steps"] == [0]

    def test_non_taylor_config_keys_are_carried(self):
        # Kept so the alternative modes stay selectable; see cal_type for why
        # no behaviour is invented for them.
        cache_dic, _ = cache_init(None, 20)
        assert cache_dic["fresh_ratio"] == 0.0
        assert cache_dic["fresh_ratio_schedule"] == "ToCa"
        assert cache_dic["soft_fresh_weight"] == 0.0
        assert cache_dic["cache_type"] == "random"
        assert cache_dic["Delta-DiT"] is False

    def test_cache_and_index_roots_exist(self):
        cache_dic, _ = cache_init(None, 20)
        assert cache_dic["cache"][-1] == {}
        assert cache_dic["cache_index"][-1] == {}

    def test_layer_slots_are_precreated_when_the_pipeline_exposes_a_count(self):
        class _Transformer:
            num_layers = 3

        class _Pipe:
            transformer = _Transformer()

        cache_dic, _ = cache_init(_Pipe(), 20)
        assert sorted(cache_dic["cache"][-1]["layers_stream"]) == [0, 1, 2]
        assert sorted(cache_dic["cache_index"][-1]) == [0, 1, 2]


class TestEndToEndSchedule:
    """The prediction actually used: a linear feature is reproduced exactly."""

    def test_linear_feature_is_predicted_exactly_once_order_one_exists(self):
        # A feature that is exactly linear in the step index is, after order 1
        # is available, reproduced with zero error -- the series is exact for
        # any polynomial of degree <= the retained order. Hand-checkable and
        # independent of the divided-difference bookkeeping.
        base = torch.tensor([1.0, 4.0])
        cache_dic, current = _init(first_enhance=1, fresh_threshold=2)

        errors: list[float] = []

        def on_full(step: int) -> None:
            derivative_approximation(cache_dic, current, base * float(step))

        def on_taylor(step: int) -> None:
            # Full steps at 0, 2, 4, ... so order 1 exists from step 3 onward.
            if step >= 3:
                predicted = taylor_formula(cache_dic, current)
                errors.append(float((predicted - base * float(step)).abs().max()))

        _run(cache_dic, current, 11, on_full=on_full, on_taylor=on_taylor)

        assert errors, "no cached step was evaluated"
        assert max(errors) == 0.0

    def test_quadratic_feature_is_predicted_with_the_expected_lag(self):
        # The ladder estimates derivatives by BACKWARD differences, so the
        # order-1 term at a full step approximates the slope half a spacing
        # earlier. The series is therefore exact for degree <= 1 (the test
        # above) and only first-order accurate beyond it. That approximation is
        # the acceleration, not a defect -- so what is worth pinning is its
        # exact arithmetic, which a silent change to the ladder or to the
        # factorial weighting would move.
        base = torch.tensor([1.0, 2.0])
        cache_dic, current = _init(first_enhance=1, fresh_threshold=2)

        def value(step: int) -> torch.Tensor:
            return base * float(step * step)

        predictions: dict[int, torch.Tensor] = {}

        _run(
            cache_dic,
            current,
            6,
            on_full=lambda step: derivative_approximation(
                cache_dic, current, value(step)
            ),
            on_taylor=lambda step: predictions.__setitem__(
                step, taylor_formula(cache_dic, current)
            ),
        )

        # Full steps at 0, 2 and 4 with h = 2. After the full step at 4 the
        # stored terms are
        #   order 0 = f(4)             = 16b
        #   order 1 = (16b - 4b) / 2   =  6b
        #   order 2 = ( 6b - 2b) / 2   =  2b
        # and the cached step at 5 sits at offset 1 from the last full compute:
        #   16b + 6b * (1/1!) + 2b * (1/2!) = 16b + 6b + 1b = 23b
        assert torch.equal(predictions[5], base * 23.0)
        # The true value there is 25b. Asserting the difference keeps this test
        # honest: it pins the method's own arithmetic, not a restatement of f.
        assert not torch.equal(predictions[5], value(5))


class TestExportSurface:
    """The caller imports exactly these names; dropping one breaks sampling."""

    SCHEDULING = {"cache_init", "cal_type", "force_scheduler"}
    SERIES = {
        "derivative_approximation",
        "derivative_approximation_4_double_stream",
        "taylor_cache_init",
        "taylor_formula",
        "taylor_formula_4_double_stream",
    }

    def test_all_public_names_are_exported(self):
        from app.engine.models.families.boogu_image.vendor import taylor_cache

        expected = self.SCHEDULING | self.SERIES
        assert set(taylor_cache.__all__) == expected
        for name in expected:
            assert callable(getattr(taylor_cache, name))

    def test_the_two_package_paths_the_caller_imports_still_serve_them(self):
        # transformer_boogu.py does `from ...cache_functions import cal_type`
        # and `from ...taylorseer_utils import (...)`. Those paths are the
        # public surface: the shipped sampler breaks on a missing name, and it
        # breaks at sampling time, not at import time, for the four names the
        # older vendor import test does not cover.
        from app.engine.models.families.boogu_image.vendor import (
            cache_functions,
            taylorseer_utils,
        )

        assert set(cache_functions.__all__) == self.SCHEDULING
        assert set(taylorseer_utils.__all__) == self.SERIES
        for module, names in (
            (cache_functions, self.SCHEDULING),
            (taylorseer_utils, self.SERIES),
        ):
            for name in names:
                assert callable(getattr(module, name))

    def test_the_packages_re_export_the_same_objects(self):
        # Not copies: one implementation, two documented import paths.
        from app.engine.models.families.boogu_image.vendor import (
            cache_functions,
            taylor_cache,
            taylorseer_utils,
        )

        for module, names in (
            (cache_functions, self.SCHEDULING),
            (taylorseer_utils, self.SERIES),
        ):
            for name in names:
                assert getattr(module, name) is getattr(taylor_cache, name)

    def test_factorial_weighting_is_present_in_the_series(self):
        # Guards the one place a plain polynomial and a Taylor series differ.
        base = torch.tensor([1.0])
        cache_dic, current = _init()
        _seed(cache_dic, current, {0: base * 0.0, 2: base})
        current["activated_steps"] = [0, 4]
        current["step"] = 6  # offset 2
        # order-2 term only: 1 * 2**2 / 2! = 2, not 4.
        assert torch.equal(
            taylor_formula(cache_dic, current), base * (2**2 / math.factorial(2))
        )
