"""Every uPlot axis over an integral domain is wired to the shared constraint.

RULE-20 class S guard for the UAT-3 finding "the Training Curves x-axis shows
half-steps (5974.5) when the data only ever exists at whole steps".

The *property* — an integer-only increment ladder makes a fractional tick
unrepresentable — is pinned in the JS suite
(``frontend/src/app/shared/integer-axis.spec.ts``, which also carries the
positive control and the browser-measured increment table). What that spec
cannot see is whether the charts actually USE it: a served constraint is not a
wired constraint. This file pins the wiring.

Every assertion here states a POSITIVE fact (this file calls ``integerAxis`` for
this axis), so it fails loudly the moment a constraint is deleted or a file is
moved — no empty-offender-list vacuity to control for (CONVENTIONS "Tests" #11).
It lives in the Python suite for the same reason as
``test_frontend_release_hygiene.py``: it needs filesystem access, and the
frontend tsconfig has no ``@types/node``.

Scope, deliberately: this guard covers the four charts that existed when the
defect was found. It cannot know about a chart added later — the shared module's
docstring is what tells the next author which axes belong on the ladder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "frontend" / "src" / "app"

pytestmark = pytest.mark.skipif(not SRC.is_dir(), reason="frontend sources not present")

SHARED_MODULE = SRC / "shared" / "integer-axis.ts"

# (path, human name of the axis, the source line that must call integerAxis)
INTEGRAL_AXES = [
    (
        "components/training/training-chart/training-chart.ts",
        "Training Curves x = training step",
        "integerAxis({",
    ),
    (
        "modals/training-stats/stats-charts.ts",
        "Training Statistics adaptive/activity/histogram integral axes",
        "integerAxis(",
    ),
    (
        "components/tools/lora-tools/lora-tools.ts",
        "LoRA Tools x = layer index",
        "integerAxis({",
    ),
]


def test_shared_integer_axis_module_exists() -> None:
    """The single place the increment ladder is defined."""
    assert SHARED_MODULE.is_file(), f"missing {SHARED_MODULE}"
    text = SHARED_MODULE.read_text(encoding="utf-8")
    assert "export function integerAxis(" in text
    assert "export function integerTickIncrs(" in text


@pytest.mark.parametrize(("rel", "axis", "needle"), INTEGRAL_AXES)
def test_integral_axis_is_constrained(rel: str, axis: str, needle: str) -> None:
    """A chart whose domain is whole numbers must build its axis via the ladder."""
    path = SRC / rel
    assert path.is_file(), f"{axis}: {path} is gone — re-point this guard"
    text = path.read_text(encoding="utf-8")
    assert "integer-axis" in text, f"{axis}: {rel} no longer imports the shared constraint"
    assert needle in text, f"{axis}: {rel} no longer calls {needle} — half-steps are back"


def test_stats_adaptive_chart_constrains_both_of_its_axes() -> None:
    """Step on x, active-layer count on y — both integral, both on the ladder."""
    text = (SRC / "modals" / "training-stats" / "stats-charts.ts").read_text(encoding="utf-8")
    body = text.split("export function buildAdaptiveOpts")[1]
    assert body.count("integerAxis(") == 2, "buildAdaptiveOpts must constrain x AND y"


def test_loss_histogram_x_axis_is_left_continuous() -> None:
    """Prove the negative: not every x-axis is integral.

    The loss histogram's x is bin CENTERS (0.0315, 0.0625 ...). Putting it on the
    integer ladder would collapse the whole axis onto one tick. If a later edit
    "fixes" it for consistency, this fails and says why.
    """
    text = (SRC / "modals" / "training-stats" / "stats-charts.ts").read_text(encoding="utf-8")
    body = text.split("export function buildHistogramOpts")[1].split("export function")[0]
    x_axis_line = next(line for line in body.splitlines() if "toFixed(3)" in line)
    assert "integerAxis" not in x_axis_line, (
        "the loss histogram's x-axis is a continuous domain; it must NOT be "
        "constrained to integer increments"
    )
