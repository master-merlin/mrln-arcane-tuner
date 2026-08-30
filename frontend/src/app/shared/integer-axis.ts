import uPlot from 'uplot';

/**
 * Tick constraints for a uPlot axis whose domain is **integral by
 * construction** — training step numbers, layer indices, counts of things.
 *
 * The defect this exists to prevent (UAT-3): the Training Curves x-axis
 * declared only stroke/grid/ticks/font, so uPlot fell back to its default
 * numeric increments, `[1,2,5] x 10^n` over the whole exponent range. Those
 * include 0.5, 0.25, 0.2, 0.1 ... and uPlot picks the SMALLEST increment whose
 * tick spacing still clears `axis.space` (50px by default). Once the visible
 * span is small enough that even an increment of 1 leaves more than ~2x that
 * spacing, a sub-integer increment wins and the axis grows gridlines at
 * 5974.5 — labelling a step the run never took.
 *
 * **The constraint has to live in `incrs`, not in `values`.** uPlot's default
 * split generator emits multiples of the chosen increment, so an integer-only
 * increment list makes a fractional tick unrepresentable. Rounding in a
 * `values` formatter instead leaves the tick at 5974.5 and only relabels it,
 * producing two adjacent gridlines that both read `5974` — a duplicate-label
 * fault that reads as a rendering bug and is harder to diagnose than the
 * fraction it replaced. `integerTickValues` therefore only *agrees* with
 * `integerTickIncrs`; on its own it would be the bug.
 */

/**
 * Ascending integer increments, uPlot's own `[1, 2, 5] x 10^n` ladder with the
 * quarter stops (25 / 250 / 2500) kept, truncated at the bottom to 1. The top
 * (5e6) is above any plausible step count; uPlot clamps to the largest entry
 * rather than failing, so overshooting costs nothing.
 */
const INCRS: readonly number[] = [
    1, 2, 5,
    10, 25, 50,
    100, 250, 500,
    1_000, 2_500, 5_000,
    10_000, 25_000, 50_000,
    100_000, 250_000, 500_000,
    1_000_000, 2_500_000, 5_000_000,
];

/**
 * Integer tick increments for an integral axis. Returns a fresh array per call
 * so no two charts can share (or mutate) one options object.
 */
export function integerTickIncrs(): number[] {
    return [...INCRS];
}

/**
 * Integer tick labels, formatted exactly the way uPlot formats numbers by
 * default (`uPlot.fmtNum` — locale grouping), so constraining the increments
 * does not also change how the surviving labels look.
 */
export function integerTickValues(_u: uPlot, splits: number[]): string[] {
    return splits.map(v => (v == null || !Number.isFinite(v) ? '' : uPlot.fmtNum(Math.round(v))));
}

/**
 * A uPlot axis config for an integral domain: the increment constraint and the
 * matching formatter, with `extra` (colors, size, side, scale …) merged on top.
 *
 * `extra` may override `values` — a caller that needs its own label text (a
 * name looked up by index, say) still keeps the increment constraint, which is
 * the part that actually fixes the defect. `incrs` is applied AFTER the spread
 * and so cannot be overridden: an axis built here can never emit a fractional
 * tick, whatever the caller passes.
 */
export function integerAxis(extra: Partial<uPlot.Axis> = {}): uPlot.Axis {
    return {
        values: integerTickValues,
        ...extra,
        incrs: integerTickIncrs(),
    };
}
