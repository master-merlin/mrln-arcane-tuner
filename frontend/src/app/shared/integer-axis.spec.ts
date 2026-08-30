import { describe, it, expect } from 'vitest';
import uPlot from 'uplot';

import { integerAxis, integerTickIncrs, integerTickValues } from './integer-axis';

/**
 * UAT-3 guard: an axis over an integral domain (training steps, layer indices,
 * counts) must not be able to put a gridline at 5974.5.
 *
 * jsdom has no canvas 2D context, so a real uPlot cannot be mounted here. The
 * two behaviours that decide whether a fractional tick appears are pure
 * arithmetic and are modelled below:
 *
 *   `chooseIncr`  — uPlot picks the SMALLEST candidate increment whose on-screen
 *                   spacing still clears `axis.space` (50px default for x).
 *   `splitsFor`   — uPlot's split generator emits multiples of that increment.
 *
 * Both models are pinned against MEASURED browser output (Chrome, the running
 * app, Training Curves at a 738px plot width) in `measuredUnconstrained` —
 * if the model ever stops matching what the real chart did, that table fails
 * first and this whole spec stops making claims it cannot back.
 */

const X_AXIS_SPACE = 50;      // uPlot's default `axis.space` for the x side
const MEASURED_PLOT_PX = 738; // measured: u.bbox.width / devicePixelRatio

/**
 * uPlot's default numeric increment ladder, `[1, 2, 2.5, 5] x 10^n` ascending
 * (`allMults` — uplot/dist/uPlot.esm.js:985, v1.6.32). The 2.5 mantissa is why
 * an unconstrained axis can land on 0.25 as well as 0.5.
 */
function uplotDefaultIncrs(): number[] {
    const out: number[] = [];
    for (let e = -6; e <= 9; e++) {
        for (const m of [1, 2, 2.5, 5]) out.push(+(m * 10 ** e).toPrecision(12));
    }
    return out.sort((a, b) => a - b);
}

function chooseIncr(incrs: number[], span: number, plotPx: number, space = X_AXIS_SPACE): number {
    for (const incr of incrs) {
        if ((plotPx * incr) / span >= space) return incr;
    }
    return incrs[incrs.length - 1];
}

function splitsFor(incr: number, min: number, max: number): number[] {
    const out: number[] = [];
    const start = Math.ceil(min / incr) * incr;
    for (let v = start, i = 0; v <= max + 1e-9 && i < 500; i++, v = start + incr * i) {
        out.push(+v.toPrecision(12));
    }
    return out;
}

/**
 * What the UNCONSTRAINED axis really produced in the browser, read off the live
 * uPlot instance (`u.axes[0]._splits`) after forcing each span at 738px.
 * This is the defect, recorded: spans of 6 or fewer steps grow half-steps.
 */
const measuredUnconstrained: ReadonlyArray<{ span: number; incr: number }> = [
    { span: 500, incr: 50 },
    { span: 100, incr: 10 },
    { span: 50, incr: 5 },
    { span: 20, incr: 2 },
    { span: 12, incr: 1 },
    { span: 8, incr: 1 },
    { span: 6, incr: 0.5 },   // <- first fractional span
    { span: 4, incr: 0.5 },
    { span: 3, incr: 0.25 },
    { span: 2, incr: 0.2 },
];

describe('integer-axis model matches the measured browser behaviour', () => {
    it.each(measuredUnconstrained)(
        'unconstrained axis at span $span picks increment $incr (as measured in Chrome)',
        ({ span, incr }) => {
            expect(chooseIncr(uplotDefaultIncrs(), span, MEASURED_PLOT_PX)).toBeCloseTo(incr, 10);
        },
    );

    it('POSITIVE CONTROL: the unconstrained axis really does emit 5974.5', () => {
        // The exact shape the user reported: a long run, a handful of steps visible.
        const incr = chooseIncr(uplotDefaultIncrs(), 6, MEASURED_PLOT_PX);
        const splits = splitsFor(incr, 5970, 5976);
        expect(incr).toBe(0.5);
        expect(splits).toContain(5974.5);
        expect(splits.some(v => !Number.isInteger(v))).toBe(true);
    });
});

describe('integerTickIncrs', () => {
    it('offers only positive integers — the property that makes a fractional tick unrepresentable', () => {
        const incrs = integerTickIncrs();
        expect(incrs.length).toBeGreaterThan(0);
        expect(incrs.filter(v => !Number.isInteger(v) || v <= 0)).toEqual([]);
    });

    it('is strictly ascending, as uPlot requires of a candidate ladder', () => {
        const incrs = integerTickIncrs();
        expect(incrs).toEqual([...incrs].sort((a, b) => a - b));
        expect(new Set(incrs).size).toBe(incrs.length);
    });

    it('starts at 1 and reaches far beyond any plausible step count', () => {
        const incrs = integerTickIncrs();
        expect(incrs[0]).toBe(1);
        expect(incrs[incrs.length - 1]).toBeGreaterThanOrEqual(1_000_000);
    });

    it('hands out a fresh array so two charts can never share one options object', () => {
        const a = integerTickIncrs();
        a[0] = 0.5;
        expect(integerTickIncrs()[0]).toBe(1);
    });
});

describe('a constrained axis cannot emit a fractional tick', () => {
    // Every span the unconstrained axis got wrong, plus the degenerate ones.
    it.each([1, 2, 3, 4, 5, 6, 8, 12, 20, 50, 100, 500, 6000, 120_000])(
        'span %i: every generated split is a whole step',
        span => {
            const incr = chooseIncr(integerTickIncrs(), span, MEASURED_PLOT_PX);
            expect(Number.isInteger(incr)).toBe(true);
            const splits = splitsFor(incr, 5974 - span, 5974);
            expect(splits.length).toBeGreaterThan(0);
            expect(splits.filter(v => !Number.isInteger(v))).toEqual([]);
        },
    );

    it('holds at any plot width, not just the one that was measured', () => {
        for (const px of [200, 320, 738, 1200, 2560, 3840]) {
            for (let span = 1; span <= 40; span++) {
                const incr = chooseIncr(integerTickIncrs(), span, px);
                expect(splitsFor(incr, 0, span).filter(v => !Number.isInteger(v))).toEqual([]);
            }
        }
    });
});

describe('integerTickValues', () => {
    it('labels whole steps without inventing a fractional part', () => {
        expect(integerTickValues({} as uPlot, [5974, 5975])).toEqual([
            uPlot.fmtNum(5974), uPlot.fmtNum(5975),
        ]);
    });

    it('rounds a fractional split rather than printing it', () => {
        // Belt and braces only: `incrs` is what stops the tick existing. If this
        // formatter ever became the fix, two adjacent ticks would share a label.
        expect(integerTickValues({} as uPlot, [5974.5])).toEqual([uPlot.fmtNum(5975)]);
        expect(integerTickValues({} as uPlot, [5974.2])).toEqual([uPlot.fmtNum(5974)]);
    });

    it('renders null and non-finite splits as empty, never as "NaN"', () => {
        expect(integerTickValues({} as uPlot, [null as unknown as number, NaN, Infinity]))
            .toEqual(['', '', '']);
    });
});

describe('integerAxis', () => {
    it('carries the increment constraint and the matching formatter', () => {
        const ax = integerAxis();
        expect(ax.incrs).toEqual(integerTickIncrs());
        expect(ax.values).toBe(integerTickValues);
    });

    it('merges caller styling without dropping the constraint', () => {
        const ax = integerAxis({ size: 36, stroke: '#4b5563' });
        expect(ax.size).toBe(36);
        expect(ax.stroke).toBe('#4b5563');
        expect(ax.incrs).toEqual(integerTickIncrs());
    });

    it('lets a caller keep its own label text (lora-tools looks names up by index)', () => {
        const byIndex = () => ['conv_in'];
        const ax = integerAxis({ values: byIndex });
        expect(ax.values).toBe(byIndex);
        expect(ax.incrs).toEqual(integerTickIncrs());
    });

    it('refuses to let a caller reintroduce fractional increments', () => {
        const ax = integerAxis({ incrs: [0.1, 0.5, 1] });
        expect((ax.incrs as number[]).filter(v => !Number.isInteger(v))).toEqual([]);
    });
});
