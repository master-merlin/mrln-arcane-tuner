import { describe, it, expect } from 'vitest';
import { TestBed } from '@angular/core/testing';

import { TrainingChartComponent, type ChartDataPoint } from './training-chart';

/**
 * Adaptive layer targeting narrowing series (Task 11, decision D3).
 *
 * D3's load-bearing claim: with the feature OFF (no point carries
 * `adaptive_active`/`adaptive_hot`) the chart must be byte-identical to
 * before — same series count, same axes, same scales. `createChart()` feeds
 * uPlot from exactly the array `buildUPlotData()` returns, so a slot-count
 * mismatch there is the same bug a naive "always add the two series" change
 * would produce. Verified here against the PRIVATE data builder rather than
 * a mounted uPlot instance — jsdom has no canvas backend, and mounting is
 * unnecessary to prove the slot-count contract.
 */
function buildComponent(data: ChartDataPoint[]) {
    TestBed.configureTestingModule({ imports: [TrainingChartComponent] });
    const fixture = TestBed.createComponent(TrainingChartComponent);
    fixture.componentRef.setInput('data', data);
    // Deliberately never call fixture.detectChanges() — that would run
    // ngAfterViewInit()/createChart(), which constructs a real uPlot (needs
    // a canvas 2D context jsdom doesn't provide). The private builder under
    // test doesn't touch the DOM at all.
    return fixture.componentInstance as unknown as {
        hasAdaptive: () => boolean;
        buildUPlotData: () => ArrayLike<unknown>[];
    };
}

const pt = (over: Partial<ChartDataPoint> = {}): ChartDataPoint => ({
    step: 1,
    loss: 0.5,
    lr: 1e-4,
    ...over,
});

describe('TrainingChartComponent adaptive series (T11, D3)', () => {
    it('feature off: AdamW data keeps the pre-existing 6 slots (no adaptive slots added)', () => {
        const comp = buildComponent([pt({ step: 1 }), pt({ step: 2 })]);
        expect(comp.hasAdaptive()).toBe(false);
        expect(comp.buildUPlotData()).toHaveLength(6);
    });

    it('feature off: Prodigy data (d_estimate present) keeps the pre-existing 5 slots', () => {
        const comp = buildComponent([
            pt({ step: 1, d_estimate: 0.1 }),
            pt({ step: 2, d_estimate: 0.2 }),
        ]);
        expect(comp.hasAdaptive()).toBe(false);
        expect(comp.buildUPlotData()).toHaveLength(5);
    });

    it('empty data set: slot count still matches the (adaptive-off) pre-existing shape', () => {
        const comp = buildComponent([]);
        expect(comp.hasAdaptive()).toBe(false);
        expect(comp.buildUPlotData()).toHaveLength(6);
    });

    it('feature on: AdamW data gains exactly 2 trailing count slots (8 total)', () => {
        const comp = buildComponent([
            pt({ step: 1, adaptive_active: 8, adaptive_hot: 4 }),
            pt({ step: 2, adaptive_active: 6, adaptive_hot: 4 }),
        ]);
        expect(comp.hasAdaptive()).toBe(true);
        const out = comp.buildUPlotData();
        expect(out).toHaveLength(8);
        expect(Array.from(out[6] as ArrayLike<number | null>)).toEqual([8, 6]);
        expect(Array.from(out[7] as ArrayLike<number | null>)).toEqual([4, 4]);
    });

    it('feature on: Prodigy data gains exactly 2 trailing count slots (7 total)', () => {
        const comp = buildComponent([
            pt({ step: 1, d_estimate: 0.1, adaptive_active: 8, adaptive_hot: 4 }),
            pt({ step: 2, d_estimate: 0.2, adaptive_active: 6, adaptive_hot: 4 }),
        ]);
        expect(comp.hasAdaptive()).toBe(true);
        expect(comp.buildUPlotData()).toHaveLength(7);
    });

    it('adaptive detection scans every point, not just the first', () => {
        const comp = buildComponent([pt({ step: 1 }), pt({ step: 2, adaptive_active: 3 })]);
        expect(comp.hasAdaptive()).toBe(true);
    });
});
