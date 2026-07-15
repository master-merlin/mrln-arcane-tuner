import { buildActivityChart, buildHistogramChart } from './stats-charts';

describe('buildActivityChart', () => {
    it('gap-fills missing weeks and stacks cumulatively', () => {
        const chart = buildActivityChart([
            { week_start: '2026-06-01', completed: 2, failed: 1, stopped: 0, other: 0 },
            // 2026-06-08 missing → must be zero-filled
            { week_start: '2026-06-15', completed: 0, failed: 0, stopped: 1, other: 1 },
        ]);
        expect(chart.labels).toEqual(['2026-06-01', '2026-06-08', '2026-06-15']);
        expect(chart.xs.length).toBe(3);
        // cumulative stacking: completed ≤ +failed ≤ +stopped(+other)
        expect(chart.completedCum).toEqual([2, 0, 0]);
        expect(chart.failedCum).toEqual([3, 0, 0]);
        expect(chart.stoppedCum).toEqual([3, 0, 2]);   // other folds into stopped
    });

    it('returns empty arrays for no activity', () => {
        const chart = buildActivityChart([]);
        expect(chart.xs).toEqual([]);
    });
});

describe('buildHistogramChart', () => {
    it('maps bin edges to centers', () => {
        const r = buildHistogramChart({ edges: [0, 1, 2], counts: [3, 5] })!;
        expect(r.xs).toEqual([0.5, 1.5]);
        expect(r.counts).toEqual([3, 5]);
    });

    it('returns null for an empty histogram', () => {
        expect(buildHistogramChart({ edges: [], counts: [] })).toBeNull();
    });

    it('handles the single-bin degenerate shape', () => {
        const r = buildHistogramChart({ edges: [0.5, 0.5], counts: [3] })!;
        expect(r.xs).toEqual([0.5]);
        expect(r.counts).toEqual([3]);
    });
});
