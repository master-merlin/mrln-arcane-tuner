import {
    buildActivityChart, buildHistogramChart,
    readAxisTheme, themedAxis, buildActivityOpts, buildHistogramOpts,
    activityTooltip, histogramTooltip,
} from './stats-charts';

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

describe('axis theming', () => {
    it('themedAxis applies theme stroke/grid/font and merges extras', () => {
        const theme = { stroke: '#aaa', grid: '#333', font: '10px monospace' };
        const ax = themedAxis(theme, { size: 36 });
        expect(ax.stroke).toBe('#aaa');
        expect(ax.font).toBe('10px monospace');
        expect((ax.grid as { stroke: string }).stroke).toBe('#333');
        expect((ax.ticks as { stroke: string }).stroke).toBe('#333');
        expect(ax.size).toBe(36);
    });

    it('readAxisTheme falls back when CSS vars are absent (jsdom)', () => {
        const theme = readAxisTheme();
        expect(theme.stroke.length).toBeGreaterThan(0);
        expect(theme.grid.length).toBeGreaterThan(0);
    });

    it('both opts builders produce themed axes, visible cursor and a tooltip plugin', () => {
        const theme = { stroke: '#aaa', grid: '#333', font: '10px monospace' };
        const chart = buildActivityChart([
            { week_start: '2026-07-13', completed: 3, failed: 1, stopped: 1, other: 0 },
        ]);
        const colors = { success: '#0f0', danger: '#f00', warning: '#fa0', brand: '#00f' };
        const act = buildActivityOpts(theme, colors, chart);
        const hist = buildHistogramOpts(theme, '#00f', [0.1, 0.2, 0.3]);
        for (const opts of [act, hist]) {
            expect(opts.cursor?.show).not.toBe(false);
            expect(opts.axes?.every(a => a.stroke === '#aaa')).toBe(true);
            expect(opts.plugins?.length).toBe(1);
        }
    });
});

describe('tooltip formatters', () => {
    it('activity tooltip de-cumulates the stacked series', () => {
        const chart = buildActivityChart([
            { week_start: '2026-07-13', completed: 3, failed: 1, stopped: 1, other: 0 },
        ]);
        const text = activityTooltip(chart)(null as never, 0);
        expect(text).toBe('week of 2026-07-13: 3 completed · 1 failed · 1 stopped/other');
    });

    it('histogram tooltip shows the bin range and count', () => {
        const u = { data: [[0.15, 0.25], [2, 5]] } as never;
        expect(histogramTooltip([0.1, 0.2, 0.3])(u, 1)).toBe('loss 0.200–0.300: 5 runs');
        const uNull = { data: [[0.15], [null]] } as never;
        expect(histogramTooltip([0.1, 0.2])(uNull, 0)).toBeNull();
    });
});
