import uPlot from 'uplot';
import type { ActivityWeek } from '../../services/job';
import { tooltipPlugin, type TooltipFormatter } from './stats-tooltip';

const WEEK_MS = 7 * 24 * 3600 * 1000;

export interface ActivityChart {
    xs: number[];            // unix seconds of each week's Monday (uPlot x)
    completedCum: number[];  // completed
    failedCum: number[];     // completed + failed
    stoppedCum: number[];    // completed + failed + stopped + other
    labels: string[];        // ISO date labels, aligned with xs
}

/** Gap-fill missing ISO weeks and pre-stack outcome counts cumulatively.
 *  Rendering order (bottom→top color): stoppedCum drawn first (full height),
 *  then failedCum, then completedCum on top. */
export function buildActivityChart(weeks: ActivityWeek[]): ActivityChart {
    const out: ActivityChart = { xs: [], completedCum: [], failedCum: [], stoppedCum: [], labels: [] };
    if (!weeks.length) return out;
    const byWeek = new Map(weeks.map(w => [w.week_start, w]));
    const first = Date.parse(weeks[0].week_start + 'T00:00:00Z');
    const last = Date.parse(weeks[weeks.length - 1].week_start + 'T00:00:00Z');
    for (let t = first; t <= last; t += WEEK_MS) {
        const iso = new Date(t).toISOString().slice(0, 10);
        const w = byWeek.get(iso) ?? { week_start: iso, completed: 0, failed: 0, stopped: 0, other: 0 };
        out.xs.push(t / 1000);
        out.labels.push(iso);
        out.completedCum.push(w.completed);
        out.failedCum.push(w.completed + w.failed);
        out.stoppedCum.push(w.completed + w.failed + w.stopped + w.other);
    }
    return out;
}

/** Bin centers + counts for a bar rendering; null when there is no histogram. */
export function buildHistogramChart(
    h: { edges: number[]; counts: number[] },
): { xs: number[]; counts: number[] } | null {
    if (!h.counts.length || h.edges.length < 2) return null;
    const xs = h.counts.map((_, i) => (h.edges[i] + h.edges[i + 1]) / 2);
    return { xs, counts: h.counts };
}

/** One curve row this builder reads — subset of `JobMetricsCurveRow` (job.ts). */
export interface AdaptiveCurveRow { step: number; active_layers: number | null; }

/**
 * Active-module-count staircase, aligned to step. NULL rows ("no data for
 * this step") are skipped rather than plotted as 0 — 0 is itself a real,
 * meaningful value ("every remaining layer just froze") and must stay
 * distinguishable from "no data" (mirrors the NULL-never-0 contract on
 * `LossCurvePoint.active_layers`, backend history_routes.py).
 */
export function buildAdaptiveSeries(
    rows: ReadonlyArray<AdaptiveCurveRow>,
): { steps: number[]; counts: number[] } {
    const steps: number[] = [];
    const counts: number[] = [];
    for (const r of rows) {
        if (r.active_layers == null) continue;
        steps.push(r.step);
        counts.push(r.active_layers);
    }
    return { steps, counts };
}

// ── Axis theming + chart options ─────────────────────────────────────────

export interface AxisTheme { stroke: string; grid: string; font: string; }
export interface SeriesColors { success: string; danger: string; warning: string; brand: string; }

/** Resolve axis colors from the design tokens (jsdom-safe fallbacks). */
export function readAxisTheme(): AxisTheme {
    const css = getComputedStyle(document.documentElement);
    return {
        stroke: css.getPropertyValue('--color-text-muted').trim() || '#8a8f98',
        grid: css.getPropertyValue('--color-border-subtle').trim() || 'rgba(128,128,128,0.18)',
        font: '10px monospace',
    };
}

/** uPlot axis with theme colors applied; `extra` merges on top. */
export function themedAxis(theme: AxisTheme, extra: Partial<uPlot.Axis> = {}): uPlot.Axis {
    return {
        stroke: theme.stroke,
        font: theme.font,
        ticks: { stroke: theme.grid, width: 1 },
        grid: { stroke: theme.grid, width: 1 },
        ...extra,
    };
}

const CURSOR: uPlot.Cursor = { points: { show: false }, drag: { x: false, y: false } };

export function activityTooltip(chart: ActivityChart): TooltipFormatter {
    return (_u, idx) => {
        const done = chart.completedCum[idx] ?? 0;
        const failed = (chart.failedCum[idx] ?? 0) - done;
        const stopped = (chart.stoppedCum[idx] ?? 0) - (chart.failedCum[idx] ?? 0);
        const label = chart.labels[idx];
        if (label == null) return null;
        return `week of ${label}: ${done} completed · ${failed} failed · ${stopped} stopped/other`;
    };
}

export function histogramTooltip(edges: number[]): TooltipFormatter {
    return (u, idx) => {
        const count = u.data[1]?.[idx];
        if (count == null) return null;
        return `loss ${edges[idx].toFixed(3)}–${edges[idx + 1].toFixed(3)}: ${count} runs`;
    };
}

export function adaptiveTooltip(): TooltipFormatter {
    return (u, idx) => {
        const step = u.data[0]?.[idx];
        const count = u.data[1]?.[idx];
        if (step == null || count == null) return null;
        return `step ${step}: ${count} active layers`;
    };
}

export function buildActivityOpts(
    theme: AxisTheme, colors: SeriesColors, chart: ActivityChart,
): Omit<uPlot.Options, 'width' | 'height'> {
    const bars = uPlot.paths.bars!({ size: [0.6, 100] });
    return {
        legend: { show: false },
        cursor: CURSOR,
        scales: { x: { time: true } },
        axes: [
            themedAxis(theme),
            themedAxis(theme, { size: 36, incrs: [1, 2, 5, 10, 25, 50, 100] }),
        ],
        series: [
            {},
            // draw order bottom layer first: full cumulative in "stopped" color
            { paths: bars, fill: colors.warning, stroke: 'transparent', points: { show: false } },
            { paths: bars, fill: colors.danger, stroke: 'transparent', points: { show: false } },
            { paths: bars, fill: colors.success, stroke: 'transparent', points: { show: false } },
        ],
        plugins: [tooltipPlugin(activityTooltip(chart))],
    };
}

export function buildHistogramOpts(
    theme: AxisTheme, barColor: string, edges: number[],
): Omit<uPlot.Options, 'width' | 'height'> {
    const bars = uPlot.paths.bars!({ size: [0.8, 100] });
    return {
        legend: { show: false },
        cursor: CURSOR,
        scales: { x: { time: false } },
        axes: [
            themedAxis(theme, { values: (_u, vals) => vals.map(v => Number(v).toFixed(3)) }),
            themedAxis(theme, { size: 36 }),
        ],
        series: [
            {},
            { paths: bars, fill: barColor, stroke: 'transparent', points: { show: false } },
        ],
        plugins: [tooltipPlugin(histogramTooltip(edges))],
    };
}

/** Stepped staircase chart opts — mirrors the live training-chart's narrowing
 *  series (`uPlot.paths.stepped`, Task 11) so the post-hoc replay reads the same. */
export function buildAdaptiveOpts(
    theme: AxisTheme, color: string,
): Omit<uPlot.Options, 'width' | 'height'> {
    const steppedPath = uPlot.paths.stepped ? uPlot.paths.stepped({ align: 1 }) : undefined;
    return {
        legend: { show: false },
        cursor: CURSOR,
        scales: { x: { time: false } },
        axes: [
            themedAxis(theme),
            themedAxis(theme, { size: 36 }),
        ],
        series: [
            {},
            { stroke: color, width: 2, paths: steppedPath, points: { show: false } },
        ],
        plugins: [tooltipPlugin(adaptiveTooltip())],
    };
}
