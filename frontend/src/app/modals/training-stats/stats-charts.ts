import type { ActivityWeek } from '../../services/job';

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
