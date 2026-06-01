/**
 * Pure parsers for training-job log streams.
 *
 * The trainer emits one JSON object per step, optionally prefixed with
 * `STEP_LOG:`. These helpers turn the raw `job.logs` string array into the
 * structured metrics the Jobs screen (KPI row, curves, status) renders.
 *
 * Kept framework-free and side-effect-free so they're shared by both the
 * Jobs-screen detail pane and the training-job-queue without coupling.
 */

const STEP_LOG_PREFIX = 'STEP_LOG:';

/** Throttled fields emitted only every N steps — carry forward last known. */
const CARRY_FORWARD_KEYS = ['vram_allocated_mb', 'vram_reserved_mb', 'amp_scale', 'resolution'] as const;

export interface StepMetrics {
    status?: string;
    step: number;
    loss?: number;
    learning_rate?: number;
    grad_norm?: number;
    d_estimate?: number;
    epoch?: number;
    progress?: number;
    step_time?: number;
    samples_per_sec?: number;
    eta?: number;
    vram_allocated_mb?: number;
    vram_reserved_mb?: number;
    resolution?: string;
    nan_count?: number;
    total_steps?: number | string;
    [k: string]: unknown;
}

export interface LossPoint {
    step: number;
    loss: number;
    lr: number;
    grad_norm?: number;
    d_estimate?: number;
}

export type LossTone = 'success' | 'warning' | 'danger';

export interface LossStatus {
    text: string;
    tone: LossTone;
    tooltip: string;
}

/** Parse one log line into a metrics object, or null if it isn't a STEP_LOG. */
export function parseStepLog(line: string): StepMetrics | null {
    let jsonStr = line;
    if (line.includes(STEP_LOG_PREFIX)) {
        jsonStr = line.split(STEP_LOG_PREFIX)[1];
    } else if (!line.trim().startsWith('{')) {
        return null;
    }
    try {
        const m = JSON.parse(jsonStr);
        return typeof m?.step === 'number' ? (m as StepMetrics) : null;
    } catch {
        return null;
    }
}

/**
 * Most recent `status:'training'` metrics, back-filling throttled fields from
 * recent history and resolving `total_steps`.
 */
export function latestMetrics(
    logs: ReadonlyArray<string> | undefined,
    maxTrainSteps?: number,
): StepMetrics | null {
    if (!logs || logs.length === 0) return null;
    for (let i = logs.length - 1; i >= 0; i--) {
        const m = parseStepLog(logs[i]);
        if (m && m.status === 'training') {
            const totalSteps =
                maxTrainSteps ||
                (m.progress ? Math.round(m.step / (m.progress / 100)) : 0) ||
                '?';

            const carryForward: Record<string, unknown> = {};
            const needed = new Set(CARRY_FORWARD_KEYS.filter((k) => m[k] == null));
            if (needed.size > 0) {
                const lookback = Math.min(50, i);
                for (let j = i - 1; j >= i - lookback && needed.size > 0; j--) {
                    const prev = parseStepLog(logs[j]);
                    if (!prev) continue;
                    for (const key of [...needed]) {
                        if (prev[key] != null) {
                            carryForward[key] = prev[key];
                            needed.delete(key);
                        }
                    }
                }
            }
            return { ...m, ...carryForward, total_steps: totalSteps };
        }
    }
    return null;
}

/** Loss/LR series for charting (steps ≥ `startStep`, finite losses only). */
export function lossSeries(logs: ReadonlyArray<string> | undefined, startStep = 5): LossPoint[] {
    if (!logs) return [];
    const points: LossPoint[] = [];
    for (const line of logs) {
        const m = parseStepLog(line);
        if (m && m.step >= startStep && typeof m.loss === 'number') {
            points.push({
                step: m.step,
                loss: m.loss,
                lr: m.learning_rate ?? 0,
                grad_norm: m.grad_norm,
                d_estimate: m.d_estimate,
            });
        }
    }
    return points;
}

/** Lowest loss in a series with the step it occurred at. */
export function bestLoss(series: ReadonlyArray<LossPoint>): { loss: number; step: number } | null {
    let best = Infinity;
    let step = 0;
    for (const p of series) {
        if (p.loss < best) {
            best = p.loss;
            step = p.step;
        }
    }
    return Number.isFinite(best) ? { loss: best, step } : null;
}

/** Last `n` loss values (oldest→newest) for a KPI sparkline. */
export function lossSpark(series: ReadonlyArray<LossPoint>, n = 24): number[] {
    return series.slice(-n).map((p) => p.loss);
}

/** Running-minimum loss series (the "best loss" descending staircase). */
export function bestLossSpark(series: ReadonlyArray<LossPoint>, n = 24): number[] {
    let min = Infinity;
    const out: number[] = [];
    for (const p of series) {
        if (p.loss < min) min = p.loss;
        out.push(min);
    }
    return out.slice(-n);
}

/** Last `n` values of an arbitrary numeric step field, for a sparkline. */
export function metricSpark(
    logs: ReadonlyArray<string> | undefined,
    key: string,
    n = 24,
): number[] {
    if (!logs) return [];
    const out: number[] = [];
    for (const line of logs) {
        const m = parseStepLog(line);
        const v = m?.[key];
        if (typeof v === 'number' && Number.isFinite(v)) out.push(v);
    }
    return out.slice(-n);
}

export interface LogLine {
    tone: 'teal' | 'warning' | 'danger';
    level: string;
    text: string;
}

/**
 * Last `n` human-readable log lines (STEP_LOG JSON filtered out), classified
 * by severity for the colored log-tail tags. Falls back to raw lines if every
 * line is a STEP_LOG entry.
 */
export function logTail(logs: ReadonlyArray<string> | undefined, n = 12): LogLine[] {
    if (!logs || logs.length === 0) return [];
    const human = logs.filter(
        (l) => !l.includes(STEP_LOG_PREFIX) && !l.trim().startsWith('{'),
    );
    const src = human.length ? human : logs;
    return src.slice(-n).map((line) => {
        const upper = line.toUpperCase();
        let tone: LogLine['tone'] = 'teal';
        let level = 'INFO';
        if (upper.includes('ERROR') || upper.includes('TRACEBACK') || upper.includes('CRITICAL')) {
            tone = 'danger';
            level = 'ERR';
        } else if (upper.includes('WARN')) {
            tone = 'warning';
            level = 'WARN';
        }
        return { tone, level, text: line };
    });
}

/**
 * Convergence verdict by comparing the recent vs. earlier loss average over a
 * `window`-sized tail. Needs ≥ `window` parsed losses to report.
 */
/**
 * Convergence/plateau/divergence sliding window (steps). ~half a sample cycle
 * for most runs — wide enough to smooth heavy step-to-step jitter. Shared so
 * the queue mini-cards and the center detail pane render the SAME verdict for a
 * job (they were drifting at 50 vs 125 before).
 */
export const CONVERGENCE_WINDOW = 125;

export function lossStatus(
    logs: ReadonlyArray<string> | undefined,
    window = CONVERGENCE_WINDOW,
): LossStatus | null {
    if (!logs || logs.length < window) return null;
    const losses: number[] = [];
    for (let i = logs.length - 1; i >= 0 && losses.length < window * 2; i--) {
        const m = parseStepLog(logs[i]);
        if (m?.loss != null) losses.unshift(m.loss);
    }
    if (losses.length < window) return null;
    const recent = losses.slice(-window).reduce((a, b) => a + b, 0) / window;
    const earlier = losses.slice(0, window).reduce((a, b) => a + b, 0) / window;
    const delta = (recent - earlier) / Math.max(earlier, 1e-8);

    let text = 'plateau';
    let tone: LossTone = 'warning';
    if (delta < -0.01) {
        text = 'converging';
        tone = 'success';
    } else if (delta > 0.02) {
        text = 'diverging';
        tone = 'danger';
    }
    return {
        text,
        tone,
        tooltip: `Last ${window} steps · recent avg ${recent.toFixed(5)} · earlier avg ${earlier.toFixed(5)}`,
    };
}

/** `1h 36m` / `4m 12s` / `--:--` ETA formatting from seconds. */
export function formatEta(seconds: number | undefined | null): string {
    if (!seconds || seconds < 0) return '--:--';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m ${s}s`;
}

/** Elapsed `h:mm:ss` / `m:ss` from a start (epoch seconds) to an end (epoch ms). */
export function formatDuration(startedAtSec: number | undefined, endMs: number): string {
    if (!startedAtSec) return '0:00';
    const seconds = Math.floor((endMs - startedAtSec * 1000) / 1000);
    if (seconds < 0) return '0:00';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    return `${m}:${s.toString().padStart(2, '0')}`;
}

/** Grad-norm formatting: scientific for huge values, else fixed precision. */
export function formatGradNorm(gn: number | undefined | null): string {
    if (gn == null) return '—';
    if (gn >= 1000) return gn.toExponential(1);
    if (gn >= 1) return gn.toFixed(2);
    return gn.toFixed(4);
}
