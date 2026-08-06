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
    /** Adaptive layer targeting (Task 5): count of currently-trainable LoRA modules. */
    adaptive_active?: number;
    /** Count in the "essential" tier from the controller's last analysis. */
    adaptive_hot?: number;
    /** Total module count — emitted once per process (re-appears after a rebuild restart). */
    adaptive_total?: number;
    [k: string]: unknown;
}

export interface LossPoint {
    step: number;
    loss: number;
    lr: number;
    grad_norm?: number;
    d_estimate?: number;
    /** Adaptive layer targeting narrowing series (Task 11) — see StepMetrics. */
    adaptive_active?: number;
    adaptive_hot?: number;
}

export interface AdaptEvent {
    step: number;
    kind: 'narrow' | 'probe_open' | 'probe_apply' | 'rebuild_request';
    active_count: number;
    total_count: number;
    hot_count?: number;
    active_param_pct?: number;
    earliest_active_block?: number | null;
    /**
     * Fields below are always present on the durable per-run history
     * (`GET /jobs/history/{job_id}/adaptive`, Task 12) — the live `{"adapt":
     * …}` broadcast (Task 11) carries the SAME dict, but the two live-view
     * tests only exercised the subset above, so these stayed optional here
     * rather than widen an already-covered call site's assumptions.
     */
    /** Sequence number within a run's adaptive timeline; stable identity for
     *  list rendering (do NOT `@for … track $index` — a rebuild detaches
     *  rows keyed that way). */
    event_index?: number;
    frozen_this_event?: number;
    reactivated_this_event?: number;
    top_modules?: string[];
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
                adaptive_active: m.adaptive_active,
                adaptive_hot: m.adaptive_hot,
            });
        }
    }
    return points;
}

/**
 * Parse `{"adapt": {...}}` broadcast lines (adaptive layer targeting analysis
 * events — Task 7's websocket `job_log` payloads). These carry a `step` field
 * but no `loss`/`status`, so `parseStepLog`'s top-level-`step` check already
 * keeps them out of `lossSeries`/`latestMetrics` — this is a SEPARATE parse
 * over the same `job.logs` array, not a filter on top of the step parser.
 */
export function adaptEvents(logs: ReadonlyArray<string> | undefined): AdaptEvent[] {
    if (!logs) return [];
    const out: AdaptEvent[] = [];
    for (const line of logs) {
        if (!line.includes('"adapt"')) continue;
        try {
            const parsed = JSON.parse(line);
            const a = parsed?.adapt;
            if (a && typeof a.step === 'number' && typeof a.kind === 'string') {
                out.push(a as AdaptEvent);
            }
        } catch {
            // not JSON — skip
        }
    }
    return out;
}

/**
 * Most recent adaptive-layer-targeting event, or null. `job.logs` is
 * live-session-only (cleared + rotated on a rebuild restart), so this is
 * "latest known state this session" — NOT durable history. The persisted
 * timeline lives at GET /api/jobs/history/{job_id}/adaptive (consumed
 * elsewhere, not here).
 */
export function latestAdaptState(logs: ReadonlyArray<string> | undefined): AdaptEvent | null {
    const events = adaptEvents(logs);
    return events.length ? events[events.length - 1] : null;
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

/**
 * Parse a bucket `resolution` string to megapixels (×frames for video).
 * Accepts "WxH" (image) and "WxHxFf" (video, F frames), e.g. "1888x1056" → 2.0,
 * "1888x1056x25f" → 49.8. Returns null for unparseable/absent input.
 */
export function resolutionToMpx(res: string | undefined | null): number | null {
    if (!res) return null;
    const m = /^(\d+)x(\d+)(?:x(\d+)f)?$/.exec(res.trim());
    if (!m) return null;
    const frames = m[3] ? Number(m[3]) : 1;
    return (Number(m[1]) * Number(m[2]) * frames) / 1e6;
}

/** Last `n` per-step megapixel values, derived from the `resolution` field. */
export function resolutionMpxSpark(
    logs: ReadonlyArray<string> | undefined,
    n = 24,
): number[] {
    if (!logs) return [];
    const out: number[] = [];
    for (const line of logs) {
        const mpx = resolutionToMpx(parseStepLog(line)?.resolution);
        if (mpx != null && Number.isFinite(mpx)) out.push(mpx);
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
