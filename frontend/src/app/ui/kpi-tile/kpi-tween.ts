/**
 * Pure helpers for the KPI tile's numeric count-up animation.
 *
 * Live counters (e.g. the training "Step" tile) receive values in bursts —
 * the backend streams progress and zoneless coalesces rapid updates — so a
 * raw binding visibly jumps (0 → 3 → 11 → 19). These helpers let the tile
 * glide between values instead, while snapping instantly for cases where a
 * count-up would look wrong (decreases, or a large initial jump when an
 * already-running job is first selected).
 */

/** Maximum forward delta we animate; larger jumps snap (avoids long counts). */
const MAX_TWEEN_DELTA = 200;

/** easeOutCubic — fast start, gentle landing. */
function easeOutCubic(t: number): number {
    return 1 - Math.pow(1 - t, 3);
}

/**
 * Eased integer value along the tween from `from` to `to` at progress `t`
 * (clamped to [0,1]). Always returns an integer so the counter never shows a
 * fractional step.
 */
export function tweenStep(from: number, to: number, t: number): number {
    const clamped = t <= 0 ? 0 : t >= 1 ? 1 : t;
    return Math.round(from + (to - from) * easeOutCubic(clamped));
}

/**
 * Whether to animate the transition from `from` to `to`. Only smooth, bounded
 * forward progress animates; decreases, no-ops, and huge jumps snap instantly.
 */
export function shouldTween(from: number, to: number): boolean {
    return to > from && to - from <= MAX_TWEEN_DELTA;
}
