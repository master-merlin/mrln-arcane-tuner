/**
 * Shared canvas-footer metadata builder. Both Details mode and Edit mode
 * render the same `<app-canvas-footer>` strip; this helper keeps their
 * res/AR/orientation/size/HPS derivation identical so the two surfaces
 * can't drift out of sync.
 *
 * `hasOverlay` is intentionally NOT included — Details and Edit derive
 * it differently (Details from `metadata.has_overlay`, Edit from the
 * already-resolved `hasOverlay` input). The caller spreads in their own
 * value when building the final `CanvasMeta`.
 */

export interface BuiltCanvasMetaCore {
    res: string | null;
    ar: string | null;
    orientation: string | null;
    size: string | null;
    hpsLabel: string | null;
    hpsTone: 'success' | 'warning' | 'danger' | null;
    /** Video-only: framerate label (e.g. "24 fps"), null for stills. */
    fps: string | null;
    /** Video-only: duration label (e.g. "0:05"), null for stills. */
    duration: string | null;
    /** Video-only: frame-count label (e.g. "120 frames", "~120 frames"). */
    frameCount: string | null;
}

export function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

/** Format a duration in seconds as ``m:ss`` (e.g. 65 → "1:05"). */
export function formatDuration(seconds: number): string {
    const total = Math.max(0, Math.round(seconds));
    const m = Math.floor(total / 60);
    const s = total % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
}

export function buildCanvasMeta(
    metadata: Record<string, unknown> | null | undefined,
): BuiltCanvasMetaCore {
    const m = metadata ?? {};
    const w = typeof m['width']  === 'number' ? (m['width']  as number) : null;
    const h = typeof m['height'] === 'number' ? (m['height'] as number) : null;
    const res = w && h ? `${w}×${h}` : null;
    const ar = typeof m['aspect_ratio'] === 'number'
        ? (m['aspect_ratio'] as number).toFixed(3)
        : (w && h ? (w / h).toFixed(3) : null);
    const orientation = typeof m['orientation'] === 'string'
        ? (m['orientation'] as string) : null;
    const size = typeof m['size_bytes'] === 'number'
        ? formatBytes(m['size_bytes'] as number) : null;
    const q = m['quality_score'];
    const hpsLabel = typeof q === 'number' ? `HPS ${q.toFixed(4)}` : null;
    const hpsTone: 'success' | 'warning' | 'danger' | null = typeof q === 'number'
        ? (q >= 0.27 ? 'success' : q >= 0.24 ? 'warning' : 'danger')
        : null;

    // Video-only fields — present on probed clips, absent on stills.
    const fpsVal = m['fps'];
    const fps = typeof fpsVal === 'number' && fpsVal > 0
        ? `${Number(fpsVal.toFixed(2))} fps` : null;
    const durVal = m['duration_s'];
    const duration = typeof durVal === 'number' && durVal > 0
        ? formatDuration(durVal) : null;
    const fcVal = m['frame_count'];
    const estimated = m['frame_count_estimated'] === true;
    const frameCount = typeof fcVal === 'number' && fcVal > 0
        ? `${estimated ? '~' : ''}${fcVal} frames` : null;

    return { res, ar, orientation, size, hpsLabel, hpsTone, fps, duration, frameCount };
}
