import { buildCanvasMeta, formatBytes } from '../media-meta';

describe('formatBytes', () => {
    it('formats bytes under 1 KB as raw bytes', () => {
        expect(formatBytes(0)).toBe('0 B');
        expect(formatBytes(512)).toBe('512 B');
    });

    it('formats KB / MB / GB with one decimal', () => {
        expect(formatBytes(1536)).toBe('1.5 KB');
        expect(formatBytes(2_500_000)).toBe('2.4 MB');
        expect(formatBytes(3_500_000_000)).toBe('3.3 GB');
    });
});

describe('buildCanvasMeta', () => {
    it('returns all-null fields when metadata is null', () => {
        expect(buildCanvasMeta(null)).toEqual({
            res: null, ar: null, orientation: null, size: null,
            hpsLabel: null, hpsTone: null,
            fps: null, duration: null, frameCount: null,
        });
    });

    it('treats undefined metadata identically to null', () => {
        // The signature accepts `Record<string, unknown> | null | undefined`
        // and both falsy branches route through the same `?? {}` fallback.
        expect(buildCanvasMeta(undefined)).toEqual({
            res: null, ar: null, orientation: null, size: null,
            hpsLabel: null, hpsTone: null,
            fps: null, duration: null, frameCount: null,
        });
    });

    it('builds res/ar from width+height', () => {
        const meta = buildCanvasMeta({ width: 1920, height: 1080 });
        expect(meta.res).toBe('1920×1080');
        expect(meta.ar).toBe('1.778');
    });

    it('prefers metadata.aspect_ratio when present', () => {
        const meta = buildCanvasMeta({ width: 1920, height: 1080, aspect_ratio: 1.5 });
        expect(meta.ar).toBe('1.500');
    });

    it('formats size_bytes via formatBytes', () => {
        expect(buildCanvasMeta({ size_bytes: 1024 }).size).toBe('1.0 KB');
    });

    it('builds HPS label + tone bucket from quality_score', () => {
        expect(buildCanvasMeta({ quality_score: 0.30 }).hpsLabel).toBe('HPS 0.3000');
        expect(buildCanvasMeta({ quality_score: 0.30 }).hpsTone).toBe('success');
        expect(buildCanvasMeta({ quality_score: 0.25 }).hpsTone).toBe('warning');
        expect(buildCanvasMeta({ quality_score: 0.20 }).hpsTone).toBe('danger');
    });

    it('discriminates the tone bucket at the exact threshold values', () => {
        // Boundaries are `q >= 0.27` (success) and `q >= 0.24` (warning).
        // Probe both sides of each so a future `>` vs `>=` typo or a
        // threshold drift would fail loudly.
        expect(buildCanvasMeta({ quality_score: 0.27 }).hpsTone).toBe('success');
        expect(buildCanvasMeta({ quality_score: 0.2699 }).hpsTone).toBe('warning');
        expect(buildCanvasMeta({ quality_score: 0.24 }).hpsTone).toBe('warning');
        expect(buildCanvasMeta({ quality_score: 0.2399 }).hpsTone).toBe('danger');
    });

    it('passes orientation through when it is a string', () => {
        expect(buildCanvasMeta({ orientation: 'landscape' }).orientation).toBe('landscape');
        expect(buildCanvasMeta({ orientation: 123 as unknown as string }).orientation).toBeNull();
    });

    it('builds video fps/duration/frame-count labels when present', () => {
        const meta = buildCanvasMeta({
            fps: 24, duration_s: 65, frame_count: 1560,
        });
        expect(meta.fps).toBe('24 fps');
        expect(meta.duration).toBe('1:05');
        expect(meta.frameCount).toBe('1560 frames');
    });

    it('marks an estimated frame count with a leading ~', () => {
        const meta = buildCanvasMeta({ frame_count: 120, frame_count_estimated: true });
        expect(meta.frameCount).toBe('~120 frames');
    });

    it('leaves video fields null for stills (no/zero video metadata)', () => {
        const meta = buildCanvasMeta({ width: 800, height: 600 });
        expect(meta.fps).toBeNull();
        expect(meta.duration).toBeNull();
        expect(meta.frameCount).toBeNull();
    });
});
