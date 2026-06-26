import { describe, it, expect } from 'vitest';
import { resolutionToMpx, resolutionMpxSpark } from './job-metrics';

const STEP_LOG_PREFIX = 'STEP_LOG:';

describe('resolutionToMpx', () => {
    it('parses an image WxH bucket to megapixels', () => {
        // 1888 * 1056 / 1e6 = 1.9937…
        expect(resolutionToMpx('1888x1056')).toBeCloseTo(1.994, 2);
        expect(resolutionToMpx('1248x1600')).toBeCloseTo(1.997, 2);
    });

    it('parses a video WxHxFf bucket (×frames)', () => {
        // 1888 * 1056 * 25 / 1e6
        expect(resolutionToMpx('1888x1056x25f')).toBeCloseTo(49.843, 2);
    });

    it('returns null for absent/unparseable input', () => {
        expect(resolutionToMpx(undefined)).toBeNull();
        expect(resolutionToMpx(null)).toBeNull();
        expect(resolutionToMpx('')).toBeNull();
        expect(resolutionToMpx('not-a-res')).toBeNull();
    });
});

describe('resolutionMpxSpark', () => {
    it('builds a per-step megapixel series from STEP_LOG lines', () => {
        const logs = [
            `${STEP_LOG_PREFIX}${JSON.stringify({ step: 1, status: 'training', resolution: '1248x1600' })}`,
            `${STEP_LOG_PREFIX}${JSON.stringify({ step: 2, status: 'training', resolution: '672x384' })}`,
            'some human log line without metrics',
        ];
        const series = resolutionMpxSpark(logs);
        expect(series).toHaveLength(2);
        expect(series[0]).toBeCloseTo(1.997, 2);
        expect(series[1]).toBeCloseTo(0.258, 2);
    });

    it('returns empty for undefined logs', () => {
        expect(resolutionMpxSpark(undefined)).toEqual([]);
    });
});
