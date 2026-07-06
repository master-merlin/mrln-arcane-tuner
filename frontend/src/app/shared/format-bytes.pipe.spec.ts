import { describe, it, expect } from 'vitest';
import { FormatBytesPipe } from './format-bytes.pipe';

describe('FormatBytesPipe', () => {
    const pipe = new FormatBytesPipe();

    it('delegates to formatBytes() — "1.5 GB" for 1.5 * 2^30', () => {
        expect(pipe.transform(1.5 * 2 ** 30)).toBe('1.5 GB');
    });

    it('delegates to formatBytes() — "0 B" for non-positive input', () => {
        expect(pipe.transform(0)).toBe('0 B');
    });
});
