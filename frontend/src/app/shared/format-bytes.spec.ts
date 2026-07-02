import { describe, it, expect } from 'vitest';
import { formatBytes } from './format-bytes';

// Expected strings pinned from the pre-dedup private implementations that
// lived in datasets-screen.ts and jobs-screen.ts (character-identical):
//
//   protected formatBytes(n: number): string {
//       if (!Number.isFinite(n) || n <= 0) return '0 B';
//       const units = ['B', 'KB', 'MB', 'GB', 'TB'];
//       let i = 0;
//       while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
//       return `${n.toFixed(i > 0 ? 1 : 0)} ${units[i]}`;
//   }
describe('formatBytes', () => {
    it('formats 0 (and non-positive/non-finite) as "0 B"', () => {
        expect(formatBytes(0)).toBe('0 B');
        expect(formatBytes(-5)).toBe('0 B');
        expect(formatBytes(NaN)).toBe('0 B');
        expect(formatBytes(Infinity)).toBe('0 B');
    });

    it('formats sub-KB integers as whole bytes', () => {
        expect(formatBytes(1023)).toBe('1023 B');
    });

    it('formats exactly 1024 as 1.0 KB', () => {
        expect(formatBytes(1024)).toBe('1.0 KB');
    });

    it('formats 1536 as 1.5 KB', () => {
        expect(formatBytes(1536)).toBe('1.5 KB');
    });

    it('formats 10^6 as 976.6 KB', () => {
        expect(formatBytes(1_000_000)).toBe('976.6 KB');
    });

    it('formats 1.5 * 2^30 as 1.5 GB', () => {
        expect(formatBytes(1.5 * 2 ** 30)).toBe('1.5 GB');
    });
});
