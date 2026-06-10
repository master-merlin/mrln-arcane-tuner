// caption-text.utils.spec.ts
import { dedupeTags, normalizeCommaSpacing } from './caption-text.utils';

describe('normalizeCommaSpacing', () => {
    it('collapses whitespace and enforces ", " separators', () => {
        expect(normalizeCommaSpacing('a ,b,   c')).toBe('a, b, c');
    });
    it('trims leading/trailing separators and whitespace', () => {
        expect(normalizeCommaSpacing('  , a, b ,  ')).toBe('a, b');
    });
    it('collapses internal runs of whitespace within a tag', () => {
        expect(normalizeCommaSpacing('red   car, blue   sky')).toBe('red car, blue sky');
    });
    it('returns empty string for empty/separators-only input', () => {
        expect(normalizeCommaSpacing('   ,, ,  ')).toBe('');
    });
});

describe('dedupeTags', () => {
    it('removes case-insensitive duplicate tags, preserving first occurrence + its casing', () => {
        expect(dedupeTags('Cat, dog, CAT, Dog, bird')).toBe('Cat, dog, bird');
    });
    it('normalizes spacing as part of dedupe', () => {
        expect(dedupeTags('a ,a,  b')).toBe('a, b');
    });
    it('is a no-op for already-clean unique tags', () => {
        expect(dedupeTags('a, b, c')).toBe('a, b, c');
    });
    it('handles empty input', () => {
        expect(dedupeTags('')).toBe('');
    });
});
