// cooccurrence-heatmap.spec.ts
import { cooccurrenceCells } from './cooccurrence-heatmap';

describe('cooccurrenceCells', () => {
    it('produces one cell per matrix entry with normalized intensity', () => {
        const cells = cooccurrenceCells(['a', 'b'], [[4, 2], [2, 3]]);
        expect(cells.length).toBe(4);
        const max = Math.max(...cells.map(c => c.value));
        expect(max).toBe(4);
        const top = cells.find(c => c.row === 0 && c.col === 0)!;
        expect(top.intensity).toBe(1); // 4/4
        const mid = cells.find(c => c.row === 0 && c.col === 1)!;
        expect(mid.intensity).toBeCloseTo(0.5); // 2/4
    });

    it('handles an all-zero matrix without dividing by zero', () => {
        const cells = cooccurrenceCells(['a'], [[0]]);
        expect(cells[0].intensity).toBe(0);
    });

    it('returns no cells for empty labels', () => {
        expect(cooccurrenceCells([], [])).toEqual([]);
    });
});
