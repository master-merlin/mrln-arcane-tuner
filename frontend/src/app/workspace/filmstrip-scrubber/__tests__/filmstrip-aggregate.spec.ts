import { aggregateFilmstrip, type StripCell } from '../filmstrip-aggregate';

interface Img { harmonized: boolean; captioned: boolean; masked: boolean; }

const mk = (n: number, all = true): Img[] =>
    Array.from({ length: n }, () => ({ harmonized: all, captioned: all, masked: all }));

describe('aggregateFilmstrip', () => {
    it('returns one cell per image when count ≤ threshold', () => {
        const cells = aggregateFilmstrip(mk(50), 140);
        expect(cells.length).toBe(50);
        for (const c of cells) expect(c.count).toBe(1);
    });

    it('returns ≤ threshold cells when count > threshold', () => {
        const cells = aggregateFilmstrip(mk(2000), 140);
        expect(cells.length).toBeLessThanOrEqual(140);
    });

    it('each cell covers a contiguous range', () => {
        const cells = aggregateFilmstrip(mk(2000), 140);
        let cursor = 0;
        for (const c of cells) {
            expect(c.startIndex).toBe(cursor);
            cursor += c.count;
        }
        expect(cursor).toBe(2000);
    });

    it('marks cell readiness based on whether ALL images in range are ready', () => {
        const imgs = [
            ...mk(5, true),
            ...mk(5, false),
        ];
        const cells = aggregateFilmstrip(imgs, 2);  // tiny threshold for force-aggregation
        expect(cells.length).toBe(2);
        expect(cells[0].state.harmonized).toBe(true);
        expect(cells[1].state.harmonized).toBe(false);
    });

    it('handles count of 0', () => {
        expect(aggregateFilmstrip([], 140)).toEqual([]);
    });

    it('handles count of 1', () => {
        const cells = aggregateFilmstrip(mk(1), 140);
        expect(cells).toEqual([{
            startIndex: 0,
            count: 1,
            state: { harmonized: true, captioned: true, masked: true },
        } as StripCell]);
    });
});
