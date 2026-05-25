import { linearTicks, mapY, ema } from '../loss-chart-geometry';

describe('linearTicks', () => {
    it('generates 5 evenly spaced ticks between min and max', () => {
        const ticks = linearTicks(0, 1, 5);
        expect(ticks).toEqual([0, 0.25, 0.5, 0.75, 1]);
    });

    it('handles min > 0', () => {
        const ticks = linearTicks(0.2, 0.6, 5);
        expect(ticks[0]).toBeCloseTo(0.2);
        expect(ticks[4]).toBeCloseTo(0.6);
    });
});

describe('mapY', () => {
    it('maps value to inverted y coordinate in pixel space', () => {
        // chart height 100, value 0 is at bottom (y=100), value 1 is at top (y=0)
        expect(mapY(0, 0, 1, 100)).toBe(100);
        expect(mapY(1, 0, 1, 100)).toBe(0);
        expect(mapY(0.5, 0, 1, 100)).toBe(50);
    });
});

describe('ema', () => {
    it('returns input unchanged when alpha=1', () => {
        const data = [1, 2, 3, 4];
        expect(ema(data, 1)).toEqual(data);
    });

    it('smooths input when alpha<1', () => {
        const out = ema([0, 1, 1, 1], 0.5);
        expect(out[0]).toBe(0);
        expect(out[1]).toBeCloseTo(0.5);
        expect(out[2]).toBeCloseTo(0.75);
        expect(out[3]).toBeCloseTo(0.875);
    });

    it('handles empty input', () => {
        expect(ema([], 0.5)).toEqual([]);
    });
});
