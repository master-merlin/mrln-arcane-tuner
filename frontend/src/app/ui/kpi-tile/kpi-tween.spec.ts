import { describe, it, expect } from 'vitest';
import { shouldTween, tweenStep } from './kpi-tween';

describe('tweenStep', () => {
    it('returns the start value at t=0', () => {
        expect(tweenStep(100, 200, 0)).toBe(100);
    });

    it('returns the target value at t=1', () => {
        expect(tweenStep(100, 200, 1)).toBe(200);
    });

    it('eases out — past the linear midpoint at t=0.5', () => {
        // easeOutCubic(0.5) ≈ 0.875, so the value is well past halfway.
        const mid = tweenStep(0, 100, 0.5);
        expect(mid).toBeGreaterThan(50);
        expect(mid).toBeLessThan(100);
    });

    it('returns an integer (counter never shows fractional steps)', () => {
        expect(Number.isInteger(tweenStep(0, 7, 0.3))).toBe(true);
    });

    it('clamps t outside [0,1]', () => {
        expect(tweenStep(0, 100, -1)).toBe(0);
        expect(tweenStep(0, 100, 5)).toBe(100);
    });
});

describe('shouldTween', () => {
    it('tweens a small forward step (normal live progress)', () => {
        expect(shouldTween(740, 749)).toBe(true);
    });

    it('snaps on a decrease (e.g. switching to another job)', () => {
        expect(shouldTween(749, 12)).toBe(false);
    });

    it('snaps when there is no movement', () => {
        expect(shouldTween(500, 500)).toBe(false);
    });

    it('snaps a huge forward jump (initial load of an in-progress run)', () => {
        expect(shouldTween(0, 5000)).toBe(false);
    });
});
