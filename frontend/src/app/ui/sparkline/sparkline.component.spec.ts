import { describe, it, expect } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { SparklineComponent } from './sparkline.component';

function paths(fixture: { nativeElement: HTMLElement }): { line: string | null; area: string | null } {
    const svg = fixture.nativeElement.querySelector('svg')!;
    const ps = svg.querySelectorAll('path');
    // Area path (if present) is rendered first, line path second — matches
    // template source order (`@if (areaPath())` before `@if (linePath())`).
    if (ps.length === 2) return { area: ps[0].getAttribute('d'), line: ps[1].getAttribute('d') };
    if (ps.length === 1) return { area: null, line: ps[0].getAttribute('d') };
    return { area: null, line: null };
}

describe('SparklineComponent path building', () => {
    it('renders no path for an empty series', () => {
        const fixture = TestBed.createComponent(SparklineComponent);
        fixture.componentRef.setInput('data', []);
        fixture.detectChanges();
        expect(paths(fixture).line).toBeNull();
        expect(paths(fixture).area).toBeNull();
    });

    it('renders no path for a single-point series (needs >= 2 points)', () => {
        const fixture = TestBed.createComponent(SparklineComponent);
        fixture.componentRef.setInput('data', [5]);
        fixture.detectChanges();
        expect(paths(fixture).line).toBeNull();
    });

    it('builds a two-point line path spanning the full 0-100 x-range', () => {
        const fixture = TestBed.createComponent(SparklineComponent);
        fixture.componentRef.setInput('data', [0, 10]);
        fixture.componentRef.setInput('height', 24);
        fixture.detectChanges();
        // height=24, pad=2 -> usable=20. First point (min) -> y = pad + 1*20 = 22.
        // Second point (max) -> y = pad + 0*20 = 2. x steps 0 -> 100.
        expect(paths(fixture).line).toBe('M 0.00 22.00 L 100.00 2.00');
    });

    it('flat series (all equal values) centers the line without dividing by zero', () => {
        const fixture = TestBed.createComponent(SparklineComponent);
        fixture.componentRef.setInput('data', [7, 7, 7]);
        fixture.componentRef.setInput('height', 24);
        fixture.detectChanges();
        // span falls back to 1 when hi === lo, so v - lo === 0 for every point
        // -> y = pad + 1*usable = 22 for all three (flat line at the bottom-most
        // padded row, since (1 - 0/1) = 1).
        expect(paths(fixture).line).toBe('M 0.00 22.00 L 50.00 22.00 L 100.00 22.00');
    });

    it('closes the area path back down to the baseline and to the first x', () => {
        const fixture = TestBed.createComponent(SparklineComponent);
        fixture.componentRef.setInput('data', [0, 10]);
        fixture.componentRef.setInput('height', 24);
        fixture.componentRef.setInput('area', true);
        fixture.detectChanges();
        const { area } = paths(fixture);
        expect(area).toBe('M 0.00 22.00 L 100.00 2.00 L 100.00 24 L 0.00 24 Z');
    });

    it('omits the area path when area=false', () => {
        const fixture = TestBed.createComponent(SparklineComponent);
        fixture.componentRef.setInput('data', [0, 10]);
        fixture.componentRef.setInput('area', false);
        fixture.detectChanges();
        expect(paths(fixture).area).toBeNull();
        expect(paths(fixture).line).not.toBeNull();
    });

    it('three-point rising series places the midpoint at the correct x/y', () => {
        const fixture = TestBed.createComponent(SparklineComponent);
        fixture.componentRef.setInput('data', [0, 5, 10]);
        fixture.componentRef.setInput('height', 24);
        fixture.detectChanges();
        // stepX = 100/2 = 50; midpoint value 5 is exactly mid-span -> y = pad + 0.5*usable = 12.
        expect(paths(fixture).line).toBe('M 0.00 22.00 L 50.00 12.00 L 100.00 2.00');
    });
});
