import { describe, it, expect } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { KpiTileComponent } from './kpi-tile.component';

function valueText(fixture: { nativeElement: HTMLElement }): string {
    return (
        fixture.nativeElement
            .querySelector('[data-testid="kpi-tile-value"]')
            ?.textContent ?? ''
    ).trim();
}

describe('KpiTileComponent value rendering', () => {
    it('renders a numeric value verbatim when animate is off (default)', () => {
        const fixture = TestBed.createComponent(KpiTileComponent);
        fixture.componentRef.setInput('label', 'Step');
        fixture.componentRef.setInput('value', 742);
        fixture.detectChanges();
        expect(valueText(fixture)).toBe('742');
    });

    it('renders a string value verbatim and never animates it', () => {
        const fixture = TestBed.createComponent(KpiTileComponent);
        fixture.componentRef.setInput('label', 'Loss');
        fixture.componentRef.setInput('value', '0.1234');
        fixture.componentRef.setInput('animate', true);
        fixture.detectChanges();
        // Non-numeric values fall straight through — no count-up applies.
        expect(valueText(fixture)).toBe('0.1234');
    });

    it('shows the real value immediately when animate is on (first render snaps)', () => {
        const fixture = TestBed.createComponent(KpiTileComponent);
        fixture.componentRef.setInput('label', 'Step');
        fixture.componentRef.setInput('value', 500);
        fixture.componentRef.setInput('animate', true);
        fixture.detectChanges();
        // No flash to 0 / blank on first paint — the initial value is honoured.
        expect(valueText(fixture)).toBe('500');
    });

    it('animates a forward update and converges on the new value', () => {
        // Drive requestAnimationFrame + the clock deterministically so the
        // count-up runs to completion without relying on real-timer rAF (which
        // does not auto-fire in the headless test environment).
        const rafCbs: FrameRequestCallback[] = [];
        const origRaf = globalThis.requestAnimationFrame;
        const origCancel = globalThis.cancelAnimationFrame;
        const origNow = performance.now.bind(performance);
        let clock = 1000;
        globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) =>
            rafCbs.push(cb)) as typeof globalThis.requestAnimationFrame;
        globalThis.cancelAnimationFrame = (() => {}) as typeof globalThis.cancelAnimationFrame;
        (performance as unknown as { now: () => number }).now = () => clock;
        try {
            const fixture = TestBed.createComponent(KpiTileComponent);
            fixture.componentRef.setInput('label', 'Step');
            fixture.componentRef.setInput('value', 100);
            fixture.componentRef.setInput('animate', true);
            fixture.detectChanges();
            expect(valueText(fixture)).toBe('100');

            // A small forward step kicks off the count-up (100 → 105).
            fixture.componentRef.setInput('value', 105);
            fixture.detectChanges();

            // Advance past the 400ms tween duration and flush queued frames.
            clock = 2000;
            let guard = 0;
            while (rafCbs.length && guard++ < 100) {
                rafCbs.shift()!(clock);
            }
            fixture.detectChanges();
            // Lands exactly on the target — the rAF loop ran without error.
            expect(valueText(fixture)).toBe('105');
        } finally {
            globalThis.requestAnimationFrame = origRaf;
            globalThis.cancelAnimationFrame = origCancel;
            (performance as unknown as { now: () => number }).now = origNow;
        }
    });
});
