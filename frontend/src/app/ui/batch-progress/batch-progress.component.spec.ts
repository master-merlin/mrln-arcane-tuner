import { describe, it, expect } from 'vitest';
import { Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { BatchProgressComponent } from './batch-progress.component';

function root(fixture: { nativeElement: HTMLElement }): HTMLElement {
    return fixture.nativeElement.querySelector('[data-testid="batch-progress"]') as HTMLElement;
}
function bar(fixture: { nativeElement: HTMLElement }): HTMLElement {
    return fixture.nativeElement.querySelector('[data-testid="batch-progress-bar"]') as HTMLElement;
}

describe('BatchProgressComponent', () => {
    it('uses the explicit percent input for the readout and the fill width', () => {
        const fixture = TestBed.createComponent(BatchProgressComponent);
        fixture.componentRef.setInput('label', 'NEURAL PROCESSING');
        fixture.componentRef.setInput('current', 3);
        fixture.componentRef.setInput('total', 10);
        fixture.componentRef.setInput('percent', 30);
        fixture.detectChanges();
        expect(root(fixture).textContent).toContain('30%');
        const fill = fixture.nativeElement.querySelector('[data-testid="batch-progress-fill"]') as HTMLElement;
        expect(fill.style.width).toBe('30%');
    });

    it('derives percent from current/total when percent is not supplied', () => {
        const fixture = TestBed.createComponent(BatchProgressComponent);
        fixture.componentRef.setInput('label', 'X');
        fixture.componentRef.setInput('current', 1);
        fixture.componentRef.setInput('total', 4);
        fixture.detectChanges();
        expect(root(fixture).textContent).toContain('25%');
    });

    it('guards a zero total (0%, no divide-by-zero)', () => {
        const fixture = TestBed.createComponent(BatchProgressComponent);
        fixture.componentRef.setInput('label', 'X');
        fixture.componentRef.setInput('current', 0);
        fixture.componentRef.setInput('total', 0);
        fixture.detectChanges();
        expect(root(fixture).textContent).toContain('0%');
    });

    it('exposes progressbar semantics with aria-valuenow/min/max', () => {
        const fixture = TestBed.createComponent(BatchProgressComponent);
        fixture.componentRef.setInput('label', 'NEURAL PROCESSING');
        fixture.componentRef.setInput('current', 5);
        fixture.componentRef.setInput('total', 10);
        fixture.componentRef.setInput('percent', 50);
        fixture.detectChanges();
        const b = bar(fixture);
        expect(b.getAttribute('role')).toBe('progressbar');
        expect(b.getAttribute('aria-valuenow')).toBe('50');
        expect(b.getAttribute('aria-valuemin')).toBe('0');
        expect(b.getAttribute('aria-valuemax')).toBe('100');
        expect(b.getAttribute('aria-label')).toContain('NEURAL PROCESSING');
    });

    it('renders the label, queue count, and current item', () => {
        const fixture = TestBed.createComponent(BatchProgressComponent);
        fixture.componentRef.setInput('label', 'PIPELINE RENDERING');
        fixture.componentRef.setInput('current', 2);
        fixture.componentRef.setInput('total', 8);
        fixture.componentRef.setInput('currentItem', 'frame_007.png');
        fixture.detectChanges();
        const text = root(fixture).textContent ?? '';
        expect(text).toContain('PIPELINE RENDERING');
        expect(text).toContain('2 / 8');
        expect(text).toContain('frame_007.png');
    });

    it('defaults the queue/current labels but allows overrides', () => {
        const fixture = TestBed.createComponent(BatchProgressComponent);
        fixture.componentRef.setInput('label', 'X');
        fixture.componentRef.setInput('current', 1);
        fixture.componentRef.setInput('total', 2);
        fixture.detectChanges();
        expect(root(fixture).textContent).toContain('QUEUE STATUS');
        expect(root(fixture).textContent).toContain('CURRENT FRAME');

        fixture.componentRef.setInput('queueLabel', 'QUEUE');
        fixture.componentRef.setInput('currentLabel', 'CURRENT');
        fixture.detectChanges();
        expect(root(fixture).textContent).toContain('QUEUE');
        expect(root(fixture).textContent).toContain('CURRENT');
    });

    it('renders an optional hint only when provided', () => {
        const fixture = TestBed.createComponent(BatchProgressComponent);
        fixture.componentRef.setInput('label', 'X');
        fixture.componentRef.setInput('current', 1);
        fixture.componentRef.setInput('total', 2);
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="batch-progress-hint"]')).toBeNull();

        fixture.componentRef.setInput('hint', 'Runs in the background.');
        fixture.detectChanges();
        const hint = fixture.nativeElement.querySelector('[data-testid="batch-progress-hint"]') as HTMLElement;
        expect(hint.textContent).toContain('Runs in the background.');
    });

    it('drives the accent through the --batch-accent custom property', () => {
        const fixture = TestBed.createComponent(BatchProgressComponent);
        fixture.componentRef.setInput('label', 'X');
        fixture.componentRef.setInput('current', 1);
        fixture.componentRef.setInput('total', 2);
        fixture.componentRef.setInput('accent', 'var(--color-violet)');
        fixture.detectChanges();
        // The host element carries the custom property so the scoped fill reads it.
        const host = fixture.nativeElement as HTMLElement;
        expect(host.style.getPropertyValue('--batch-accent')).toContain('--color-violet');
    });

    it('projects footer actions (e.g. a Stop button) into the panel', () => {
        @Component({
            standalone: true,
            imports: [BatchProgressComponent],
            template: `<app-batch-progress label="X" [current]="1" [total]="2">
                <button data-testid="stop">Stop</button>
            </app-batch-progress>`,
        })
        class Host {}
        const fixture = TestBed.createComponent(Host);
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="stop"]')).not.toBeNull();
    });
});
