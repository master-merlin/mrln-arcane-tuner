import { describe, it, expect } from 'vitest';
import { Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { ModalEmptyComponent } from './modal-empty.component';

function root(fixture: { nativeElement: HTMLElement }): HTMLElement {
    return fixture.nativeElement.querySelector('[data-testid="modal-empty"]') as HTMLElement;
}

describe('ModalEmptyComponent', () => {
    it('renders the message text', () => {
        const fixture = TestBed.createComponent(ModalEmptyComponent);
        fixture.componentRef.setInput('message', 'Open a dataset workspace first.');
        fixture.detectChanges();
        expect(root(fixture).textContent).toContain('Open a dataset workspace first.');
    });

    it('renders a title when provided and omits it otherwise', () => {
        const fixture = TestBed.createComponent(ModalEmptyComponent);
        fixture.componentRef.setInput('message', 'msg');
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="modal-empty-title"]')).toBeNull();

        fixture.componentRef.setInput('title', 'Nothing here');
        fixture.detectChanges();
        const title = fixture.nativeElement.querySelector('[data-testid="modal-empty-title"]') as HTMLElement;
        expect(title).not.toBeNull();
        expect(title.textContent).toContain('Nothing here');
    });

    it('renders the icon glyph (app-ico → svg)', () => {
        const fixture = TestBed.createComponent(ModalEmptyComponent);
        fixture.componentRef.setInput('message', 'msg');
        fixture.componentRef.setInput('icon', 'Info');
        fixture.detectChanges();
        expect(root(fixture).querySelector('svg')).not.toBeNull();
    });

    it('is readable to assistive tech (role=note on the block)', () => {
        const fixture = TestBed.createComponent(ModalEmptyComponent);
        fixture.componentRef.setInput('message', 'msg');
        fixture.detectChanges();
        expect(root(fixture).getAttribute('role')).toBe('note');
        // The decorative icon must not be announced.
        const svg = root(fixture).querySelector('svg');
        expect(svg?.closest('[aria-hidden="true"]')).not.toBeNull();
    });

    it('projects a CTA into the block', () => {
        @Component({
            standalone: true,
            imports: [ModalEmptyComponent],
            template: `<app-modal-empty message="msg"><button data-testid="cta">Go</button></app-modal-empty>`,
        })
        class Host {}
        const fixture = TestBed.createComponent(Host);
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="cta"]')).not.toBeNull();
    });
});
