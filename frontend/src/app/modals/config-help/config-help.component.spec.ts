import { describe, it, expect, beforeEach } from 'vitest';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { ConfigHelpModalComponent, type ConfigHelpData } from './config-help.component';
import { OverlayStore } from '../../state/overlay.store';

function setup(over: Partial<ConfigHelpData> = {}): {
    fixture: ComponentFixture<ConfigHelpModalComponent>;
    overlay: OverlayStore;
} {
    const data: ConfigHelpData = {
        title: 'Learning Rate',
        tip: 'How fast the model learns.',
        detailHtml: '<p>Detail with <strong>bold</strong> and <code>code</code>.</p>',
        ...over,
    };
    TestBed.configureTestingModule({ imports: [ConfigHelpModalComponent] });
    const overlay = TestBed.inject(OverlayStore);
    overlay.openModal('config-help', data);
    const fixture = TestBed.createComponent(ConfigHelpModalComponent);
    fixture.detectChanges();
    return { fixture, overlay };
}

describe('ConfigHelpModalComponent', () => {
    beforeEach(() => TestBed.resetTestingModule());

    it('renders the title, tip and detail HTML from the overlay payload', () => {
        const { fixture } = setup();
        const el = fixture.nativeElement as HTMLElement;
        expect(el.querySelector('[data-testid="config-help-title"]')?.textContent).toContain('Learning Rate');
        expect(el.textContent).toContain('How fast the model learns.');
        const detail = el.querySelector('[data-testid="config-help-detail"]') as HTMLElement;
        // innerHTML markup is preserved (bold/code survive Angular's sanitizer).
        expect(detail.querySelector('strong')?.textContent).toBe('bold');
        expect(detail.querySelector('code')?.textContent).toBe('code');
    });

    it('closes the modal when Got it is clicked', () => {
        const { fixture, overlay } = setup();
        const btn = (fixture.nativeElement as HTMLElement)
            .querySelector('[data-testid="config-help-got-it"]') as HTMLButtonElement;
        btn.click();
        expect(overlay.modalStack().length).toBe(0);
    });

    it('closes the modal via the header close button', () => {
        const { fixture, overlay } = setup();
        const btn = (fixture.nativeElement as HTMLElement)
            .querySelector('[data-testid="config-help-close"]') as HTMLButtonElement;
        btn.click();
        expect(overlay.modalStack().length).toBe(0);
    });
});
