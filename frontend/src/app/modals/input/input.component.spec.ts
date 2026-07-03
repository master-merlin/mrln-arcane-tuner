import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { InputModalComponent, type InputModalData } from './input.component';
import { OverlayStore } from '../../state/overlay.store';

function setup(over: Partial<InputModalData> = {}): {
    fixture: ComponentFixture<InputModalComponent>;
    overlay: OverlayStore;
    onConfirm: ReturnType<typeof vi.fn>;
    onCancel: ReturnType<typeof vi.fn>;
} {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();
    TestBed.configureTestingModule({ imports: [InputModalComponent] });
    const overlay = TestBed.inject(OverlayStore);
    overlay.openModal('input', {
        title: 'Save as Template',
        label: 'Template name',
        confirmLabel: 'Save',
        onConfirm,
        onCancel,
        ...over,
    });
    const fixture = TestBed.createComponent(InputModalComponent);
    fixture.detectChanges();
    return { fixture, overlay, onConfirm, onCancel };
}

function fields(fixture: ComponentFixture<InputModalComponent>) {
    const host = fixture.nativeElement as HTMLElement;
    return {
        input: host.querySelector('[data-testid="input-modal-field"]') as HTMLInputElement,
        buttons: host.querySelectorAll('.modal-foot button'),
    };
}

describe('InputModalComponent', () => {
    beforeEach(() => TestBed.resetTestingModule());

    it('confirm returns the entered (trimmed) value via onConfirm and closes', () => {
        const { fixture, overlay, onConfirm, onCancel } = setup();
        const { input, buttons } = fields(fixture);
        input.value = '  My Template  ';
        input.dispatchEvent(new Event('input'));
        fixture.detectChanges();
        (buttons[1] as HTMLButtonElement).click();
        fixture.destroy();
        expect(onConfirm).toHaveBeenCalledTimes(1);
        expect(onConfirm).toHaveBeenCalledWith('My Template');
        expect(onCancel).not.toHaveBeenCalled();
        expect(overlay.modalStack().length).toBe(0);
    });

    it('pre-fills from `initial` so a rename starts with the existing name', () => {
        const { fixture } = setup({ initial: 'Existing' });
        const { input } = fields(fixture);
        expect(input.value).toBe('Existing');
    });

    it('confirm is disabled for empty/whitespace input (no blank submit)', () => {
        const { fixture, onConfirm } = setup();
        const { input, buttons } = fields(fixture);
        input.value = '   ';
        input.dispatchEvent(new Event('input'));
        fixture.detectChanges();
        expect((buttons[1] as HTMLButtonElement).disabled).toBe(true);
        (buttons[1] as HTMLButtonElement).click();
        expect(onConfirm).not.toHaveBeenCalled();
    });

    it('cancel fires onCancel exactly once (destroy does not double-fire)', () => {
        const { fixture, overlay, onConfirm, onCancel } = setup();
        const { buttons } = fields(fixture);
        (buttons[0] as HTMLButtonElement).click();
        fixture.destroy();
        expect(onCancel).toHaveBeenCalledTimes(1);
        expect(onConfirm).not.toHaveBeenCalled();
        expect(overlay.modalStack().length).toBe(0);
    });

    it('dismissal without a choice (backdrop/Esc → closeModal + destroy) fires onCancel', () => {
        const { fixture, overlay, onConfirm, onCancel } = setup();
        overlay.closeModal();
        fixture.destroy();
        expect(onCancel).toHaveBeenCalledTimes(1);
        expect(onConfirm).not.toHaveBeenCalled();
    });

    it('being occluded by a child modal (entry still stacked) is NOT a dismissal', () => {
        const { fixture, onConfirm, onCancel } = setup();
        const overlay = TestBed.inject(OverlayStore);
        overlay.openModal('confirm', { title: 't', message: '' });
        fixture.destroy();
        expect(onCancel).not.toHaveBeenCalled();
        expect(onConfirm).not.toHaveBeenCalled();
    });
});
