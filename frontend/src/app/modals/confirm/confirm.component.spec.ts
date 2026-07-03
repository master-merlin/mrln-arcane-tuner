import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { ConfirmModalComponent, type ConfirmModalData } from './confirm.component';
import { OverlayStore } from '../../state/overlay.store';

function setup(over: Partial<ConfirmModalData> = {}): {
    fixture: ComponentFixture<ConfirmModalComponent>;
    overlay: OverlayStore;
    onConfirm: ReturnType<typeof vi.fn>;
    onCancel: ReturnType<typeof vi.fn>;
} {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();
    TestBed.configureTestingModule({ imports: [ConfirmModalComponent] });
    const overlay = TestBed.inject(OverlayStore);
    overlay.openModal('confirm', {
        title: 'Model Changed',
        message: 'Keep or switch?',
        cancelLabel: 'Keep',
        confirmLabel: 'Switch & Reset',
        onConfirm,
        onCancel,
        ...over,
    });
    const fixture = TestBed.createComponent(ConfirmModalComponent);
    fixture.detectChanges();
    return { fixture, overlay, onConfirm, onCancel };
}

describe('ConfirmModalComponent', () => {
    beforeEach(() => TestBed.resetTestingModule());

    it('confirm button fires onConfirm (not onCancel) and closes', () => {
        const { fixture, overlay, onConfirm, onCancel } = setup();
        const buttons = (fixture.nativeElement as HTMLElement).querySelectorAll('.modal-foot button');
        (buttons[1] as HTMLButtonElement).click();
        // Destroy afterwards (as modal-layer does once the entry pops) — the
        // dismissal fallback must NOT double-resolve.
        fixture.destroy();
        expect(onConfirm).toHaveBeenCalledTimes(1);
        expect(onCancel).not.toHaveBeenCalled();
        expect(overlay.modalStack().length).toBe(0);
    });

    it('cancel button fires onCancel exactly once (destroy does not double-fire)', () => {
        const { fixture, overlay, onConfirm, onCancel } = setup();
        const buttons = (fixture.nativeElement as HTMLElement).querySelectorAll('.modal-foot button');
        (buttons[0] as HTMLButtonElement).click();
        fixture.destroy();
        expect(onCancel).toHaveBeenCalledTimes(1);
        expect(onConfirm).not.toHaveBeenCalled();
        expect(overlay.modalStack().length).toBe(0);
    });

    it('dismissal without a choice (backdrop/Esc → closeModal + destroy) fires onCancel', () => {
        const { fixture, overlay, onConfirm, onCancel } = setup();
        // modal-layer's backdrop click and the global Esc shortcut both call
        // closeModal() directly; the `@if (last)` then destroys this component.
        overlay.closeModal();
        fixture.destroy();
        expect(onCancel).toHaveBeenCalledTimes(1);
        expect(onConfirm).not.toHaveBeenCalled();
    });

    it('being occluded by a child modal (entry still stacked) is NOT a dismissal', () => {
        const { fixture, onConfirm, onCancel } = setup();
        // A modal pushed on top destroys the confirm component (modal-layer
        // renders only the last entry) while its own entry stays on the stack.
        const overlay = TestBed.inject(OverlayStore);
        overlay.openModal('config-help', { title: 't', tip: '', detailHtml: '' });
        fixture.destroy();
        expect(onCancel).not.toHaveBeenCalled();
        expect(onConfirm).not.toHaveBeenCalled();
    });
});
