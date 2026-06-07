import type { MockInstance } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { FilmstripScrubberComponent } from '../filmstrip-scrubber.component';
import { OverlayStore } from '../../../state/overlay.store';

class StubOverlay {
    modalStack = signal<any[]>([]);
}

describe('FilmstripScrubberComponent — arrow nav suppression', () => {
    let cmp: FilmstripScrubberComponent;
    let overlay: StubOverlay;
    let seekSpy: MockInstance;

    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [
                FilmstripScrubberComponent,
                { provide: OverlayStore, useClass: StubOverlay },
            ],
        });
        cmp = TestBed.inject(FilmstripScrubberComponent);
        overlay = TestBed.inject(OverlayStore) as any;
        // Drive the inputs the handler reads.
        (cmp as any).images = () => [
            { media_file: 'a' }, { media_file: 'b' }, { media_file: 'c' },
        ];
        (cmp as any).activeIndex = () => 1;
        seekSpy = vi.spyOn((cmp as any).seek, 'emit');
    });

    it('emits seek on ArrowRight when no modal is open', () => {
        (cmp as any).onKey(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
        expect(seekSpy).toHaveBeenCalledWith(2);
    });

    it('emits seek on ArrowLeft when no modal is open', () => {
        (cmp as any).onKey(new KeyboardEvent('keydown', { key: 'ArrowLeft' }));
        expect(seekSpy).toHaveBeenCalledWith(0);
    });

    it('does NOT emit seek when a modal is on the stack (ArrowRight)', () => {
        overlay.modalStack.set([{ kind: 'crop-preview', data: {} }]);
        (cmp as any).onKey(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
        expect(seekSpy).not.toHaveBeenCalled();
    });

    it('does NOT emit seek when a modal is on the stack (ArrowLeft)', () => {
        overlay.modalStack.set([{ kind: 'mask-preview', data: {} }]);
        (cmp as any).onKey(new KeyboardEvent('keydown', { key: 'ArrowLeft' }));
        expect(seekSpy).not.toHaveBeenCalled();
    });

    it('still skips when an input is focused (regression guard)', () => {
        const input = document.createElement('input');
        document.body.appendChild(input);
        input.focus();
        const ev = new KeyboardEvent('keydown', { key: 'ArrowRight' });
        Object.defineProperty(ev, 'target', { value: input });
        (cmp as any).onKey(ev);
        expect(seekSpy).not.toHaveBeenCalled();
        input.remove();
    });
});
