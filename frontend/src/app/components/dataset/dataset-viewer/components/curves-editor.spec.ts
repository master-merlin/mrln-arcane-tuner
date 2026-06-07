import { TestBed } from '@angular/core/testing';
import { CurvesEditorComponent } from './curves-editor';

/**
 * Preset dropdown is a momentary action menu: picking a preset applies it and
 * the control snaps back to the "Preset…" placeholder. Regression guard for the
 * stuck-dropdown bug — a one-way [ngModel] bound to '' never wrote back after a
 * manual pick ('' → '' is a no-change), so the select stayed showing the preset
 * even after Reset Channel.
 */
describe('CurvesEditorComponent — preset dropdown', () => {
    function setup() {
        const fixture = TestBed.createComponent(CurvesEditorComponent);
        fixture.detectChanges();
        const emitted: { channel: string; points: { x: number; y: number }[] }[] = [];
        fixture.componentInstance.curveChanged.subscribe(e => emitted.push(e));
        const select: HTMLSelectElement =
            fixture.nativeElement.querySelector('[data-testid="curves-preset-select"]');
        return { fixture, emitted, select };
    }

    it('applies the picked preset and snaps the <select> back to the placeholder', () => {
        const { fixture, emitted, select } = setup();

        select.value = 'S-Curve';
        select.dispatchEvent(new Event('change'));
        fixture.detectChanges();

        expect(emitted.length).toBe(1);
        expect(emitted[0].points).toEqual([
            { x: 0, y: 0 }, { x: 64, y: 48 }, { x: 192, y: 208 }, { x: 255, y: 255 },
        ]);
        // The core regression: the control must NOT stay on "S-Curve".
        expect(select.value).toBe('');
    });

    it('Reset Channel emits the identity curve for the active channel', () => {
        const { fixture, emitted } = setup();
        (fixture.componentInstance as any).resetChannel();
        expect(emitted.length).toBe(1);
        expect(emitted[0].channel).toBe('master');
        expect(emitted[0].points).toEqual([{ x: 0, y: 0 }, { x: 255, y: 255 }]);
    });

    it('picking the placeholder option emits nothing', () => {
        const { fixture, emitted, select } = setup();
        select.value = '';
        select.dispatchEvent(new Event('change'));
        fixture.detectChanges();
        expect(emitted.length).toBe(0);
    });
});
