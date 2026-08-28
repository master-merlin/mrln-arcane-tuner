import { TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { FilmstripScrubberComponent } from '../filmstrip-scrubber.component';
import { OverlayStore } from '../../../state/overlay.store';

/**
 * UAT-3.1 — the filmstrip printed a readiness legend that its thumbnails did
 * not answer.
 *
 * `.cell.thumb` sets its own background and `opacity: 1`, deliberately
 * overriding the colour-bar treatment; once every cell renders a thumbnail
 * (the normal case for a dataset under the aggregation threshold) the legend
 * explained three colours that were nowhere on screen. The fix repeats the
 * readiness colour as a bar along the bottom edge of each thumbnail.
 *
 * These are rendered-DOM assertions on purpose: the defect was invisible to
 * every headless spec because the state classes were always correct — it was
 * only ever a question of what covered them.
 */

class StubOverlay {
    modalStack = signal<any[]>([]);
}

@Component({
    standalone: true,
    imports: [FilmstripScrubberComponent],
    template: `<app-filmstrip-scrubber [images]="images()" [activeIndex]="0"/>`,
})
class Host {
    images = signal<any[]>([]);
}

function mount(images: any[]) {
    TestBed.configureTestingModule({
        imports: [Host],
        providers: [{ provide: OverlayStore, useClass: StubOverlay }],
    });
    const f = TestBed.createComponent(Host);
    f.componentInstance.images.set(images);
    f.detectChanges();
    return f;
}

const thumb = (over: Record<string, unknown> = {}) => ({
    thumbnailUrl: 'blob:x',
    mediaType: 'image',
    ...over,
});

afterEach(() => TestBed.resetTestingModule());

describe('filmstrip thumbnails carry the legend colour', () => {
    it('renders a status bar inside every thumbnail cell', () => {
        const f = mount([thumb(), thumb(), thumb()]);
        const cells = f.nativeElement.querySelectorAll('.cell.thumb');
        expect(cells.length).toBe(3);
        for (const c of cells) {
            expect(c.querySelector('.status-bar')).toBeTruthy();
        }
    });

    it('gives the bar the SAME state classes the legend explains', () => {
        const f = mount([
            thumb({ harmonized: true, captioned: true, masked: true }), // masked
            thumb({ harmonized: true, captioned: true }),               // captioned only
            thumb({ harmonized: true }),                                // missing
        ]);
        const cells = [...f.nativeElement.querySelectorAll('.cell.thumb')] as HTMLElement[];

        // The bar takes its colour from the cell's classes, so the contract is
        // that the three legend states stay distinguishable on the cell.
        expect(cells[0].classList.contains('c')).toBe(true);
        expect(cells[0].classList.contains('m')).toBe(true);
        expect(cells[1].classList.contains('c')).toBe(true);
        expect(cells[1].classList.contains('m')).toBe(false);
        expect(cells[2].classList.contains('c')).toBe(false);
        expect(cells[2].classList.contains('m')).toBe(false);
        for (const c of cells) expect(c.querySelector('.status-bar')).toBeTruthy();
    });

    it('does not add a bar to aggregated cells, which show the colour themselves', () => {
        // No thumbnailUrl -> the cell paints its own background; a bar there
        // would be a second, redundant indicator on the same surface.
        const f = mount(Array.from({ length: 4 }, () => ({ captioned: true })));
        const cells = f.nativeElement.querySelectorAll('.cell');
        expect(cells.length).toBeGreaterThan(0);
        for (const c of cells) {
            expect(c.classList.contains('thumb')).toBe(false);
            expect(c.querySelector('.status-bar')).toBeFalsy();
        }
    });

    it('keeps the bar when the thumbnail fails to load', () => {
        // The error glyph replaces the <img>, not the readiness indicator:
        // a broken thumbnail is still a file with a caption/mask state.
        const f = mount([thumb({ captioned: true })]);
        const cmp = f.debugElement.children[0].componentInstance as FilmstripScrubberComponent;
        (cmp as any).onImgError(0);
        f.detectChanges();
        const cell = f.nativeElement.querySelector('.cell.thumb') as HTMLElement;
        expect(cell.querySelector('.thumb-error')).toBeTruthy();
        expect(cell.querySelector('.status-bar')).toBeTruthy();
    });
});
