/**
 * bbox-overlay.spec.ts
 *
 * TDD spec for:
 *  1. ModelContextStore.activeCaptionFormat signal
 *  2. pxToNorm / normToPx coordinate helpers
 *  3. BboxOverlayComponent render + select behavior
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ModelContextStore, type DefinitionRef } from '../../../../../state/model-context.store';
import { BboxOverlayComponent, pxToNorm, normToPx } from './bbox-overlay';

// ---------------------------------------------------------------------------
// 1. ModelContextStore.activeCaptionFormat
// ---------------------------------------------------------------------------

const PLAIN_DEF: DefinitionRef = { id: 'flux1-schnell', family: 'flux1', name: 'Flux.1 Schnell' };
const IDEOGRAM_DEF: DefinitionRef = {
    id: 'ideogram4',
    family: 'ideogram4',
    name: 'Ideogram 4',
    caption_format: 'ideogram4_json',
};

describe('ModelContextStore.activeCaptionFormat', () => {
    beforeEach(() => localStorage.clear());

    it('returns "plain" when model-aware is off', () => {
        const store = new ModelContextStore();
        expect(store.activeCaptionFormat()).toBe('plain');
    });

    it('returns "plain" when model-aware is on but no definition is set', () => {
        const store = new ModelContextStore();
        store.setModelAware(true);
        expect(store.activeCaptionFormat()).toBe('plain');
    });

    it('returns "plain" for a definition with no caption_format', () => {
        const store = new ModelContextStore();
        store.setModelAware(true);
        store.setDefinition(PLAIN_DEF);
        expect(store.activeCaptionFormat()).toBe('plain');
    });

    it('returns the definition caption_format when active and set', () => {
        const store = new ModelContextStore();
        store.setModelAware(true);
        store.setDefinition(IDEOGRAM_DEF);
        expect(store.activeCaptionFormat()).toBe('ideogram4_json');
    });

    it('reverts to "plain" when model-aware is toggled off', () => {
        const store = new ModelContextStore();
        store.setModelAware(true);
        store.setDefinition(IDEOGRAM_DEF);
        expect(store.activeCaptionFormat()).toBe('ideogram4_json');
        store.setModelAware(false);
        expect(store.activeCaptionFormat()).toBe('plain');
    });
});

// ---------------------------------------------------------------------------
// 2. Coordinate helper functions — pxToNorm / normToPx
// ---------------------------------------------------------------------------

describe('pxToNorm', () => {
    // Image is 400px wide × 300px tall in display.
    const W = 400;
    const H = 300;

    it('converts a pixel rect to y-first 0–1000 normalized coords', () => {
        // Pixel rect: left=100, top=60, right=200, bottom=150
        // → x fraction: 100/400=0.25, 200/400=0.5   → x_min=250, x_max=500
        // → y fraction: 60/300=0.2,  150/300=0.5    → y_min=200, y_max=500
        // y-first output: [y_min, x_min, y_max, x_max] = [200, 250, 500, 500]
        const result = pxToNorm(100, 60, 200, 150, W, H);
        expect(result).toEqual([200, 250, 500, 500]);
    });

    it('clamps out-of-bounds pixel coords to [0, 1000]', () => {
        // left=-50 → 0, top=-30 → 0, right=500 > W → 1000, bottom=400 > H → 1000
        const result = pxToNorm(-50, -30, 500, 400, W, H);
        expect(result).toEqual([0, 0, 1000, 1000]);
    });

    it('handles zero-size pixel rects (degenerate)', () => {
        // A zero-area rect at origin
        const result = pxToNorm(0, 0, 0, 0, W, H);
        expect(result).toEqual([0, 0, 0, 0]);
    });

    it('round-trips with normToPx', () => {
        const W2 = 800;
        const H2 = 600;
        const pxIn = [80, 120, 400, 300] as [number, number, number, number]; // left, top, right, bottom
        const norm = pxToNorm(...pxIn, W2, H2);
        const back = normToPx(norm, W2, H2);
        // Back should match the pixel rect
        expect(back.left).toBeCloseTo(80, 0);
        expect(back.top).toBeCloseTo(120, 0);
        expect(back.width).toBeCloseTo(400 - 80, 0);
        expect(back.height).toBeCloseTo(300 - 120, 0);
    });
});

describe('normToPx', () => {
    const W = 400;
    const H = 300;

    it('converts y-first 0–1000 bbox to pixel left/top/width/height', () => {
        // bbox = [y_min=200, x_min=250, y_max=500, x_max=500]
        // → top = 200/1000 * 300 = 60, left = 250/1000 * 400 = 100
        // → bottom = 500/1000 * 300 = 150, right = 500/1000 * 400 = 200
        // → width = 100, height = 90
        const rect = normToPx([200, 250, 500, 500], W, H);
        expect(rect.left).toBeCloseTo(100, 5);
        expect(rect.top).toBeCloseTo(60, 5);
        expect(rect.width).toBeCloseTo(100, 5);
        expect(rect.height).toBeCloseTo(90, 5);
    });

    it('full-frame bbox [0,0,1000,1000] fills the entire image', () => {
        const rect = normToPx([0, 0, 1000, 1000], W, H);
        expect(rect.left).toBe(0);
        expect(rect.top).toBe(0);
        expect(rect.width).toBeCloseTo(W, 5);
        expect(rect.height).toBeCloseTo(H, 5);
    });
});

// ---------------------------------------------------------------------------
// 3. BboxOverlayComponent — render + select
// ---------------------------------------------------------------------------

describe('BboxOverlayComponent', () => {
    beforeEach(() => {
        TestBed.configureTestingModule({
            imports: [BboxOverlayComponent],
        });
    });

    it('renders two box elements when two boxes are provided', () => {
        const fixture = TestBed.createComponent(BboxOverlayComponent);
        fixture.componentRef.setInput('boxes', [
            { id: 'a', bbox: [100, 100, 500, 500] },
            { id: 'b', bbox: [600, 200, 900, 800] },
        ]);
        fixture.detectChanges();
        const boxes = fixture.nativeElement.querySelectorAll('[data-testid="bbox-box"]');
        expect(boxes.length).toBe(2);
    });

    it('renders zero box elements when boxes is empty', () => {
        const fixture = TestBed.createComponent(BboxOverlayComponent);
        fixture.componentRef.setInput('boxes', []);
        fixture.detectChanges();
        const boxes = fixture.nativeElement.querySelectorAll('[data-testid="bbox-box"]');
        expect(boxes.length).toBe(0);
    });

    it('applies selected class to the selected box', () => {
        const fixture = TestBed.createComponent(BboxOverlayComponent);
        fixture.componentRef.setInput('boxes', [
            { id: 'a', bbox: [100, 100, 500, 500] },
            { id: 'b', bbox: [600, 200, 900, 800] },
        ]);
        fixture.componentRef.setInput('selectedId', 'b');
        fixture.detectChanges();
        const boxes = fixture.nativeElement.querySelectorAll('[data-testid="bbox-box"]');
        expect(boxes[0].classList.contains('bbox-selected')).toBe(false);
        expect(boxes[1].classList.contains('bbox-selected')).toBe(true);
    });

    it('emits boxSelected with the box id when a box is clicked', () => {
        const fixture: ComponentFixture<BboxOverlayComponent> = TestBed.createComponent(BboxOverlayComponent);
        fixture.componentRef.setInput('boxes', [
            { id: 'box1', bbox: [100, 100, 500, 500] },
            { id: 'box2', bbox: [600, 200, 900, 800] },
        ]);
        fixture.detectChanges();

        const emitted: string[] = [];
        fixture.componentInstance.boxSelected.subscribe((id: string) => emitted.push(id));

        const boxes = fixture.nativeElement.querySelectorAll('[data-testid="bbox-box"]');
        boxes[0].click();
        expect(emitted).toEqual(['box1']);

        boxes[1].click();
        expect(emitted).toEqual(['box1', 'box2']);
    });

    it('does not emit boxSelected when clicking the container (not a box)', () => {
        const fixture: ComponentFixture<BboxOverlayComponent> = TestBed.createComponent(BboxOverlayComponent);
        fixture.componentRef.setInput('boxes', []);
        fixture.detectChanges();

        const emitted: string[] = [];
        fixture.componentInstance.boxSelected.subscribe((id: string) => emitted.push(id));

        const container = fixture.nativeElement.querySelector('[data-testid="bbox-container"]');
        container?.click();
        expect(emitted).toEqual([]);
    });

    it('does not start draw when drawEnabled is false', () => {
        const fixture: ComponentFixture<BboxOverlayComponent> = TestBed.createComponent(BboxOverlayComponent);
        fixture.componentRef.setInput('drawEnabled', false);
        fixture.detectChanges();

        const emitted: number[][] = [];
        fixture.componentInstance.boxAdded.subscribe((b: number[]) => emitted.push(b));

        const container = fixture.nativeElement.querySelector('[data-testid="bbox-container"]');
        if (container) {
            container.dispatchEvent(new MouseEvent('pointerdown', { clientX: 10, clientY: 10, bubbles: true }));
            container.dispatchEvent(new MouseEvent('pointermove', { clientX: 50, clientY: 50, bubbles: true }));
            container.dispatchEvent(new MouseEvent('pointerup', { clientX: 50, clientY: 50, bubbles: true }));
        }
        expect(emitted).toEqual([]);
    });
});
