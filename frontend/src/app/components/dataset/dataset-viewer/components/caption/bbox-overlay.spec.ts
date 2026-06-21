/**
 * bbox-overlay.spec.ts
 *
 * TDD spec for:
 *  1. ModelContextStore.activeCaptionFormat signal
 *  2. pxToNorm / normToPx coordinate helpers
 *  3. movedBbox / resizedBbox geometry helpers (move + resize)
 *  4. BboxOverlayComponent render + select + handle render behavior
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ModelContextStore, type DefinitionRef } from '../../../../../state/model-context.store';
import { BboxOverlayComponent, BboxItem, pxToNorm, normToPx, movedBbox, resizedBbox } from './bbox-overlay';

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
// 3. movedBbox — pure geometry helper for translating a box
// ---------------------------------------------------------------------------

describe('movedBbox', () => {
    // Starting box: y_min=200, x_min=100, y_max=500, x_max=400 (300h × 300w)
    const box: number[] = [200, 100, 500, 400];

    it('translates down and right by the given deltas', () => {
        // dy=+50, dx=+100 → y_min=250, x_min=200, y_max=550, x_max=500
        const result = movedBbox(box, 50, 100);
        expect(result).toEqual([250, 200, 550, 500]);
    });

    it('translates up and left by negative deltas', () => {
        // dy=-50, dx=-50 → y_min=150, x_min=50, y_max=450, x_max=350
        const result = movedBbox(box, -50, -50);
        expect(result).toEqual([150, 50, 450, 350]);
    });

    it('preserves the original width and height after translation', () => {
        const result = movedBbox(box, 30, 70);
        const origH = box[2] - box[0];
        const origW = box[3] - box[1];
        expect(result[2] - result[0]).toBe(origH);
        expect(result[3] - result[1]).toBe(origW);
    });

    it('clamps at the top-left boundary (cannot go negative)', () => {
        // dy=-9999, dx=-9999 → clamped to top-left corner, preserving size
        const result = movedBbox(box, -9999, -9999);
        expect(result[0]).toBe(0);   // y_min = 0
        expect(result[1]).toBe(0);   // x_min = 0
        expect(result[2]).toBe(300); // y_max = 0 + (500-200)
        expect(result[3]).toBe(300); // x_max = 0 + (400-100)
    });

    it('clamps at the bottom-right boundary (cannot exceed 1000)', () => {
        // dy=+9999, dx=+9999 → clamped so trailing edge stays at 1000
        const result = movedBbox(box, 9999, 9999);
        const origH = box[2] - box[0]; // 300
        const origW = box[3] - box[1]; // 300
        expect(result[2]).toBe(1000);               // y_max = 1000
        expect(result[3]).toBe(1000);               // x_max = 1000
        expect(result[0]).toBe(1000 - origH);        // y_min
        expect(result[1]).toBe(1000 - origW);        // x_min
    });

    it('zero delta is a no-op (returns rounded copy of original)', () => {
        const result = movedBbox(box, 0, 0);
        expect(result).toEqual([200, 100, 500, 400]);
    });

    it('handles fractional normalized deltas (rounds to integers)', () => {
        // 0.4 in normalized space — should be rounded
        const result = movedBbox([100, 100, 200, 200], 0.4, 0.6);
        // ny_min: round(100 + 0.4) = round(100.4) = 100
        // nx_min: round(100 + 0.6) = round(100.6) = 101
        // height = 200-100 = 100; width = 200-100 = 100
        expect(result[0]).toBe(100);  // ny_min
        expect(result[1]).toBe(101);  // nx_min
        expect(result[2]).toBe(200);  // ny_min + h = 100 + 100
        expect(result[3]).toBe(201);  // nx_min + w = 101 + 100
    });
});

// ---------------------------------------------------------------------------
// 4. resizedBbox — pure geometry helper for corner resize
// ---------------------------------------------------------------------------

describe('resizedBbox', () => {
    // Starting box: [y_min=100, x_min=100, y_max=500, x_max=500]
    const box: number[] = [100, 100, 500, 500];

    describe('tl (top-left) corner — opposite is bottom-right (500,500)', () => {
        it('moves the top-left corner to new position', () => {
            // Drag tl to (50, 50) → new box [50,50,500,500]
            const result = resizedBbox(box, 'tl', 50, 50);
            expect(result).toEqual([50, 50, 500, 500]);
        });

        it('clamps to [0,0] at the min boundary', () => {
            const result = resizedBbox(box, 'tl', -100, -100);
            expect(result).toEqual([0, 0, 500, 500]);
        });

        it('handles cross-over: dragging past bottom-right flips y_min/y_max and x_min/x_max', () => {
            // Drag tl to (700, 700) — past the fixed corner at (500,500)
            // → sorted: y_min=500, x_min=500, y_max=700, x_max=700
            const result = resizedBbox(box, 'tl', 700, 700);
            expect(result).toEqual([500, 500, 700, 700]);
        });
    });

    describe('tr (top-right) corner — opposite is bottom-left (500,100)', () => {
        it('moves the top-right corner to new position', () => {
            // Drag tr to (50, 700) → new box [50,100,500,700]
            const result = resizedBbox(box, 'tr', 50, 700);
            expect(result).toEqual([50, 100, 500, 700]);
        });

        it('clamps x_max to 1000', () => {
            const result = resizedBbox(box, 'tr', 50, 1200);
            expect(result[3]).toBe(1000);
        });

        it('cross-over: dragging past the fixed x (x_min=100) swaps x columns', () => {
            // Drag tr to (50, 50) — x=50 is left of fixed x_min=100
            // → x sorted: min(100,50)=50, max(100,50)=100
            const result = resizedBbox(box, 'tr', 50, 50);
            expect(result[1]).toBe(50);
            expect(result[3]).toBe(100);
        });
    });

    describe('bl (bottom-left) corner — opposite is top-right (100,500)', () => {
        it('moves the bottom-left corner to new position', () => {
            // Drag bl to (700, 50) → new box [100,50,700,500]
            const result = resizedBbox(box, 'bl', 700, 50);
            expect(result).toEqual([100, 50, 700, 500]);
        });

        it('cross-over: dragging y above top flips y rows', () => {
            // Drag bl to (50, 50) — y=50 is above fixed y_min=100
            // → y sorted: min(100,50)=50, max(100,50)=100
            const result = resizedBbox(box, 'bl', 50, 50);
            expect(result[0]).toBe(50);
            expect(result[2]).toBe(100);
        });
    });

    describe('br (bottom-right) corner — opposite is top-left (100,100)', () => {
        it('moves the bottom-right corner to new position', () => {
            // Drag br to (700, 700) → new box [100,100,700,700]
            const result = resizedBbox(box, 'br', 700, 700);
            expect(result).toEqual([100, 100, 700, 700]);
        });

        it('clamps to [0,BBOX_MAX]', () => {
            const result = resizedBbox(box, 'br', 1200, 1200);
            expect(result[2]).toBe(1000);
            expect(result[3]).toBe(1000);
        });

        it('cross-over past top-left produces sorted output', () => {
            // Drag br to (50, 50) — past the fixed corner at (100,100)
            // → sorted: y_min=50, x_min=50, y_max=100, x_max=100
            const result = resizedBbox(box, 'br', 50, 50);
            expect(result).toEqual([50, 50, 100, 100]);
        });
    });

    it('result always has y_min <= y_max and x_min <= x_max regardless of corner or direction', () => {
        const corners: Array<'tl' | 'tr' | 'bl' | 'br'> = ['tl', 'tr', 'bl', 'br'];
        const positions = [
            [0, 0], [1000, 1000], [50, 900], [900, 50],
            [500, 500], [200, 800], [800, 200],
        ];
        for (const corner of corners) {
            for (const [y, x] of positions) {
                const result = resizedBbox(box, corner, y, x);
                expect(result[0]).toBeLessThanOrEqual(result[2]);
                expect(result[1]).toBeLessThanOrEqual(result[3]);
                expect(result[0]).toBeGreaterThanOrEqual(0);
                expect(result[1]).toBeGreaterThanOrEqual(0);
                expect(result[2]).toBeLessThanOrEqual(1000);
                expect(result[3]).toBeLessThanOrEqual(1000);
            }
        }
    });
});

// ---------------------------------------------------------------------------
// 5. BboxOverlayComponent — render + select + handle render
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

    // -------------------------------------------------------------------------
    // Handle rendering (jsdom-compatible: checks presence in DOM, not pixel pos)
    // -------------------------------------------------------------------------

    it('renders four corner handles only for the selected box', () => {
        const fixture: ComponentFixture<BboxOverlayComponent> = TestBed.createComponent(BboxOverlayComponent);
        fixture.componentRef.setInput('boxes', [
            { id: 'a', bbox: [100, 100, 500, 500] },
            { id: 'b', bbox: [600, 200, 900, 800] },
        ]);
        fixture.componentRef.setInput('selectedId', 'a');
        fixture.detectChanges();

        // All four handles present for box 'a'
        expect(fixture.nativeElement.querySelectorAll('[data-testid="bbox-handle-tl"]').length).toBe(1);
        expect(fixture.nativeElement.querySelectorAll('[data-testid="bbox-handle-tr"]').length).toBe(1);
        expect(fixture.nativeElement.querySelectorAll('[data-testid="bbox-handle-bl"]').length).toBe(1);
        expect(fixture.nativeElement.querySelectorAll('[data-testid="bbox-handle-br"]').length).toBe(1);
    });

    it('renders no handles when no box is selected', () => {
        const fixture: ComponentFixture<BboxOverlayComponent> = TestBed.createComponent(BboxOverlayComponent);
        fixture.componentRef.setInput('boxes', [
            { id: 'a', bbox: [100, 100, 500, 500] },
        ]);
        fixture.componentRef.setInput('selectedId', null);
        fixture.detectChanges();

        expect(fixture.nativeElement.querySelectorAll('[data-testid^="bbox-handle-"]').length).toBe(0);
    });

    it('switches handles to the newly selected box when selectedId changes', () => {
        const fixture: ComponentFixture<BboxOverlayComponent> = TestBed.createComponent(BboxOverlayComponent);
        fixture.componentRef.setInput('boxes', [
            { id: 'a', bbox: [100, 100, 500, 500] },
            { id: 'b', bbox: [600, 200, 900, 800] },
        ]);
        fixture.componentRef.setInput('selectedId', 'a');
        fixture.detectChanges();

        // Initially handles on box 'a' — 4 handles total
        expect(fixture.nativeElement.querySelectorAll('[data-testid^="bbox-handle-"]').length).toBe(4);

        // Select box 'b'
        fixture.componentRef.setInput('selectedId', 'b');
        fixture.detectChanges();

        // Still 4 handles total (now on box 'b')
        expect(fixture.nativeElement.querySelectorAll('[data-testid^="bbox-handle-"]').length).toBe(4);
    });

    // -------------------------------------------------------------------------
    // boxChanged via helpers — pointer-drag emit exercised via helper math
    // (jsdom getBoundingClientRect returns zero-rect so DOM drag can't be tested)
    // -------------------------------------------------------------------------

    it('movedBbox helper produces correct y-first clamped output (unit integration)', () => {
        // Validates the exact math that onPointerMove / onPointerUp call
        const orig = [200, 100, 500, 400]; // h=300, w=300
        const moved = movedBbox(orig, 50, 100);
        expect(moved).toEqual([250, 200, 550, 500]);
        // Width/height preserved
        expect(moved[2] - moved[0]).toBe(orig[2] - orig[0]);
        expect(moved[3] - moved[1]).toBe(orig[3] - orig[1]);
    });

    it('resizedBbox helper produces sorted clamped output for all corners (unit integration)', () => {
        const orig = [100, 100, 500, 500];
        // br drag to (800, 800)
        expect(resizedBbox(orig, 'br', 800, 800)).toEqual([100, 100, 800, 800]);
        // tl drag to (0, 0)
        expect(resizedBbox(orig, 'tl', 0, 0)).toEqual([0, 0, 500, 500]);
        // Cross-over: tl dragged past br
        const crossed = resizedBbox(orig, 'tl', 700, 700);
        expect(crossed[0]).toBeLessThanOrEqual(crossed[2]);
        expect(crossed[1]).toBeLessThanOrEqual(crossed[3]);
    });

    // NOTE: Full pointer-drag → boxChanged emit path cannot be exercised in jsdom
    // because getBoundingClientRect() always returns zero-sized rects, causing
    // imgW/imgH === 0 guards to short-circuit both move and resize emission.
    // The pure-helper tests above validate the geometry math directly.
    // E2E / Playwright tests would cover the full interaction path.
});

// ---------------------------------------------------------------------------
// 6. Input-array immutability during drag — _liveBbox signal-based override
// ---------------------------------------------------------------------------

/**
 * Helper to access the private _liveBbox signal for test assertions.
 * Using `(comp as any)` is standard Angular test practice for private fields.
 */
function getLiveBbox(comp: BboxOverlayComponent): { id: string; bbox: number[] } | null {
    return (comp as any)._liveBbox();
}

/**
 * Helper to drive the internal live-preview setter directly, bypassing the
 * pointer-event path that jsdom cannot exercise (getBoundingClientRect → zero rect).
 */
function callUpdateLiveBbox(comp: BboxOverlayComponent, id: string, bbox: number[]): void {
    (comp as any)._updateLiveBbox(id, bbox);
}

describe('BboxOverlayComponent — input-array immutability during drag', () => {
    beforeEach(() => {
        TestBed.configureTestingModule({
            imports: [BboxOverlayComponent],
        });
    });

    it('does NOT mutate the input bbox array when _updateLiveBbox is called during a drag', () => {
        const fixture = TestBed.createComponent(BboxOverlayComponent);
        const origBbox = [100, 100, 500, 500];
        const inputBoxes: BboxItem[] = [{ id: 'box1', bbox: origBbox }];
        fixture.componentRef.setInput('boxes', inputBoxes);
        fixture.detectChanges();

        // Capture reference to the original array
        const bboxRef = inputBoxes[0].bbox;
        const bboxSnapshot = [...bboxRef];

        // Simulate mid-drag live-preview update (as onPointerMove would call it)
        const movedPreview = [150, 150, 550, 550];
        callUpdateLiveBbox(fixture.componentInstance, 'box1', movedPreview);

        // The input array must be unchanged
        expect(bboxRef).toEqual(bboxSnapshot);
        // Array identity must be preserved (no splice/mutation)
        expect(bboxRef[0]).toBe(bboxSnapshot[0]);
        expect(bboxRef[1]).toBe(bboxSnapshot[1]);
        expect(bboxRef[2]).toBe(bboxSnapshot[2]);
        expect(bboxRef[3]).toBe(bboxSnapshot[3]);
    });

    it('_liveBbox signal holds the override after _updateLiveBbox', () => {
        const fixture = TestBed.createComponent(BboxOverlayComponent);
        fixture.componentRef.setInput('boxes', [{ id: 'box1', bbox: [100, 100, 500, 500] }]);
        fixture.detectChanges();

        const preview = [200, 200, 600, 600];
        callUpdateLiveBbox(fixture.componentInstance, 'box1', preview);

        const live = getLiveBbox(fixture.componentInstance);
        expect(live).not.toBeNull();
        expect(live!.id).toBe('box1');
        expect(live!.bbox).toEqual(preview);
    });

    it('boxStyle() returns style based on _liveBbox override for the dragged box, not the input bbox', () => {
        const fixture = TestBed.createComponent(BboxOverlayComponent);
        // jsdom: no real layout, so boxStyle falls back to %-based style using bbox values
        const inputBbox = [100, 100, 500, 500];
        fixture.componentRef.setInput('boxes', [{ id: 'box1', bbox: inputBbox }]);
        fixture.detectChanges();

        // Before drag: style reflects the input bbox
        const comp = fixture.componentInstance;
        const styleBefore = (comp as any).boxStyle({ id: 'box1', bbox: inputBbox });
        expect(styleBefore).toContain('left:10%'); // x_min=100 → 10%

        // Simulate drag: set live preview
        const livePreviewBbox = [200, 300, 600, 700];
        callUpdateLiveBbox(comp, 'box1', livePreviewBbox);

        // During drag: style should reflect the live override, not the (unchanged) input
        const styleDuring = (comp as any).boxStyle({ id: 'box1', bbox: inputBbox });
        expect(styleDuring).toContain('left:30%'); // x_min=300 → 30%
        expect(styleDuring).not.toContain('left:10%');

        // Input array still unchanged
        expect(inputBbox).toEqual([100, 100, 500, 500]);
    });

    it('boxStyle() falls back to input bbox for boxes whose id does NOT match the live override', () => {
        const fixture = TestBed.createComponent(BboxOverlayComponent);
        const box1Bbox = [100, 100, 500, 500];
        const box2Bbox = [600, 200, 900, 800];
        fixture.componentRef.setInput('boxes', [
            { id: 'box1', bbox: box1Bbox },
            { id: 'box2', bbox: box2Bbox },
        ]);
        fixture.detectChanges();

        const comp = fixture.componentInstance;
        // Set override only for box1
        callUpdateLiveBbox(comp, 'box1', [200, 300, 600, 700]);

        // box2 must still use its own bbox (left = x_min=200 → 20%)
        const styleBox2 = (comp as any).boxStyle({ id: 'box2', bbox: box2Bbox });
        expect(styleBox2).toContain('left:20%'); // x_min=200 → 20%
    });

    it('clearing _liveBbox (set null) reverts boxStyle() to the input bbox', () => {
        const fixture = TestBed.createComponent(BboxOverlayComponent);
        const inputBbox = [100, 100, 500, 500];
        fixture.componentRef.setInput('boxes', [{ id: 'box1', bbox: inputBbox }]);
        fixture.detectChanges();

        const comp = fixture.componentInstance;

        // Set live override
        callUpdateLiveBbox(comp, 'box1', [200, 300, 600, 700]);
        expect((comp as any).boxStyle({ id: 'box1', bbox: inputBbox })).toContain('left:30%');

        // Clear override (as onPointerUp does)
        (comp as any)._liveBbox.set(null);

        // boxStyle must revert to the original input bbox
        const styleAfter = (comp as any).boxStyle({ id: 'box1', bbox: inputBbox });
        expect(styleAfter).toContain('left:10%'); // back to x_min=100 → 10%
    });

    it('_liveBbox is null initially (no drag in progress)', () => {
        const fixture = TestBed.createComponent(BboxOverlayComponent);
        fixture.componentRef.setInput('boxes', [{ id: 'box1', bbox: [100, 100, 500, 500] }]);
        fixture.detectChanges();

        expect(getLiveBbox(fixture.componentInstance)).toBeNull();
    });

    it('does not mutate input bbox array during a resize live-preview update either', () => {
        const fixture = TestBed.createComponent(BboxOverlayComponent);
        const origBbox = [100, 100, 500, 500];
        const inputBoxes: BboxItem[] = [{ id: 'r1', bbox: origBbox }];
        fixture.componentRef.setInput('boxes', inputBoxes);
        fixture.detectChanges();

        const bboxRef = inputBoxes[0].bbox;
        const snapshot = [...bboxRef];

        // Simulate a resize live-preview (what onPointerMove resize path calls)
        const resizedPreview = resizedBbox(origBbox, 'br', 800, 800);
        callUpdateLiveBbox(fixture.componentInstance, 'r1', resizedPreview);

        // Input array untouched
        expect(bboxRef).toEqual(snapshot);
    });
});
