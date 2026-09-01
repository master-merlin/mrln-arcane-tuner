import {
    GRID_GAP_PX,
    HPS_PILL_PX,
    MIN_TILE_PX,
    TILE_ACTION_ROW_PX,
    TILE_HEADER_GAP_PX,
    TILE_HEADER_INSET_PX,
    createGridFit,
    effectiveColumns,
    headerBandFits,
} from './grid-fit';

/**
 * Geometry the browser measured on the live workspace grid (18 tiles, all
 * five action buttons present, viewport 1920x1200, chrome 58px). These are
 * the numbers the fix has to keep true, so the pinning test computes with
 * them rather than with the module's own constants where it can.
 */
const MEASURED_PILL_PX = 87.1;
const MEASURED_ACTIONS_PX = 133;
const CHROME_PX = 58;
/** The three widths the defect was measured at, plus a wide desktop. */
const VIEWPORTS = [1280, 1440, 1920, 2560];
const DENSITIES = [3, 4, 5, 6, 7];

/** Width of one column for `columns` columns in `available` px of grid. */
function tileWidth(available: number, columns: number): number {
    return (available - (columns - 1) * GRID_GAP_PX) / columns;
}

describe('effectiveColumns', () => {
    it('never returns more columns than were requested', () => {
        for (const w of [320, 800, 1280, 1920, 2560, 5120]) {
            for (const d of DENSITIES) {
                expect(effectiveColumns(w, d)).toBeLessThanOrEqual(d);
            }
        }
    });

    it('is the identity when the width is generous — the widest viewport at the lowest density is unchanged', () => {
        expect(effectiveColumns(2560 - CHROME_PX, 3)).toBe(3);
        expect(effectiveColumns(1920 - CHROME_PX, 3)).toBe(3);
        expect(effectiveColumns(1920 - CHROME_PX, 4)).toBe(4);
        expect(effectiveColumns(1440 - CHROME_PX, 3)).toBe(3);
        expect(effectiveColumns(1280 - CHROME_PX, 3)).toBe(3);
    });

    it('returns the request untouched when the width has not been measured', () => {
        // First paint, jsdom, SSR, or a browser without ResizeObserver: degrade
        // to the layout that shipped, never to a single column.
        for (const d of DENSITIES) {
            expect(effectiveColumns(0, d)).toBe(d);
            expect(effectiveColumns(-1, d)).toBe(d);
            expect(effectiveColumns(Number.NaN, d)).toBe(d);
        }
    });

    it('never returns less than one column', () => {
        expect(effectiveColumns(10, 7)).toBe(1);
    });

    /**
     * THE PINNING TEST (RULE-20, prevention class T).
     *
     * The defect: the grid used the raw density as its column count, so at
     * 1440/density 5 the tile was 250.8px and the centred filename ran 42.9px
     * under the HPS pill and 88.7px under the action row; at 1280/density 7
     * the 133px action row overflowed the 147.1px tile's own inset by 2.9px.
     * Mutation-proof: replace the body of `effectiveColumns` with
     * `return requestedColumns` (the shipped behaviour) and this goes red on
     * eleven of the twenty (viewport, density) pairs.
     */
    it('never produces a tile too narrow for the header band, at any viewport x density', () => {
        const collisions: string[] = [];
        for (const viewport of VIEWPORTS) {
            const available = viewport - CHROME_PX;
            for (const requested of DENSITIES) {
                const columns = effectiveColumns(available, requested);
                const tile = tileWidth(available, columns);
                // The band lays out as: inset | pill | gap | name | gap | actions | inset.
                const flankers =
                    TILE_HEADER_INSET_PX + MEASURED_PILL_PX + TILE_HEADER_GAP_PX + MEASURED_ACTIONS_PX;
                if (tile < flankers) {
                    collisions.push(
                        `${viewport}px/density ${requested} -> ${columns} cols,`
                        + ` tile ${tile.toFixed(1)}px < band ${flankers.toFixed(1)}px`,
                    );
                }
                expect(headerBandFits(tile)).toBe(true);
            }
        }
        expect(collisions).toEqual([]);
    });
});

describe('MIN_TILE_PX', () => {
    it('is the measured min-content width of the header band', () => {
        expect(MIN_TILE_PX).toBeGreaterThanOrEqual(
            TILE_HEADER_INSET_PX + MEASURED_PILL_PX + TILE_HEADER_GAP_PX + MEASURED_ACTIONS_PX,
        );
        // The constants must not drift below what the browser measured.
        expect(HPS_PILL_PX).toBeGreaterThanOrEqual(MEASURED_PILL_PX);
        expect(TILE_ACTION_ROW_PX).toBeGreaterThanOrEqual(MEASURED_ACTIONS_PX);
    });
});

describe('createGridFit', () => {
    class FakeRO {
        static instances: FakeRO[] = [];
        disconnected = 0;
        targets: Element[] = [];
        constructor(private cb: ResizeObserverCallback) {
            FakeRO.instances.push(this);
        }
        observe(el: Element): void {
            this.targets.push(el);
        }
        unobserve(): void {}
        disconnect(): void {
            this.disconnected++;
            this.targets = [];
        }
        fire(): void {
            this.cb([], this as unknown as ResizeObserver);
        }
    }

    let original: typeof ResizeObserver | undefined;

    beforeEach(() => {
        FakeRO.instances = [];
        original = globalThis.ResizeObserver;
        globalThis.ResizeObserver = FakeRO as unknown as typeof ResizeObserver;
    });

    afterEach(() => {
        if (original) globalThis.ResizeObserver = original;
        else delete (globalThis as { ResizeObserver?: unknown }).ResizeObserver;
    });

    function el(width: number): HTMLElement {
        const node = document.createElement('div');
        Object.defineProperty(node, 'clientWidth', { value: width, configurable: true });
        return node;
    }

    it('starts at zero and reports the element width on observe', () => {
        const fit = createGridFit();
        expect(fit.width()).toBe(0);
        fit.observe(el(1862));
        expect(fit.width()).toBe(1862);
    });

    it('re-observing the same element never accumulates observers', () => {
        const fit = createGridFit();
        const node = el(1000);
        fit.observe(node);
        fit.observe(node);
        fit.observe(node);
        expect(FakeRO.instances.length).toBe(1);
    });

    it('picks up a later resize of the observed element', () => {
        const fit = createGridFit();
        const node = el(1000);
        fit.observe(node);
        Object.defineProperty(node, 'clientWidth', { value: 640, configurable: true });
        FakeRO.instances[0].fire();
        expect(fit.width()).toBe(640);
    });

    /**
     * A teardown guard is not a guard until a test has seen it STOP something:
     * assert the effect (the width no longer moves) rather than a flag.
     */
    it('stops updating after destroy — a late observer callback changes nothing', () => {
        const fit = createGridFit();
        const node = el(1000);
        fit.observe(node);
        const ro = FakeRO.instances[0];
        // `observe` disconnects before (re)observing, so the counter is
        // already non-zero here; the guard is that destroy adds one more
        // AND that nothing moves afterwards.
        const before = ro.disconnected;
        fit.destroy();
        expect(ro.disconnected).toBe(before + 1);
        expect(ro.targets).toEqual([]);

        Object.defineProperty(node, 'clientWidth', { value: 320, configurable: true });
        ro.fire();
        expect(fit.width()).toBe(1000);

        // …and a fresh observe after destroy is a no-op, not a resurrection.
        fit.observe(el(444));
        expect(fit.width()).toBe(1000);
        expect(FakeRO.instances.length).toBe(1);
    });

    it('degrades to unmeasured when ResizeObserver is absent', () => {
        delete (globalThis as { ResizeObserver?: unknown }).ResizeObserver;
        const fit = createGridFit();
        fit.observe(el(0));
        expect(fit.width()).toBe(0);
        expect(effectiveColumns(fit.width(), 7)).toBe(7);
    });
});
