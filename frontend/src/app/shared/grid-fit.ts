import { signal, type Signal } from '@angular/core';

/**
 * How many columns a tile grid may actually paint, given the width it has.
 *
 * Why this exists, measured rather than assumed: the workspace browse grid
 * laid itself out as `repeat(density(), minmax(0, 1fr))` where `density` is a
 * raw column count from a 3-7 slider **with no viewport term at all**, so the
 * same request produced the same column count at 2560px and at 1280px and the
 * canonical 1400/1200/900 breakpoints never reached the grid. Each tile then
 * carries a header band (HPS pill + filename + a five-button action row) whose
 * two flankers are intrinsically sized, so below a certain tile width the band
 * cannot be laid out at all. Measured on the live 1920px grid (18 tiles, all
 * five action buttons present):
 *
 *   viewport 1280 / density 7 -> tile 147.1px, the action row alone (133px)
 *                               overflowed the tile's own 8px inset by 2.9px
 *   viewport 1440 / density 5 -> tile 250.8px, the centred filename ran
 *                               42.9px under the pill and 88.7px under the
 *                               action row
 *   viewport 1920 / density 5 -> tile 346.8px, filename 45.9px under the
 *                               action row (the SHIPPED default)
 *
 * The band itself is now a flex row, so the filename truncates instead of
 * sliding underneath its neighbours; this module supplies the other half —
 * a column count that cannot ask for a tile narrower than that band's
 * min-content width.
 */

/** `left-2` + `right-2` insets of the header band inside the tile. */
export const TILE_HEADER_INSET_PX = 16;
/**
 * The band's two `gap-2` gutters. Both are charged even when the filename
 * shrinks to zero: a flex gap is drawn between every pair of items,
 * regardless of how small the middle one gets.
 */
export const TILE_HEADER_GAP_PX = 16;
/** HPS pill, measured 87.1px (fixed digit count, so it does not vary). */
export const HPS_PILL_PX = 88;
/** Action row at its widest: five 25px buttons + gutters, measured 133px. */
export const TILE_ACTION_ROW_PX = 136;
/** Enough filename for one glyph plus the truncation ellipsis. */
export const TILE_NAME_MIN_PX = 24;

/**
 * Narrowest tile whose header band still lays out without a collision:
 * 16 insets + 88 pill + 16 gaps + 136 actions + 24 name.
 */
export const MIN_TILE_PX =
    TILE_HEADER_INSET_PX + HPS_PILL_PX + TILE_HEADER_GAP_PX + TILE_ACTION_ROW_PX + TILE_NAME_MIN_PX;

/** The grid's `gap-8` gutter between columns, in px. */
export const GRID_GAP_PX = 32;

/**
 * The column count the grid should actually use.
 *
 * Never MORE than the user asked for — the density slider stays a request,
 * and a wide window honours it exactly, so at the widest viewport and the
 * lowest density this function is the identity. It only ever caps.
 *
 * `availableWidthPx <= 0` means "not measured yet" (first paint, jsdom, any
 * SSR pass, a browser without `ResizeObserver`) and returns the request
 * untouched, which is precisely the behaviour that shipped before this
 * module existed: an unmeasured grid degrades to the old layout, never to a
 * single column.
 */
export function effectiveColumns(
    availableWidthPx: number,
    requestedColumns: number,
    minTilePx: number = MIN_TILE_PX,
    gapPx: number = GRID_GAP_PX,
): number {
    const requested = Math.max(1, Math.floor(requestedColumns) || 1);
    if (!Number.isFinite(availableWidthPx) || availableWidthPx <= 0) return requested;
    // n columns occupy n*tile + (n-1)*gap, so n = (width + gap) / (tile + gap).
    const fits = Math.floor((availableWidthPx + gapPx) / (minTilePx + gapPx));
    return Math.max(1, Math.min(requested, fits));
}

/**
 * True when a tile of `tileWidthPx` can lay out the header band without any
 * two of its three parts overlapping. This is the arithmetic the pinning
 * test asserts against every (viewport, density) pair the grid can produce.
 */
export function headerBandFits(
    tileWidthPx: number,
    minTilePx: number = MIN_TILE_PX,
): boolean {
    return tileWidthPx >= minTilePx;
}

export interface GridFit {
    /** Content width of the observed grid element, or 0 before first measure. */
    readonly width: Signal<number>;
    /**
     * Measure `el` now and keep measuring it. Safe to call after every
     * render: the observer instance is reused, so repeated calls never
     * accumulate observers. A null element is ignored.
     */
    observe(el: HTMLElement | null | undefined): void;
    /** Release the observer. After this, `observe` is a no-op and the
     *  width signal never changes again. */
    destroy(): void;
}

/**
 * Build a width tracker. Creating one has no side effect of its own — no
 * observer exists until the first `observe` with a live element.
 */
export function createGridFit(): GridFit {
    const width = signal(0);
    let observer: ResizeObserver | null = null;
    let observed: HTMLElement | null = null;
    let dead = false;

    const measure = (el: HTMLElement): void => {
        if (dead) return;
        const w = el.clientWidth;
        if (w > 0 && w !== width()) width.set(w);
    };

    return {
        width: width.asReadonly(),
        observe(el: HTMLElement | null | undefined): void {
            if (dead || !el || el === observed) {
                if (!dead && el && el === observed) measure(el);
                return;
            }
            observed = el;
            measure(el);
            if (typeof ResizeObserver === 'undefined') return;
            observer ??= new ResizeObserver(() => {
                if (observed) measure(observed);
            });
            observer.disconnect();
            observer.observe(el);
        },
        destroy(): void {
            dead = true;
            observer?.disconnect();
            observer = null;
            observed = null;
        },
    };
}
