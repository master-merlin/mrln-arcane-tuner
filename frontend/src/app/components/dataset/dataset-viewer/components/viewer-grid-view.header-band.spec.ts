/**
 * The tile header band, and the column count that has to leave room for it.
 *
 * The defect (LANE-59, reported by the user as "on dense workspaces our
 * screen elements overlap heavily"): the grid laid itself out as
 * `repeat(density(), minmax(0, 1fr))` where density is a raw 3-7 column
 * count **with no viewport term**, and inside each tile the filename, the
 * HPS pill and the five-button action row were three independent
 * `absolute top-2 …` overlays — two intrinsically sized flankers with a
 * `max-w-[58%]` label centred between them. Measured in the browser on the
 * live grid:
 *
 *   1920 / density 5 (SHIPPED DEFAULT) -> filename 45.9px under the actions
 *   1440 / density 5 -> 42.9px under the pill, 88.7px under the actions
 *   1280 / density 7 -> action row overflowed the tile's own inset by 2.9px
 *
 * These specs pin the two halves of the fix structurally: the three parts
 * are siblings of ONE flex row (so the name yields and truncates instead of
 * sliding underneath), and the rendered column count is capped against the
 * measured width. `grid-fit.spec.ts` carries the arithmetic across every
 * (viewport, density) pair.
 */
import { TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { afterEach, beforeEach } from 'vitest';
import { ViewerGridViewComponent } from './viewer-grid-view';
import type { DatasetPair } from '../../../../services/dataset';
import { MIN_TILE_PX } from '../../../../shared/grid-fit';

/**
 * jsdom does no layout and ships no ResizeObserver, so the component's width
 * tracker would never see anything. This fake is the only stand-in: the seam
 * under test (the clamp) is untouched, only the browser API below it.
 */
class FakeResizeObserver {
    static callbacks: (() => void)[] = [];
    constructor(cb: () => void) {
        FakeResizeObserver.callbacks.push(cb);
    }
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
}

function fireResize(): void {
    for (const cb of FakeResizeObserver.callbacks) cb();
}

let realResizeObserver: unknown;

beforeEach(() => {
    FakeResizeObserver.callbacks = [];
    realResizeObserver = (globalThis as Record<string, unknown>)['ResizeObserver'];
    (globalThis as Record<string, unknown>)['ResizeObserver'] = FakeResizeObserver;
});

afterEach(() => {
    (globalThis as Record<string, unknown>)['ResizeObserver'] = realResizeObserver;
});

function makePair(i: number, overrides: Partial<DatasetPair> = {}): DatasetPair {
    return {
        stem: `img${i}`,
        media_file: `a-rather-long-filename-${i}.jpg`,
        media_type: 'image',
        caption_file: `img${i}.txt`,
        caption_content: `caption ${i}`,
        masked_caption_content: null,
        lyrics_file: null,
        lyrics_content: '',
        metadata: { enabled: true, width: 512, height: 512, quality_score: 0.2814 },
        control_files: [],
        role_order: null,
        effective_target: `img${i}.jpg`,
        effective_controls: [],
        ...overrides,
    } as DatasetPair;
}

@Component({
    standalone: true,
    imports: [ViewerGridViewComponent],
    template: `
        <app-viewer-grid-view
            [pairs]="pairs()"
            [datasetName]="'ds'"
            [mediaBaseUrl]="'/media'"
            [apiUrl]="'/api'"
            [hideToolbar]="true"
            [density]="density()"
            [datasetKind]="'standard'"
            (effectiveDensityChange)="effective.push($event)"/>
    `,
})
class Host {
    pairs = signal<DatasetPair[]>([]);
    density = signal<number>(5);
    effective: number[] = [];
}

/** A host that passes NO density, to pin the component's own default. */
@Component({
    standalone: true,
    imports: [ViewerGridViewComponent],
    template: `
        <app-viewer-grid-view
            [pairs]="pairs()"
            [datasetName]="'ds'"
            [mediaBaseUrl]="'/media'"
            [apiUrl]="'/api'"
            [hideToolbar]="true"
            [datasetKind]="'standard'"/>
    `,
})
class DefaultHost {
    pairs = signal<DatasetPair[]>([makePair(0)]);
}

/**
 * Render the grid with the grid element reporting `gridWidth` px, which is
 * what the component's ResizeObserver-backed tracker reads. jsdom does no
 * layout, so the width has to be stamped on the element the same way the
 * browser would report it.
 */
function render(pairs: DatasetPair[], density: number, gridWidth: number | null) {
    TestBed.resetTestingModule();
    TestBed.configureTestingModule({ imports: [Host] });
    const fixture = TestBed.createComponent(Host);
    fixture.componentInstance.pairs.set(pairs);
    fixture.componentInstance.density.set(density);
    fixture.detectChanges();
    if (gridWidth != null) {
        const grid = fixture.nativeElement.querySelector('div.grid') as HTMLElement;
        Object.defineProperty(grid, 'clientWidth', { value: gridWidth, configurable: true });
        // Drive it the way a browser does: the component measures through a
        // ResizeObserver, so the width only reaches it when the observer
        // fires. jsdom has no real one, hence the fake installed above.
        fireResize();
        fixture.detectChanges();
    }
    return fixture;
}

function grid(fixture: { nativeElement: HTMLElement }): HTMLElement {
    return fixture.nativeElement.querySelector('div.grid') as HTMLElement;
}

describe('viewer-grid-view — tile header band is one laid-out row', () => {
    it('renders pill, filename and actions as siblings of a single flex band', () => {
        const fixture = render([makePair(0)], 4, null);
        const band = fixture.nativeElement.querySelector('[data-testid="tile-header-band"]') as HTMLElement;
        expect(band).toBeTruthy();
        expect(band.className).toContain('flex');

        const pill = band.querySelector('.hps-pill');
        const name = band.querySelector('[data-testid="tile-filename"]');
        const actions = band.querySelector('[data-testid="tile-actions"]');
        expect(pill).toBeTruthy();
        expect(name).toBeTruthy();
        expect(actions).toBeTruthy();

        // All three inside the ONE band — the defect was three separate
        // `absolute` overlays that shared only a `top-2`.
        for (const el of [pill!, name!, actions!]) {
            expect(band.contains(el)).toBe(true);
        }
    });

    it('leaves none of the three parts absolutely positioned — the band positions, the row lays out', () => {
        const fixture = render([makePair(0)], 4, null);
        const band = fixture.nativeElement.querySelector('[data-testid="tile-header-band"]') as HTMLElement;
        const parts = [
            band.querySelector('.hps-pill')!,
            band.querySelector('[data-testid="tile-filename"]')!,
            band.querySelector('[data-testid="tile-actions"]')!,
        ];
        for (const el of parts) {
            expect(el.className).not.toContain('absolute');
            // A percentage cap on the label between fixed-pixel flankers is
            // exactly what let it run underneath them.
            expect(el.className).not.toContain('max-w-[58%]');
            expect(el.className).not.toContain('-translate-x-1/2');
        }
    });

    it('lets the filename shrink (min-w-0) while the flankers cannot (shrink-0)', () => {
        const fixture = render([makePair(0)], 4, null);
        const band = fixture.nativeElement.querySelector('[data-testid="tile-header-band"]') as HTMLElement;
        expect(band.querySelector('[data-testid="tile-filename"]')!.className).toContain('min-w-0');
        expect(band.querySelector('[data-testid="tile-filename"]')!.className).toContain('truncate');
        expect(band.querySelector('.hps-pill')!.className).toContain('shrink-0');
        expect(band.querySelector('[data-testid="tile-actions"]')!.className).toContain('shrink-0');
    });

    it('keeps the filename start-aligned so truncation clips the END, never the front', () => {
        // UAT 2026-09-02: with `text-center` on the truncating label, a name
        // that overflowed its box was clipped on BOTH sides — the ellipsis
        // marks only the end, so "alfa-romeo-…" rendered as "-romeo-…" with
        // no sign anything was missing. Centred nowrap overflow is clipped
        // symmetrically; only start-aligned text keeps the front visible.
        // Alignment cannot be observed in jsdom (no layout), so the pin is
        // the class contract itself.
        const fixture = render([makePair(0)], 4, null);
        const name = fixture.nativeElement.querySelector('[data-testid="tile-filename"]')!;
        expect(name.className).toContain('truncate');
        expect(name.className).not.toContain('text-center');
    });
});

describe('viewer-grid-view — column count is capped by the measured width', () => {
    it('honours the request exactly when the grid is wide enough (1920 desktop, density 4)', () => {
        const fixture = render([makePair(0)], 4, 1862);
        expect(grid(fixture).style.gridTemplateColumns).toBe('repeat(4, minmax(0, 1fr))');
        expect(grid(fixture).getAttribute('data-effective-density')).toBe('4');
    });

    it('caps the request when the grid is too narrow for that many header bands', () => {
        // 1280px laptop: 1222px of grid. 7 columns would be a 147px tile; the
        // action row alone is 133px and overflowed the tile in the browser.
        const fixture = render([makePair(0)], 7, 1222);
        const columns = Number(grid(fixture).getAttribute('data-effective-density'));
        expect(columns).toBeLessThan(7);
        const tile = (1222 - (columns - 1) * 32) / columns;
        expect(tile).toBeGreaterThanOrEqual(MIN_TILE_PX);
    });

    it('reports the painted column count to the host so the readout cannot lie', () => {
        const fixture = render([makePair(0)], 7, 1222);
        const emitted = fixture.componentInstance.effective;
        expect(emitted.length).toBeGreaterThan(0);
        expect(emitted[emitted.length - 1]).toBe(
            Number(grid(fixture).getAttribute('data-effective-density')),
        );
    });

    it('falls back to the requested count while the width is unmeasured', () => {
        const fixture = render([makePair(0)], 7, null);
        expect(grid(fixture).style.gridTemplateColumns).toBe('repeat(7, minmax(0, 1fr))');
    });

    it('leaves the shipped default density at 5 - the cap is the fix, not a new constant', () => {
        // Deliberately NOT re-defaulted to 4: swapping one hardcoded column
        // count for another is the same defect with a different number. A
        // caller that passes no density still gets 5; the width cap does the
        // rest, and on a narrow viewport 5 resolves to fewer columns.
        TestBed.resetTestingModule();
        TestBed.configureTestingModule({ imports: [DefaultHost] });
        const fixture = TestBed.createComponent(DefaultHost);
        fixture.detectChanges();
        const g = fixture.nativeElement.querySelector('div.grid') as HTMLElement;
        expect(g.getAttribute('data-effective-density')).toBe('5');
    });
});
