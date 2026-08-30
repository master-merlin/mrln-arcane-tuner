/**
 * The tile loader must not animate for tiles nobody can see, and adding
 * `data-index` to the tile must not create a second, drifting notion of
 * "which item is this".
 *
 * Measured on a 263-item dataset (production build, `ng serve
 * --configuration production`): opening the workspace left **1503 CSS
 * animations running at once** — 714 of them the grid's own loader dots,
 * three per unloaded tile, on 238 tiles whose `loading="lazy"` media had
 * never been requested and therefore would never fire `load`. The page
 * rendered at 35.5 ms / 28 fps *standing still*: idle frame cost equalled
 * scrolling frame cost, which is why this was never a scroll problem and
 * why virtualizing the grid was not the fix. Bounding the animation to
 * on-screen tiles took the same sweep to 18.1 ms / 55 fps.
 *
 * The index guard is here because `detailRequested.emit(i)` carries an
 * index into `pairs()` that `browse-mode.translateVisibleIdx` resolves
 * back to an unfiltered index by `media_file`. `data-index` is a SECOND
 * expression of that same index; if the two ever disagree, open-detail,
 * edit, crop, exclude, delete and the cover pin all silently target the
 * wrong image. These specs assert the rendered attribute and the emitted
 * value agree on the same tile.
 */
import { TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { ViewerGridViewComponent } from './viewer-grid-view';
import type { DatasetPair } from '../../../../services/dataset';

type Cb = (entries: IntersectionObserverEntry[]) => void;

interface FakeIO { cb: Cb; observed: HTMLElement[] }

let ios: FakeIO[] = [];
let realIO: unknown;

function installFakeIO(): void {
    realIO = (globalThis as Record<string, unknown>)['IntersectionObserver'];
    class Fake {
        constructor(cb: Cb) { this.rec = { cb, observed: [] }; ios.push(this.rec); }
        private rec: FakeIO;
        observe(el: HTMLElement): void { this.rec.observed.push(el); }
        unobserve(): void { /* unused */ }
        disconnect(): void { this.rec.observed = []; }
        takeRecords(): [] { return []; }
    }
    (globalThis as Record<string, unknown>)['IntersectionObserver'] = Fake;
}

function makePair(i: number, overrides: Partial<DatasetPair> = {}): DatasetPair {
    return {
        stem: `img${i}`,
        media_file: `img${i}.jpg`,
        media_type: 'image',
        caption_file: `img${i}.txt`,
        caption_content: `caption ${i}`,
        masked_caption_content: null,
        lyrics_file: null,
        lyrics_content: '',
        metadata: { enabled: true, width: 512, height: 512 },
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
            [datasetKind]="'standard'"
            (detailRequested)="emitted.push($event)"/>
    `,
})
class Host {
    pairs = signal<DatasetPair[]>([]);
    emitted: number[] = [];
}

function render(pairs: DatasetPair[]) {
    TestBed.resetTestingModule();
    TestBed.configureTestingModule({ imports: [Host] });
    const fixture = TestBed.createComponent(Host);
    fixture.componentInstance.pairs.set(pairs);
    fixture.detectChanges();
    return fixture;
}

function tiles(fixture: { nativeElement: HTMLElement }): HTMLElement[] {
    return Array.from(fixture.nativeElement.querySelectorAll<HTMLElement>('.tile'));
}

function loaderCount(fixture: { nativeElement: HTMLElement }): number {
    return fixture.nativeElement.querySelectorAll('.grid-thumb-loader').length;
}

function markVisible(fixture: { nativeElement: HTMLElement }, indices: number[]): void {
    const all = tiles(fixture);
    const io = ios[ios.length - 1];
    io.cb(all.map((el, i) => ({ target: el, isIntersecting: indices.includes(i) } as unknown as IntersectionObserverEntry)));
}

describe('viewer-grid-view — loader bounded to visible tiles', () => {
    beforeEach(() => { ios = []; installFakeIO(); });
    afterEach(() => {
        if (realIO === undefined) delete (globalThis as Record<string, unknown>)['IntersectionObserver'];
        else (globalThis as Record<string, unknown>)['IntersectionObserver'] = realIO;
    });

    it('renders a loader only for tiles reported on screen', async () => {
        const fixture = render([0, 1, 2, 3].map(i => makePair(i)));
        await Promise.resolve(); // the re-observe effect defers to a microtask
        fixture.detectChanges();

        markVisible(fixture, [1, 2]);
        fixture.detectChanges();

        expect(loaderCount(fixture)).toBe(2);
    });

    it('renders no loader at all while nothing has been reported visible', async () => {
        const fixture = render([0, 1, 2].map(i => makePair(i)));
        await Promise.resolve();
        fixture.detectChanges();

        // 3 tiles x 3 dots is 9 infinite animations that nothing would stop:
        // the media is lazy, so an off-screen tile never fires `load`.
        expect(loaderCount(fixture)).toBe(0);
    });

    it('drops the loader again when a visible tile reports load', async () => {
        const fixture = render([makePair(0)]);
        await Promise.resolve();
        fixture.detectChanges();
        markVisible(fixture, [0]);
        fixture.detectChanges();
        expect(loaderCount(fixture)).toBe(1);

        const img = fixture.nativeElement.querySelector('.tile img') as HTMLImageElement;
        img.dispatchEvent(new Event('load'));
        fixture.detectChanges();

        expect(loaderCount(fixture)).toBe(0);
    });

    it('observes the scroll host, not the document, and re-observes on a list change', async () => {
        const fixture = render([0, 1].map(i => makePair(i)));
        await Promise.resolve();
        fixture.detectChanges();
        expect(ios.length).toBe(1);
        expect(ios[0].observed.length).toBe(2);

        fixture.componentInstance.pairs.set([0, 1, 2].map(i => makePair(i)));
        fixture.detectChanges();
        await Promise.resolve();
        fixture.detectChanges();

        expect(ios.length).toBe(1);
        expect(ios[0].observed.length).toBe(3);
    });

    it('falls back to showing every loader when IntersectionObserver is absent', async () => {
        delete (globalThis as Record<string, unknown>)['IntersectionObserver'];
        const fixture = render([0, 1].map(i => makePair(i)));
        await Promise.resolve();
        fixture.detectChanges();

        // Degrade to the pre-existing behaviour, never to a blank tile.
        expect(loaderCount(fixture)).toBe(2);
        installFakeIO();
    });
});

describe('viewer-grid-view — data-index cannot drift from the emitted index', () => {
    beforeEach(() => { ios = []; installFakeIO(); });
    afterEach(() => {
        if (realIO === undefined) delete (globalThis as Record<string, unknown>)['IntersectionObserver'];
        else (globalThis as Record<string, unknown>)['IntersectionObserver'] = realIO;
    });

    it('emits the tile\'s own data-index for every tile, and it resolves to that tile\'s media_file', () => {
        const pairs = [0, 1, 2, 3, 4].map(i => makePair(i));
        const fixture = render(pairs);
        const host = fixture.componentInstance;

        const all = tiles(fixture);
        expect(all.length).toBe(5);

        all.forEach((tile, position) => {
            host.emitted = [];
            (tile.querySelector('.h-80') as HTMLElement).click();
            fixture.detectChanges();

            const emitted = host.emitted[0];
            // The three must agree: DOM position, the data-index attribute
            // the observer keys on, and the index handed to browse-mode.
            expect(emitted).toBe(position);
            expect(tile.getAttribute('data-index')).toBe(String(position));
            // And that index must land on THIS tile's item in `pairs`, which
            // is what browse-mode resolves back to an unfiltered index.
            expect(pairs[emitted].media_file).toBe(tile.getAttribute('data-media-file'));
        });
    });

    it('keeps the agreement on a FILTERED list, where index != item id', () => {
        // browse-mode hands the grid `visiblePairs`, not the whole dataset.
        // Tile 0 here is `img7.jpg`; an off-by-one maps every action to the
        // neighbouring image and nothing in the UI says so.
        const pairs = [7, 3, 11].map(i => makePair(i));
        const fixture = render(pairs);
        const host = fixture.componentInstance;
        const all = tiles(fixture);

        (all[2].querySelector('.h-80') as HTMLElement).click();
        fixture.detectChanges();

        expect(host.emitted).toEqual([2]);
        expect(pairs[host.emitted[0]].media_file).toBe('img11.jpg');
        expect(all[2].getAttribute('data-index')).toBe('2');
        expect(all[2].getAttribute('data-media-file')).toBe('img11.jpg');
    });

    it('re-numbers data-index when the list shrinks, so a stale index cannot survive a filter change', () => {
        const fixture = render([7, 3, 11].map(i => makePair(i)));
        fixture.componentInstance.pairs.set([3, 11].map(i => makePair(i)));
        fixture.detectChanges();

        const all = tiles(fixture);
        expect(all.map(t => t.getAttribute('data-index'))).toEqual(['0', '1']);
        expect(all.map(t => t.getAttribute('data-media-file'))).toEqual(['img3.jpg', 'img11.jpg']);
    });
});
