/**
 * The filmstrip's loading dots are an `animation: … infinite`, one per
 * cell, three elements each. They were only ever COVERED by the thumbnail
 * — the old template comment said so — and covering an animation does not
 * stop it.
 *
 * Measured on a 263-item dataset (production build): 789 filmstrip
 * animations running permanently, and the whole app rendered at 35.5 ms /
 * 28 fps *while idle*. Removing just this component's animations took the
 * same measurement to 17.7 ms / 56 fps. Neither
 * `content-visibility: auto` on the cell nor `display: none` on the
 * spinner stopped them — both were measured and both did nothing, which
 * is why the gate is `@if` (removal from the DOM) and not a CSS hide.
 *
 * A load-only gate is also not enough and that is measured too: the
 * thumbnails are `loading="lazy"`, so 229 of 263 images had never started
 * loading and would have kept their dots forever.
 */
import { TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { FilmstripScrubberComponent } from '../filmstrip-scrubber.component';
import { OverlayStore } from '../../../state/overlay.store';

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

function images(n: number) {
    return Array.from({ length: n }, (_, i) => ({
        thumbnailUrl: `/api/thumb/${i}`,
        harmonized: false,
        captioned: false,
        masked: false,
        mediaType: 'image',
    }));
}

@Component({
    standalone: true,
    imports: [FilmstripScrubberComponent],
    template: `<app-filmstrip-scrubber [images]="imgs()" [activeIndex]="active()"/>`,
})
class Host {
    imgs = signal(images(4));
    active = signal(0);
}

function render(n = 4) {
    TestBed.resetTestingModule();
    TestBed.configureTestingModule({ imports: [Host], providers: [OverlayStore] });
    const fixture = TestBed.createComponent(Host);
    fixture.componentInstance.imgs.set(images(n));
    fixture.detectChanges();
    return fixture;
}

function spinners(fixture: { nativeElement: HTMLElement }): number {
    return fixture.nativeElement.querySelectorAll('.thumb-spinner').length;
}

function cells(fixture: { nativeElement: HTMLElement }): HTMLElement[] {
    return Array.from(fixture.nativeElement.querySelectorAll<HTMLElement>('.cell'));
}

function markVisible(fixture: { nativeElement: HTMLElement }, indices: number[]): void {
    const all = cells(fixture);
    const io = ios[ios.length - 1];
    io.cb(all.map((el, i) => ({ target: el, isIntersecting: indices.includes(i) } as unknown as IntersectionObserverEntry)));
}

describe('filmstrip-scrubber loading dots', () => {
    beforeEach(() => { ios = []; installFakeIO(); });
    afterEach(() => {
        if (realIO === undefined) delete (globalThis as Record<string, unknown>)['IntersectionObserver'];
        else (globalThis as Record<string, unknown>)['IntersectionObserver'] = realIO;
    });

    it('renders no spinner for cells that are off screen', async () => {
        const fixture = render(6);
        await Promise.resolve();
        fixture.detectChanges();

        expect(cells(fixture).length).toBe(6);
        expect(spinners(fixture)).toBe(0);
    });

    it('renders a spinner only for the cells reported on screen', async () => {
        const fixture = render(6);
        await Promise.resolve();
        fixture.detectChanges();

        markVisible(fixture, [0, 1, 2]);
        fixture.detectChanges();

        expect(spinners(fixture)).toBe(3);
    });

    it('REMOVES the spinner from the DOM when the image loads — covering it is not stopping it', async () => {
        const fixture = render(2);
        await Promise.resolve();
        fixture.detectChanges();
        markVisible(fixture, [0, 1]);
        fixture.detectChanges();
        expect(spinners(fixture)).toBe(2);

        const img = fixture.nativeElement.querySelectorAll('.cell img')[0] as HTMLImageElement;
        img.dispatchEvent(new Event('load'));
        fixture.detectChanges();

        expect(spinners(fixture)).toBe(1);
    });

    it('keeps the error glyph path — a failed thumbnail never gets dots back', async () => {
        const fixture = render(1);
        await Promise.resolve();
        fixture.detectChanges();
        markVisible(fixture, [0]);
        fixture.detectChanges();

        const img = fixture.nativeElement.querySelector('.cell img') as HTMLImageElement;
        img.dispatchEvent(new Event('error'));
        fixture.detectChanges();

        expect(spinners(fixture)).toBe(0);
        expect(fixture.nativeElement.querySelectorAll('.thumb-error').length).toBe(1);
    });

    it('still renders the thumbnail itself for off-screen cells (only the animation is bounded)', async () => {
        const fixture = render(5);
        await Promise.resolve();
        fixture.detectChanges();

        // Bounding the spinner must not turn the strip into empty boxes;
        // the lazy <img> stays so the browser can fetch it on approach.
        expect(fixture.nativeElement.querySelectorAll('.cell img').length).toBe(5);
        expect(spinners(fixture)).toBe(0);
    });

    it('observes every cell and re-observes when the image list changes', async () => {
        const fixture = render(3);
        await Promise.resolve();
        fixture.detectChanges();
        expect(ios.length).toBe(1);
        expect(ios[0].observed.length).toBe(3);

        fixture.componentInstance.imgs.set(images(5));
        fixture.detectChanges();
        await Promise.resolve();
        fixture.detectChanges();

        expect(ios.length).toBe(1);
        expect(ios[0].observed.length).toBe(5);
    });

    it('shows every spinner when IntersectionObserver is unavailable', async () => {
        delete (globalThis as Record<string, unknown>)['IntersectionObserver'];
        const fixture = render(3);
        await Promise.resolve();
        fixture.detectChanges();

        expect(spinners(fixture)).toBe(3);
        installFakeIO();
    });
});
