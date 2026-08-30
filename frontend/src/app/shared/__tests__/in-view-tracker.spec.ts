/**
 * The in-view tracker exists because of a measured defect, and these
 * specs pin the parts of it the measurement proved were needed.
 *
 * A 263-item workspace ran **1503 CSS animations at once** (789 filmstrip
 * spinner dots + 714 grid loader dots) and rendered at 35.5 ms / 28 fps
 * while completely idle. Two properties matter and both are asserted
 * here rather than assumed:
 *
 *  1. A load event is NOT a sufficient bound. The images are
 *     `loading="lazy"`, so an off-screen image never loads; 34 of 263 had
 *     loaded and 687 animations would have survived a load-only gate.
 *     Visibility is the bound, which is why this module exists at all.
 *  2. Where `IntersectionObserver` is absent the tracker must answer
 *     "visible" for everything. A missing observer may degrade to
 *     showing a spinner that is not needed; it must never degrade to
 *     hiding one that is.
 */
import { createInViewTracker } from '../in-view-tracker';

type Cb = (entries: IntersectionObserverEntry[]) => void;

interface FakeIO {
    cb: Cb;
    root: unknown;
    rootMargin: string | undefined;
    observed: HTMLElement[];
    disconnects: number;
}

let instances: FakeIO[] = [];
let realIO: unknown;

function installFakeIO(): void {
    realIO = (globalThis as Record<string, unknown>)['IntersectionObserver'];
    class Fake {
        constructor(cb: Cb, opts?: IntersectionObserverInit) {
            this.rec = { cb, root: opts?.root, rootMargin: opts?.rootMargin, observed: [], disconnects: 0 };
            instances.push(this.rec);
        }
        private rec: FakeIO;
        observe(el: HTMLElement): void { this.rec.observed.push(el); }
        unobserve(): void { /* unused */ }
        disconnect(): void { this.rec.disconnects++; this.rec.observed = []; }
        takeRecords(): [] { return []; }
    }
    (globalThis as Record<string, unknown>)['IntersectionObserver'] = Fake;
}

function restoreIO(): void {
    if (realIO === undefined) delete (globalThis as Record<string, unknown>)['IntersectionObserver'];
    else (globalThis as Record<string, unknown>)['IntersectionObserver'] = realIO;
}

function entry(el: HTMLElement, isIntersecting: boolean): IntersectionObserverEntry {
    return { target: el, isIntersecting } as unknown as IntersectionObserverEntry;
}

function rootWith(indices: (number | null)[]): HTMLElement {
    const root = document.createElement('div');
    for (const i of indices) {
        const child = document.createElement('div');
        child.className = 'cell';
        if (i !== null) child.dataset['index'] = String(i);
        root.appendChild(child);
    }
    return root;
}

describe('createInViewTracker', () => {
    beforeEach(() => { instances = []; installFakeIO(); });
    afterEach(() => { restoreIO(); });

    it('creates no observer until refresh is given a live root', () => {
        const t = createInViewTracker({ selector: '.cell[data-index]' });
        expect(instances.length).toBe(0);

        t.refresh(null);
        expect(instances.length).toBe(0);

        t.refresh(rootWith([0, 1]));
        expect(instances.length).toBe(1);
    });

    it('reports only the indices the observer says are intersecting', () => {
        const root = rootWith([0, 1, 2]);
        const t = createInViewTracker({ selector: '.cell[data-index]' });
        t.refresh(root);
        const io = instances[0];
        const cells = Array.from(root.querySelectorAll<HTMLElement>('.cell'));

        io.cb([entry(cells[0], true), entry(cells[1], false), entry(cells[2], true)]);

        expect(t.has(0)).toBe(true);
        expect(t.has(1)).toBe(false);
        expect(t.has(2)).toBe(true);
        expect([...t.indices()].sort()).toEqual([0, 2]);
    });

    it('drops an index again when it scrolls back out', () => {
        const root = rootWith([0]);
        const t = createInViewTracker({ selector: '.cell[data-index]' });
        t.refresh(root);
        const io = instances[0];
        const cell = root.querySelector<HTMLElement>('.cell')!;

        io.cb([entry(cell, true)]);
        expect(t.has(0)).toBe(true);

        io.cb([entry(cell, false)]);
        expect(t.has(0)).toBe(false);
    });

    it('ignores an observed element with no data-index instead of mapping it to 0', () => {
        const root = rootWith([null]);
        const t = createInViewTracker({ selector: '.cell' });
        t.refresh(root);
        const cell = root.querySelector<HTMLElement>('.cell')!;

        instances[0].cb([entry(cell, true)]);

        expect(t.indices().size).toBe(0);
    });

    it('reuses one observer across refreshes and re-observes the current children', () => {
        const t = createInViewTracker({ selector: '.cell[data-index]' });
        const root = rootWith([0, 1]);
        t.refresh(root);
        expect(instances.length).toBe(1);
        expect(instances[0].observed.length).toBe(2);

        root.appendChild(Object.assign(document.createElement('div'), { className: 'cell' })).setAttribute('data-index', '2');
        t.refresh(root);

        // One observer total, not one per render — an observer per refresh
        // is exactly how a list leaks callbacks.
        expect(instances.length).toBe(1);
        // Each refresh disconnects first, so nothing is observed twice:
        // two refreshes, two disconnects, one observer.
        expect(instances[0].disconnects).toBe(2);
        expect(instances[0].observed.length).toBe(3);
    });

    it('answers true for every index when IntersectionObserver is unavailable', () => {
        restoreIO();
        delete (globalThis as Record<string, unknown>)['IntersectionObserver'];

        const t = createInViewTracker({ selector: '.cell[data-index]' });
        t.refresh(rootWith([0, 1]));

        expect(t.has(0)).toBe(true);
        expect(t.has(999)).toBe(true);
        installFakeIO();
    });

    it('stops updating after destroy — asserted by effect, not by a flag', () => {
        const root = rootWith([0, 1]);
        const t = createInViewTracker({ selector: '.cell[data-index]' });
        t.refresh(root);
        const io = instances[0];
        const cells = Array.from(root.querySelectorAll<HTMLElement>('.cell'));
        io.cb([entry(cells[0], true)]);
        const before = t.indices();
        expect(before.has(0)).toBe(true);

        t.destroy();
        // A callback already queued still arrives after teardown. It must
        // not write the signal — a signal write here is a change-detection
        // run scheduled for a destroyed component.
        io.cb([entry(cells[1], true)]);

        expect(t.indices()).toBe(before);
        expect(t.indices().has(1)).toBe(false);
        // One disconnect from the refresh, one from destroy.
        expect(io.disconnects).toBe(2);
    });

    it('does not create a new observer after destroy', () => {
        const t = createInViewTracker({ selector: '.cell[data-index]' });
        t.destroy();
        t.refresh(rootWith([0]));
        expect(instances.length).toBe(0);
    });

    it('roots the observer on the scroll container, not the viewport', () => {
        const root = rootWith([0]);
        const t = createInViewTracker({ selector: '.cell[data-index]', rootMargin: '120px' });
        t.refresh(root);

        // Rooting on the document would report every item of a scroller
        // that is itself off screen as "visible" — the bug this bounds.
        expect(instances[0].root).toBe(root);
        expect(instances[0].rootMargin).toBe('120px');
    });
});
