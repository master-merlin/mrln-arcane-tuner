import { TestBed } from '@angular/core/testing';
import { UxStore } from './ux.store';

const KEY = 'mrln.ux';
const attr = () => document.documentElement.getAttribute('data-ux');

describe('UxStore (closing round: V2 is the only rendering)', () => {
    function make(): UxStore {
        TestBed.configureTestingModule({ providers: [UxStore] });
        return TestBed.inject(UxStore);
    }

    beforeEach(() => {
        localStorage.removeItem(KEY);
        document.documentElement.removeAttribute('data-ux');
    });

    afterEach(() => {
        localStorage.removeItem(KEY);
        document.documentElement.removeAttribute('data-ux');
    });

    it('sets data-ux="v2" with no stored key', () => {
        const store = make();
        expect(store.v2(), 'LANE-143: v2() is true on a fresh profile').toBe(true);
        expect(attr(), 'LANE-143: data-ux is "v2" regardless of mrln.ux').toBe('v2');
    });

    for (const stored of ['v1', 'v2', 'off', '']) {
        it(`ignores a stored mrln.ux=${JSON.stringify(stored)} and leaves it in place`, () => {
            localStorage.setItem(KEY, stored);
            const store = make();
            expect(attr(), 'LANE-143: data-ux is "v2" regardless of mrln.ux').toBe('v2');
            expect(store.v2(), 'LANE-143: v2() is true regardless of mrln.ux').toBe(true);
            expect(localStorage.getItem(KEY), 'LANE-143: a stored mrln.ux is left in place, not rewritten').toBe(stored);
        });
    }

    it('re-asserts data-ux="v2" even if the attribute was removed before bootstrap', () => {
        document.documentElement.setAttribute('data-ux', 'v1');
        make();
        expect(attr(), 'LANE-143: data-ux is "v2" regardless of mrln.ux').toBe('v2');
    });

    it('has no toggle() any more and set() never writes mrln.ux', () => {
        const store = make();
        expect((store as unknown as { toggle?: unknown }).toggle, 'LANE-143: the V1/V2 toggle is retired').toBeUndefined();
        store.set(false);
        expect(attr(), 'LANE-143: set(false) is the in-memory V1 seam (tests only)').toBeNull();
        expect(localStorage.getItem(KEY), 'LANE-143: set() never persists').toBeNull();
        store.set(true);
        expect(attr()).toBe('v2');
        expect(localStorage.getItem(KEY), 'LANE-143: set() never persists').toBeNull();
    });
});
