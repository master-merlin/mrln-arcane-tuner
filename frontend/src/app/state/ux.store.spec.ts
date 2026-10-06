import { TestBed } from '@angular/core/testing';
import { UxStore } from './ux.store';

const KEY = 'mrln.ux';
const attr = () => document.documentElement.getAttribute('data-ux');

describe('UxStore', () => {
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

    it('defaults to OFF: no key, no attribute', () => {
        const store = make();
        expect(store.v2(), 'LANE-143: v2() must be false with no stored key').toBe(false);
        expect(attr(), 'LANE-143: <html data-ux> must be absent with no stored key').toBeNull();
    });

    it('hydrates v2 from localStorage and sets the attribute', () => {
        localStorage.setItem(KEY, 'v2');
        const store = make();
        expect(store.v2()).toBe(true);
        expect(attr(), 'LANE-143: <html data-ux> must be "v2" when hydrated from storage').toBe('v2');
    });

    it('treats an unknown stored value as OFF', () => {
        localStorage.setItem(KEY, 'v1');
        const store = make();
        expect(store.v2()).toBe(false);
        expect(attr()).toBeNull();
    });

    it('toggle() persists v2 and sets the attribute; a second toggle() clears both', () => {
        const store = make();
        store.toggle();
        expect(store.v2()).toBe(true);
        expect(localStorage.getItem(KEY)).toBe('v2');
        expect(attr(), 'LANE-143: <html data-ux> must be "v2" after toggle()').toBe('v2');
        store.toggle();
        expect(store.v2()).toBe(false);
        expect(localStorage.getItem(KEY), 'LANE-143: the key must be removed on switching off').toBeNull();
        expect(attr(), 'LANE-143: <html data-ux> must be removed on switching off').toBeNull();
    });
});
