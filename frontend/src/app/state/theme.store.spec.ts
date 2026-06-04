import { TestBed } from '@angular/core/testing';
import { ThemeStore } from './theme.store';

const KEY = 'mrln.theme';

describe('ThemeStore', () => {
    function make(): ThemeStore {
        TestBed.configureTestingModule({ providers: [ThemeStore] });
        return TestBed.inject(ThemeStore);
    }

    beforeEach(() => {
        localStorage.removeItem(KEY);
        document.documentElement.removeAttribute('data-theme');
    });

    afterEach(() => {
        localStorage.removeItem(KEY);
        document.documentElement.removeAttribute('data-theme');
    });

    it('defaults to dark when nothing is stored', () => {
        const store = make();
        expect(store.theme()).toBe('dark');
        expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    });

    it('hydrates light from localStorage', () => {
        localStorage.setItem(KEY, 'light');
        const store = make();
        expect(store.theme()).toBe('light');
        expect(document.documentElement.getAttribute('data-theme')).toBe('light');
    });

    it('ignores an unknown stored value and falls back to dark', () => {
        localStorage.setItem(KEY, 'sepia');
        const store = make();
        expect(store.theme()).toBe('dark');
    });

    it('toggle flips dark -> light, persists, and reflects on <html>', () => {
        const store = make();
        store.toggle();
        expect(store.theme()).toBe('light');
        expect(localStorage.getItem(KEY)).toBe('light');
        expect(document.documentElement.getAttribute('data-theme')).toBe('light');
    });

    it('toggle flips light -> dark, persists, and reflects on <html>', () => {
        localStorage.setItem(KEY, 'light');
        const store = make();
        store.toggle();
        expect(store.theme()).toBe('dark');
        expect(localStorage.getItem(KEY)).toBe('dark');
        expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    });
});
