import { Injectable, signal } from '@angular/core';

/**
 * The user's color theme: `dark` (the default/original look) or `light`.
 *
 * Persisted to `localStorage` under `mrln.theme` and reflected onto the
 * `<html data-theme="…">` attribute, which the `[data-theme="light"]` block in
 * `styles.css` keys off to override the design tokens. Mirrors the
 * `scope.store` persistence pattern (`mrln.scope`).
 *
 * A matching inline guard in `index.html` sets `data-theme` before first paint
 * so a returning light-mode user never sees a flash of the dark theme.
 */
export type Theme = 'dark' | 'light';

const STORAGE_KEY = 'mrln.theme';

function hydrate(): Theme {
    try {
        return localStorage.getItem(STORAGE_KEY) === 'light' ? 'light' : 'dark';
    } catch {
        // Private mode / unavailable storage — default to dark.
        return 'dark';
    }
}

@Injectable({ providedIn: 'root' })
export class ThemeStore {
    readonly theme = signal<Theme>(hydrate());

    constructor() {
        // Re-affirm the hydrated value (the inline guard normally set it already,
        // but this keeps the DOM correct in tests / if the guard is absent).
        this.apply(this.theme());
    }

    /** Switch between dark and light, persisting + applying the new value. */
    toggle(): void {
        this.set(this.theme() === 'dark' ? 'light' : 'dark');
    }

    set(theme: Theme): void {
        this.theme.set(theme);
        this.apply(theme);
    }

    private apply(theme: Theme): void {
        try {
            localStorage.setItem(STORAGE_KEY, theme);
        } catch {
            // Ignore write failures (private mode); the in-memory signal still works.
        }
        document.documentElement.setAttribute('data-theme', theme);
    }
}
