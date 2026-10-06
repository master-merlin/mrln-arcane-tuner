import { Injectable, signal } from '@angular/core';

/**
 * The runtime switch for the v2 design language (LANE-143).
 *
 * Persisted under `localStorage['mrln.ux']` (`'v2'` or absent) and reflected as
 * `<html data-ux="v2">`, which `styles/ux-v2.css` keys every rule off. Default
 * OFF: no key means no attribute, so the app renders exactly as before.
 * Mirrors `ThemeStore`.
 */
const STORAGE_KEY = 'mrln.ux';

function hydrate(): boolean {
    try {
        return localStorage.getItem(STORAGE_KEY) === 'v2';
    } catch {
        // Private mode / unavailable storage — default to OFF.
        return false;
    }
}

@Injectable({ providedIn: 'root' })
export class UxStore {
    readonly v2 = signal<boolean>(hydrate());

    constructor() {
        this.apply(this.v2());
    }

    toggle(): void {
        this.set(!this.v2());
    }

    set(on: boolean): void {
        this.v2.set(on);
        this.apply(on);
    }

    private apply(on: boolean): void {
        try {
            if (on) localStorage.setItem(STORAGE_KEY, 'v2');
            else localStorage.removeItem(STORAGE_KEY);
        } catch {
            // Ignore write failures; the in-memory signal still works.
        }
        if (on) document.documentElement.setAttribute('data-ux', 'v2');
        else document.documentElement.removeAttribute('data-ux');
    }
}
