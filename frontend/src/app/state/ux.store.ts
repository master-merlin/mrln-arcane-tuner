import { Injectable, signal } from '@angular/core';

/**
 * The v2 design language is the only rendering (LANE-143 closing round).
 *
 * `<html data-ux="v2">` is set unconditionally: first by the pre-paint guard in
 * `index.html` (so nothing flashes), then re-asserted here at bootstrap.
 * `styles/ux-v2.css` keys every rule off that attribute. The former
 * `localStorage['mrln.ux']` switch is no longer read; a stored value is ignored
 * and left in place. V1 is deprecated: it is reachable only by removing the
 * attribute by hand. `set()` is an in-memory seam (it never persists) so specs
 * can still prove the `v2()` gates; no UI calls it.
 */
@Injectable({ providedIn: 'root' })
export class UxStore {
    readonly v2 = signal<boolean>(true);

    constructor() {
        this.apply(true);
    }

    set(on: boolean): void {
        this.v2.set(on);
        this.apply(on);
    }

    private apply(on: boolean): void {
        if (on) document.documentElement.setAttribute('data-ux', 'v2');
        else document.documentElement.removeAttribute('data-ux');
    }
}
