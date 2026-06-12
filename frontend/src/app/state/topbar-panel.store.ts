import { Injectable, computed, signal } from '@angular/core';

/**
 * Tracks which single topbar dropdown is currently open.
 *
 * The topbar has several independent right-aligned dropdown panels
 * (download indicator, notifications, task center) that are each
 * `position: absolute; right: 0`. Without coordination they overlap when
 * more than one is open. This store enforces a single open panel: opening
 * one closes any other.
 *
 * Each panel picks a unique id and drives its visibility through
 * `isOpen(id)` / `toggle(id)` instead of a local `open` signal.
 */
export type TopbarPanelId = 'downloads' | 'notifications' | 'tasks' | 'updates';

@Injectable({ providedIn: 'root' })
export class TopbarPanelStore {
    /** The currently open panel, or `null` when all are closed. */
    private readonly openPanel = signal<TopbarPanelId | null>(null);

    /** A signal that is `true` only while the given panel is the open one. */
    isOpen(id: TopbarPanelId) {
        return computed(() => this.openPanel() === id);
    }

    /** Open `id` if closed; close it if already open. Opening closes others. */
    toggle(id: TopbarPanelId): void {
        this.openPanel.update(cur => (cur === id ? null : id));
    }

    /** Close `id` if it is the open panel (no-op otherwise). */
    close(id: TopbarPanelId): void {
        this.openPanel.update(cur => (cur === id ? null : cur));
    }
}
