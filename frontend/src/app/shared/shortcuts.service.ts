import { DestroyRef, Injectable, inject } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { fromEvent } from 'rxjs';
import { OverlayStore } from '../state/overlay.store';

/**
 * Process-global keyboard shortcuts.
 *
 * - `Esc` — close the top modal; if no modal, close the workspace overlay.
 * - `g` / `d` / `e` — when a workspace is open, switch its mode.
 * - `←` / `→` — filmstrip nav (wired locally by the scrubber, not here).
 * - `⌘K` / `Ctrl+K` — TODO(frontend): command palette.
 *
 * Installed once from the shell component's `ngOnInit`. The subscription
 * tears down with the service's host injector via `takeUntilDestroyed`.
 */
@Injectable({ providedIn: 'root' })
export class GlobalShortcutsService {
    private overlay = inject(OverlayStore);
    private destroyRef = inject(DestroyRef);
    private installed = false;

    install(): void {
        if (this.installed) return;
        this.installed = true;
        fromEvent<KeyboardEvent>(window, 'keydown')
            .pipe(takeUntilDestroyed(this.destroyRef))
            .subscribe(e => this.handle(e));
    }

    private handle(e: KeyboardEvent): void {
        // Don't intercept keys while the user is typing.
        const target = e.target as HTMLElement | null;
        if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) {
            return;
        }

        if (e.key === 'Escape') {
            if (this.overlay.modalStack().length > 0) {
                this.overlay.closeModal();
                e.preventDefault();
            } else if (this.overlay.workspace()) {
                this.overlay.closeWorkspace();
                e.preventDefault();
            }
            return;
        }

        const ws = this.overlay.workspace();
        if (ws) {
            if (e.key === 'g') { this.overlay.setWorkspaceMode('browse'); e.preventDefault(); return; }
            if (e.key === 'd') { this.overlay.setWorkspaceMode('details'); e.preventDefault(); return; }
            if (e.key === 'e') { this.overlay.setWorkspaceMode('edit'); e.preventDefault(); return; }
        }

        // TODO(frontend): ⌘K / Ctrl+K command palette — wire when palette UI lands.
    }
}
