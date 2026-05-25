import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';

/**
 * Browse folder modal — stub.
 *
 * The full design (modals-more.jsx → BrowseFolderModal) renders a
 * server-side folder picker with breadcrumb navigation. That requires a
 * `GET /system/browse?path=…` endpoint (returns the path's children with
 * size + mtime) which doesn't exist yet.
 *
 * For now we render a backend-aware placeholder. The existing
 * `server-control` component invokes a native picker via
 * `/filesystem/pick-folder` (Tk dialog server-side); the cross-platform
 * web-native picker that this modal will replace it with needs its own
 * endpoint.
 *
 * TODO(backend): GET /system/browse server-side path picker.
 */
@Component({
    selector: 'app-modal-browse-folder',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">FILESYSTEM</div>
                <div class="modal-title">Browse Folder</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>
        <div class="modal-body">
            <p class="muted">
                The web-native folder picker needs a
                <code class="mono">GET /system/browse?path=…</code> endpoint on the
                backend to list directory contents. Until that lands, use the
                native picker from Server Control (it calls
                <code class="mono">/filesystem/pick-folder</code> which opens a
                Tk dialog server-side).
            </p>
            <!-- TODO(backend): GET /system/browse server-side path picker -->
        </div>
        <div class="modal-foot">
            <button class="btn ghost" type="button" (click)="overlay.closeModal()">Close</button>
        </div>
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .muted { color: var(--color-text-muted); font-size: 13px; line-height: 1.6; }
        .mono { font-family: var(--f-mono); padding: 1px 6px; background: var(--color-surface-mid); border-radius: 3px; }
    `],
})
export class BrowseFolderModalComponent {
    protected overlay = inject(OverlayStore);
}
