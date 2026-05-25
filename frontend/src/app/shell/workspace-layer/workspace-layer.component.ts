import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';

/**
 * Fullscreen workspace overlay placeholder. Phase 4 swaps the body for
 * the real Browse/Details/Edit modes + filmstrip scrubber. For now this
 * renders a small toolbar so the open/close cycle is verifiable.
 */
@Component({
    selector: 'app-workspace-layer',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @if (overlay.workspace(); as ws) {
            <div class="modal-backdrop" style="padding: 0;">
                <div
                    style="
                        background: var(--color-base);
                        width: 100%; height: 100%;
                        display: flex; flex-direction: column;
                    ">
                    <div class="topbar">
                        <div class="page-title" style="font-size: 14px;">
                            Workspace · {{ ws.datasetId }} · {{ ws.mode }}
                        </div>
                        <button
                            class="btn ghost"
                            style="margin-left: auto;"
                            (click)="overlay.closeWorkspace()"
                            type="button">Close</button>
                    </div>
                    <div class="content">Workspace body wired in Phase 4.</div>
                </div>
            </div>
        }
    `,
})
export class WorkspaceLayerComponent {
    protected overlay = inject(OverlayStore);
}
