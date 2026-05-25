import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetWorkspaceComponent } from '../../workspace/dataset-workspace.component';

/**
 * Fullscreen workspace overlay mount point. Mounts the heavy
 * `DatasetWorkspaceComponent` via `@defer` so the workspace bundle
 * only loads on first open.
 */
@Component({
    selector: 'app-workspace-layer',
    standalone: true,
    imports: [DatasetWorkspaceComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @if (overlay.workspace()) {
            @defer {
                <app-dataset-workspace/>
            }
        }
    `,
})
export class WorkspaceLayerComponent {
    protected overlay = inject(OverlayStore);
}
