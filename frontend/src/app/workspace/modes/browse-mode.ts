import { ChangeDetectionStrategy, Component, inject, input } from '@angular/core';
import { ViewerGridViewComponent } from '../../components/dataset/dataset-viewer/components/viewer-grid-view';
import { OverlayStore } from '../../state/overlay.store';
import { RuntimeConfigService } from '../../services/runtime-config.service';

/**
 * Browse mode — grid view of the dataset, ported design wraps the existing
 * `app-viewer-grid-view`. The grid component is reused as-is from the
 * orphan tree; this wrapper only resolves the URL bases and forwards the
 * pairs/dataset-name pair so the workspace can stay declarative.
 *
 * The grid emits `massCaptionRequested` / `massMaskingRequested` /
 * `massEditRequested` — the workspace toolbar in `DatasetWorkspace` also
 * exposes those actions, but routing them through here lets the existing
 * per-image hover actions inside the grid (open detail, crop, edit) work
 * unchanged.
 */
@Component({
    selector: 'app-workspace-browse',
    standalone: true,
    imports: [ViewerGridViewComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <app-viewer-grid-view
            [pairs]="pairs()"
            [datasetName]="datasetName()"
            [mediaBaseUrl]="rtc.mediaBaseUrl"
            [apiUrl]="rtc.apiUrl"
            (detailRequested)="openDetail($event)"
            (editRequested)="openEdit($event)"
            (massCaptionRequested)="overlay.openModal('mass-caption', { datasetName: datasetName() })"
            (massMaskingRequested)="overlay.openModal('mass-mask', { datasetName: datasetName() })"
            (massEditRequested)="overlay.openModal('mass-edit', { datasetName: datasetName() })"/>
    `,
    styles: [`
        :host { display: block; height: 100%; overflow: hidden; }
    `],
})
export class BrowseMode {
    datasetId = input.required<string>();
    /** Pairs loaded from the dataset's `/pairs` endpoint (parent fetches). */
    pairs = input.required<any[]>();
    /** HTTP-name of the dataset (URL slug). */
    datasetName = input.required<string>();

    protected overlay = inject(OverlayStore);
    protected rtc = inject(RuntimeConfigService);

    protected openDetail(idx: number): void {
        this.overlay.setWorkspaceImage(idx);
        this.overlay.setWorkspaceMode('details');
    }

    protected openEdit(idx: number): void {
        this.overlay.setWorkspaceImage(idx);
        this.overlay.setWorkspaceMode('edit');
    }
}
