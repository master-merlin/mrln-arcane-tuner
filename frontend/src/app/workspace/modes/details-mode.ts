import { ChangeDetectionStrategy, Component, computed, inject, input, model } from '@angular/core';
import { DetailMaskingSidebarComponent } from '../../components/dataset/dataset-viewer/components/detail-masking-sidebar';
import { DetailMediaContainerComponent } from '../../components/dataset/dataset-viewer/components/detail-media-container';
import { DetailCaptionSidebarComponent } from '../../components/dataset/dataset-viewer/components/detail-caption-sidebar';
import { OverlayStore } from '../../state/overlay.store';
import { RuntimeConfigService } from '../../services/runtime-config.service';

/**
 * Details mode — 3-pane layout (mask LEFT 320px / canvas / caption RIGHT
 * 320px) wrapping the orphan-tree detail components verbatim.
 *
 * The parent workspace resolves the current pair from `pairs[imageIndex]`
 * and passes it in. Prev/Next navigation flows through the workspace's
 * `imageIndex` cursor (which the filmstrip scrubber also drives).
 */
@Component({
    selector: 'app-workspace-details',
    standalone: true,
    imports: [
        DetailMaskingSidebarComponent,
        DetailMediaContainerComponent,
        DetailCaptionSidebarComponent,
    ],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @if (currentPair(); as pair) {
            <div class="details-grid">
                <aside class="pane mask">
                    <app-detail-masking-sidebar
                        [currentPair]="pair"
                        [datasetName]="datasetName()"
                        [mediaBaseUrl]="rtc.mediaBaseUrl"
                        (openMaskPreviewRequested)="openMaskPreview()"/>
                </aside>
                <main class="pane canvas">
                    <app-detail-media-container
                        [currentPair]="pair"
                        [datasetName]="datasetName()"
                        [mediaBaseUrl]="rtc.mediaBaseUrl"
                        [apiUrl]="rtc.apiUrl"
                        (prevRequested)="prev()"
                        (nextRequested)="next()"/>
                </main>
                <aside class="pane caption">
                    <app-detail-caption-sidebar
                        [datasetName]="datasetName()"
                        [currentPair]="pair"
                        [(captionText)]="captionText"/>
                </aside>
            </div>
        } @else {
            <div class="empty">No image at index {{ imageIndex() }}.</div>
        }
    `,
    styles: [`
        :host { display: block; height: 100%; overflow: hidden; }
        .details-grid {
            display: grid;
            grid-template-columns: 340px 1fr 340px;
            height: 100%;
            overflow: hidden;
        }
        .pane {
            min-height: 0;
            overflow-y: auto;
            overflow-x: hidden;
        }
        .pane.mask { border-right: 1px solid var(--color-border-subtle); background: var(--color-surface-low); }
        .pane.caption { border-left: 1px solid var(--color-border-subtle); background: var(--color-surface-low); }
        .pane.canvas { background: var(--color-base); display: flex; flex-direction: column; }
        .empty {
            display: flex; align-items: center; justify-content: center;
            height: 100%;
            color: var(--color-text-muted);
            font-size: 13px;
        }
    `],
})
export class DetailsMode {
    datasetId = input.required<string>();
    imageIndex = input.required<number>();
    /** Pairs array from parent. */
    pairs = input.required<any[]>();
    /** HTTP-name of the dataset. */
    datasetName = input.required<string>();

    protected overlay = inject(OverlayStore);
    protected rtc = inject(RuntimeConfigService);

    /** Caption editor is two-way bound; persistence is handled by the existing sidebar internals. */
    protected captionText = model<string>('');

    protected currentPair = computed(() => {
        const list = this.pairs();
        const idx = this.imageIndex();
        return idx >= 0 && idx < list.length ? list[idx] : null;
    });

    protected prev(): void {
        const idx = this.imageIndex();
        if (idx > 0) this.overlay.setWorkspaceImage(idx - 1);
    }

    protected next(): void {
        const idx = this.imageIndex();
        if (idx < this.pairs().length - 1) this.overlay.setWorkspaceImage(idx + 1);
    }

    protected openMaskPreview(): void {
        this.overlay.openModal('mask-preview', {
            datasetName: this.datasetName(),
            pair: this.currentPair(),
        });
    }
}
