import { ChangeDetectionStrategy, Component, computed, effect, inject, input } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService } from '../../services/dataset';
import { PipelineEditorState } from './edit/pipeline-editor.state';
import { PreviewPipeline } from './edit/preview/preview-pipeline';
import { EditLeftPanelComponent } from './edit/components/edit-left-panel.component';
import { EditCanvasComponent } from './edit/components/edit-canvas.component';
import { EditRightPanelComponent } from './edit/components/edit-right-panel.component';

/**
 * Edit mode — non-destructive image editor. 3-pane shell (340/1fr/340)
 * matching the Hi-Fi design's `EditBody`. Provides PipelineEditorState
 * scoped to this component so working state dies with the mode.
 */
@Component({
    selector: 'app-workspace-edit',
    standalone: true,
    imports: [EditLeftPanelComponent, EditCanvasComponent, EditRightPanelComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    providers: [PipelineEditorState, PreviewPipeline],
    template: `
        @if (currentPair(); as pair) {
            <div class="edit-grid">
                <aside class="pane left">
                    <app-edit-left-panel
                        [datasetName]="datasetName()"
                        [mediaFile]="pair.media_file"
                        [width]="pair.metadata?.width"
                        [height]="pair.metadata?.height"/>
                </aside>
                <main class="pane center">
                    <app-edit-canvas
                        [datasetName]="datasetName()"
                        [mediaFile]="pair.media_file"
                        [hasOverlay]="!!pair?.metadata?.has_overlay"
                        (prev)="onPrev()"
                        (next)="onNext()"/>
                </main>
                <aside class="pane right">
                    <app-edit-right-panel
                        [datasetName]="datasetName()"
                        [mediaFile]="pair.media_file"/>
                </aside>
            </div>
        } @else {
            <div class="edit-empty">No image at index {{ imageIndex() }}.</div>
        }
    `,
    styles: [`
        :host { display: block; height: 100%; overflow: hidden; }
        .edit-grid {
            display: grid;
            grid-template-columns: 340px 1fr 340px;
            height: 100%;
            overflow: hidden;
        }
        .pane { min-height: 0; overflow: hidden; }
        .pane.left  { border-right: 1px solid var(--color-border-subtle); background: var(--color-surface-low); overflow-y: auto; overflow-x: hidden; }
        .pane.right { border-left: 1px solid var(--color-border-subtle); background: var(--color-surface-low); overflow: hidden; display: flex; flex-direction: column; }
        .pane.center { background: var(--color-base); display: flex; flex-direction: column; overflow: hidden; }
        .edit-empty {
            display: flex; align-items: center; justify-content: center;
            height: 100%; color: var(--color-text-muted); font-size: 13px;
        }
    `],
})
export class EditMode {
    datasetId = input.required<string>();
    imageIndex = input.required<number>();
    pairs = input<any[]>([]);
    datasetName = input.required<string>();

    protected overlay = inject(OverlayStore);
    protected datasets = inject(DatasetService);
    private state = inject(PipelineEditorState);

    protected currentPair = computed(() => {
        const list = this.pairs();
        const idx = this.imageIndex();
        return idx >= 0 && idx < list.length ? list[idx] : null;
    });

    protected onPrev(): void {
        const idx = this.imageIndex();
        if (idx > 0) this.overlay.setWorkspaceImage(idx - 1);
    }

    protected onNext(): void {
        const idx = this.imageIndex();
        if (idx < this.pairs().length - 1) this.overlay.setWorkspaceImage(idx + 1);
    }

    private identity = computed<string | null>(() => {
        const p = this.currentPair();
        return p?.media_file ? `${this.datasetName()}/${p.media_file}` : null;
    });

    constructor() {
        let lastIdentity = '';
        let pendingIdentity = '';

        effect(() => {
            const id = this.identity();
            const p = this.currentPair();
            if (!id || id === lastIdentity || !p) return;

            if (!lastIdentity) {
                lastIdentity = id;
                void this.state.hydrate(this.datasetName(), p.media_file);
                return;
            }

            if (this.state.dirty()) {
                if (!confirm('Discard unsaved adjustments?')) {
                    pendingIdentity = lastIdentity;
                    queueMicrotask(() => {
                        const prev = pendingIdentity.split('/').pop();
                        if (prev) {
                            const list = this.pairs();
                            const idx = list.findIndex(x => x?.media_file === prev);
                            if (idx >= 0) this.overlay.setWorkspaceImage(idx);
                        }
                    });
                    return;
                }
            }

            lastIdentity = id;
            void this.state.hydrate(this.datasetName(), p.media_file);
        });
    }
}
