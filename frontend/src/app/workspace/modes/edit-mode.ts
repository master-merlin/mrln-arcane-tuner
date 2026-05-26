import { ChangeDetectionStrategy, Component, computed, DestroyRef, effect, inject, input } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService } from '../../services/dataset';
import { PipelineEditorState } from './edit/pipeline-editor.state';
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
    providers: [PipelineEditorState],
    template: `
        @if (currentPair(); as pair) {
            <div class="edit-grid">
                <aside class="pane left">
                    <app-edit-left-panel/>
                </aside>
                <main class="pane center">
                    <app-edit-canvas
                        [datasetName]="datasetName()"
                        [mediaFile]="pair.media_file"
                        [hasOverlay]="!!pair?.metadata?.has_overlay"/>
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
    private destroyRef = inject(DestroyRef);

    protected currentPair = computed(() => {
        const list = this.pairs();
        const idx = this.imageIndex();
        return idx >= 0 && idx < list.length ? list[idx] : null;
    });

    private renderTimer: ReturnType<typeof setTimeout> | null = null;

    private identity = computed<string | null>(() => {
        const p = this.currentPair();
        return p?.media_file ? `${this.datasetName()}/${p.media_file}` : null;
    });

    constructor() {
        // 1. Identity change → hydrate.
        let lastIdentity = '';
        effect(() => {
            const id = this.identity();
            const p = this.currentPair();
            if (!id || id === lastIdentity || !p) return;
            lastIdentity = id;
            if (this.renderTimer) { clearTimeout(this.renderTimer); this.renderTimer = null; }
            void this.state.hydrate(this.datasetName(), p.media_file);
        });

        // 2. Blocks change with stable identity → debounced render.
        let lastBlocksJson = '';
        effect(() => {
            const id = this.identity();
            const blocksJson = JSON.stringify(this.state.blocks());
            if (!id || blocksJson === lastBlocksJson) return;
            lastBlocksJson = blocksJson;
            if (this.renderTimer) clearTimeout(this.renderTimer);
            this.renderTimer = setTimeout(() => {
                this.renderTimer = null;
                void this.state.renderNow(false);
            }, 250);
        });

        this.destroyRef.onDestroy(() => {
            if (this.renderTimer) {
                clearTimeout(this.renderTimer);
                this.renderTimer = null;
            }
        });
    }
}
