import { ChangeDetectionStrategy, Component, computed, inject, input } from '@angular/core';
import { IcoComponent } from '../../../../icons/ico.component';
import { OverlayStore, Overlay } from '../../../../state/overlay.store';
import { PipelineEditorState } from '../pipeline-editor.state';
import { HistogramPanelComponent } from './histogram-panel.component';
import { PipelineOrderListComponent } from './pipeline-order-list.component';

@Component({
    selector: 'app-edit-right-panel',
    standalone: true,
    imports: [IcoComponent, HistogramPanelComponent, PipelineOrderListComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <section class="section histogram">
            <div class="section-head">
                <app-ico name="TrendingUp" [size]="11"/>
                <span class="title">HISTOGRAM</span>
                <span class="mono mute">RGB</span>
            </div>
            <div class="section-body histogram-body">
                <app-histogram-panel
                    [datasetName]="datasetName()"
                    [mediaFile]="mediaFile()"/>
            </div>
        </section>

        <div class="divider"></div>

        <section class="section pipeline">
            <div class="section-head">
                <app-ico name="Layers" [size]="11"/>
                <span class="title">PIPELINE ORDER</span>
            </div>
            <div class="section-body">
                <app-pipeline-order-list/>
            </div>
        </section>

        <footer class="actions">
            <div class="row">
                <button type="button" class="btn sm" (click)="onRevert()">
                    <app-ico name="History" [size]="12"/> Revert
                </button>
                <button type="button" class="btn sm" (click)="onCopy()">
                    <app-ico name="Copy" [size]="12"/> Copy
                </button>
            </div>
            <div class="row">
                <button type="button" class="btn primary save" (click)="onSave()" [disabled]="!state.dirty()">
                    <app-ico name="Check" [size]="13"/> Save
                </button>
                <button type="button" class="btn warn bake" (click)="onBake()" [disabled]="!canBake()">
                    <app-ico name="Flame" [size]="13"/> Bake in
                </button>
            </div>
        </footer>
    `,
    styles: [`
        :host { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
        .section { padding: 12px 16px 0; flex-shrink: 0; }
        .section.pipeline { flex: 1; padding-bottom: 12px; display: flex; flex-direction: column; overflow: hidden; min-height: 0; }
        .section.pipeline .section-body { flex: 1; overflow-y: auto; min-height: 0; }
        .section-head {
            display: flex; align-items: center; gap: 6px;
            font-size: 11px; font-weight: 700;
            letter-spacing: 0.10em; text-transform: uppercase;
            color: var(--color-text-primary);
        }
        .section-head .mute { margin-left: auto; color: var(--color-text-muted); font-size: 10.5px; }
        .section-body { padding-top: 8px; }
        .divider { height: 1px; background: var(--color-border-subtle); margin: 8px 16px; }
        .actions {
            margin-top: auto; padding: 14px 16px;
            border-top: 1px solid var(--color-border-subtle);
            background: var(--color-surface-low);
            display: flex; flex-direction: column; gap: 8px;
            flex-shrink: 0;
        }
        .row { display: flex; gap: 6px; }
        .row .btn { flex: 1; justify-content: center; }
        .btn.warn {
            background: color-mix(in oklab, var(--color-danger, oklch(0.6 0.18 30)) 16%, transparent);
            color: var(--color-danger, oklch(0.65 0.18 30));
            border-color: color-mix(in oklab, var(--color-danger, oklch(0.6 0.18 30)) 40%, transparent);
        }
        .btn.warn:disabled { opacity: 0.45; cursor: not-allowed; }
    `],
})
export class EditRightPanelComponent {
    datasetName = input.required<string>();
    mediaFile = input.required<string>();

    protected state = inject(PipelineEditorState);
    private overlayStore = inject(OverlayStore);

    /** Bake requires a saved overlay (server-side) AND no in-flight edits. */
    protected canBake = computed<boolean>(() => {
        if (this.state.dirty()) return false;
        const id = `${this.datasetName()}/${this.mediaFile()}`;
        const ov = (this.overlayStore.entities() ?? []).find((o: Overlay) => o.id === id);
        return !!ov?.overlay_file;
    });

    protected onRevert(): void {
        if (!confirm('Revert all edits and delete the saved overlay?')) return;
        void this.state.revert();
    }

    protected onCopy(): void {
        const recipe = { operations: this.state.blocks() };
        void navigator.clipboard.writeText(JSON.stringify(recipe, null, 2));
    }

    protected onSave(): void {
        void this.state.applyAndSave();
    }

    protected onBake(): void {
        if (!confirm('Bake overlay into original? This replaces the source file and clears the recipe.')) return;
        void this.state.bake();
    }
}
