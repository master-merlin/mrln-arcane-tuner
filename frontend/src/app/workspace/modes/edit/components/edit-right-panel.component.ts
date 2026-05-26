import { ChangeDetectionStrategy, Component, inject, input } from '@angular/core';
import { IcoComponent } from '../../../../icons/ico.component';
import { PipelineEditorState } from '../pipeline-editor.state';

@Component({
    selector: 'app-edit-right-panel',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <section class="section histogram">
            <div class="section-head">
                <app-ico name="TrendingUp" [size]="11"/>
                <span class="title">HISTOGRAM</span>
                <span class="mono mute">RGB</span>
            </div>
            <div class="section-body histogram-body">
                <!-- HistogramPanel mounts here in Task 13. -->
                <div class="placeholder">Histogram — Task 13</div>
            </div>
        </section>

        <div class="divider"></div>

        <section class="section pipeline">
            <div class="section-head">
                <app-ico name="Layers" [size]="11"/>
                <span class="title">PIPELINE ORDER</span>
            </div>
            <div class="section-body">
                <!-- PipelineOrderList mounts here in Task 14. -->
                <div class="placeholder">Pipeline list — Task 14</div>
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
            <button type="button" class="btn primary" (click)="onApply()" [disabled]="!state.dirty()">
                <app-ico name="Check" [size]="13"/> Apply &amp; save
            </button>
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
        .placeholder {
            padding: 14px; border: 1px dashed var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
            text-align: center; color: var(--color-text-muted); font-size: 12px;
        }
        .actions {
            margin-top: auto; padding: 14px 16px;
            border-top: 1px solid var(--color-border-subtle);
            background: var(--color-surface-low);
            display: flex; flex-direction: column; gap: 8px;
            flex-shrink: 0;
        }
        .row { display: flex; gap: 6px; }
        .row .btn { flex: 1; justify-content: center; }
        .actions > .btn.primary { width: 100%; justify-content: center; }
    `],
})
export class EditRightPanelComponent {
    datasetName = input.required<string>();
    mediaFile = input.required<string>();

    protected state = inject(PipelineEditorState);

    protected onRevert(): void {
        if (!confirm('Revert all edits and delete the saved overlay?')) return;
        void this.state.revert();
    }

    protected onCopy(): void {
        const recipe = { operations: this.state.blocks() };
        void navigator.clipboard.writeText(JSON.stringify(recipe, null, 2));
    }

    protected onApply(): void {
        void this.state.applyAndSave();
    }
}
