import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { IcoComponent } from '../../../../icons/ico.component';
import { PipelineEditorState } from '../pipeline-editor.state';
import { ModelRestorePanelComponent } from '../components/model-restore-panel.component';
import { RestoreParams } from '../operation-defs';

@Component({
    selector: 'app-face-restore-panel',
    standalone: true,
    imports: [IcoComponent, ModelRestorePanelComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="ai-hint">
            <app-ico name="Sparkles" [size]="11"/>
            <span>Applies on Save (no live preview for AI ops).</span>
        </div>
        <app-model-restore-panel
            kind="face"
            [enabled]="op().enabled"
            [folder]="op().params.folder"
            [model]="op().params.model"
            [strength]="op().params.strength"
            [tileSize]="op().params.tile_size"
            [faceOnly]="op().params.face_only ?? true"
            (enableChanged)="set('enabled', $event)"
            (folderChanged)="set('folder', $event)"
            (modelChanged)="set('model', $event)"
            (strengthChanged)="set('strength', $event)"
            (tileSizeChanged)="set('tile_size', $event)"
            (faceOnlyChanged)="set('face_only', $event)"/>
    `,
    styles: [`
        :host { display: flex; flex-direction: column; gap: 10px; }
        .ai-hint {
            display: flex; align-items: center; gap: 6px;
            padding: 6px 10px;
            background: color-mix(in oklab, var(--color-violet) 12%, transparent);
            color: var(--color-violet);
            border: 1px solid color-mix(in oklab, var(--color-violet) 35%, transparent);
            border-radius: var(--radius-theme-sm);
            font-size: 11px;
        }
    `],
})
export class FaceRestorePanelComponent {
    private state = inject(PipelineEditorState);
    protected op = computed(() => this.state.faceRestore());
    set(field: string, value: any): void {
        if (field === 'enabled') {
            this.state.faceRestore.update(o => ({ ...o, enabled: value }));
        } else {
            this.state.faceRestore.update(o => ({ ...o, enabled: true, params: { ...o.params, [field]: value } as RestoreParams }));
        }
    }
}
