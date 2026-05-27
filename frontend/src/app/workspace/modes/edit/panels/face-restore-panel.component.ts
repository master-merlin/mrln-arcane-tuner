import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { PipelineEditorState } from '../pipeline-editor.state';
import { ModelRestorePanelComponent } from '../components/model-restore-panel.component';
import { RestoreParams } from '../operation-defs';

@Component({
    selector: 'app-face-restore-panel',
    standalone: true,
    imports: [ModelRestorePanelComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
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
