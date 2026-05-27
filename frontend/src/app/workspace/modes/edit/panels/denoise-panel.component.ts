import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { PipelineEditorState } from '../pipeline-editor.state';
import { ModelRestorePanelComponent } from '../components/model-restore-panel.component';
import { RestoreParams } from '../operation-defs';

@Component({
    selector: 'app-denoise-panel',
    standalone: true,
    imports: [ModelRestorePanelComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <app-model-restore-panel
            kind="denoise"
            [enabled]="op().enabled"
            [folder]="op().params.folder"
            [model]="op().params.model"
            [strength]="op().params.strength"
            [tileSize]="op().params.tile_size"
            (enableChanged)="set('enabled', $event)"
            (folderChanged)="set('folder', $event)"
            (modelChanged)="set('model', $event)"
            (strengthChanged)="set('strength', $event)"
            (tileSizeChanged)="set('tile_size', $event)"/>
    `,
})
export class DenoisePanelComponent {
    private state = inject(PipelineEditorState);
    protected op = computed(() => this.state.denoise());
    set(field: string, value: any): void {
        if (field === 'enabled') {
            this.state.denoise.update(o => ({ ...o, enabled: value }));
        } else {
            this.state.denoise.update(o => ({ ...o, enabled: true, params: { ...o.params, [field]: value } as RestoreParams }));
        }
    }
}
