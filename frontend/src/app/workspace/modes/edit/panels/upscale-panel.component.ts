import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { PipelineEditorState } from '../pipeline-editor.state';
import { ModelRestorePanelComponent } from '../components/model-restore-panel.component';
import { UpscaleParams } from '../operation-defs';

@Component({
    selector: 'app-upscale-panel',
    standalone: true,
    imports: [ModelRestorePanelComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <app-model-restore-panel
            kind="upscale"
            [enabled]="op().enabled"
            [folder]="op().params.folder"
            [model]="op().params.model"
            [tileSize]="op().params.tile_size"
            [targetScale]="op().params.target_scale"
            [resizeMethod]="op().params.resize_method"
            (enableChanged)="set('enabled', $event)"
            (folderChanged)="set('folder', $event)"
            (modelChanged)="set('model', $event)"
            (tileSizeChanged)="set('tile_size', $event)"
            (targetScaleChanged)="set('target_scale', $event)"
            (resizeMethodChanged)="set('resize_method', $event)"/>
    `,
})
export class UpscalePanelComponent {
    private state = inject(PipelineEditorState);
    protected op = computed(() => this.state.upscale());
    set(field: string, value: any): void {
        if (field === 'enabled') {
            this.state.upscale.update(o => ({ ...o, enabled: value }));
        } else {
            this.state.upscale.update(o => ({ ...o, enabled: true, params: { ...o.params, [field]: value } as UpscaleParams }));
        }
    }
}
