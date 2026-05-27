import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { PipelineEditorState } from '../pipeline-editor.state';
import { HslParams } from '../operation-defs';
import { HSLPanelComponent as LegacyHsl, HSLConfig } from '../../../../components/dataset/dataset-viewer/components/hsl-panel';

@Component({
    selector: 'app-hsl-panel-wrap',
    standalone: true,
    imports: [LegacyHsl],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="enable-row">
            <label><input type="checkbox" [checked]="op().enabled" (change)="setEnabled($any($event.target).checked)"/> Enable</label>
        </div>
        <app-hsl-panel
            [hslConfig]="op().params"
            (hslChanged)="onChange($event)"/>
    `,
    styles: [`
        :host { display: flex; flex-direction: column; gap: 10px; }
        .enable-row label { display: flex; gap: 6px; font-size: 12px; cursor: pointer; }
    `],
})
export class HslPanelWrapperComponent {
    private state = inject(PipelineEditorState);
    protected op = computed(() => this.state.hslSelective());

    setEnabled(enabled: boolean): void {
        this.state.hslSelective.update(o => ({ ...o, enabled }));
    }

    onChange(config: HSLConfig): void {
        this.state.hslSelective.update(o => ({ ...o, enabled: true, params: config as HslParams }));
    }
}
