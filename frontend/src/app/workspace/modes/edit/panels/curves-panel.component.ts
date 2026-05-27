import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { PipelineEditorState } from '../pipeline-editor.state';
import { CurvePoint } from '../operation-defs';
import { CurvesEditorComponent } from '../../../../components/dataset/dataset-viewer/components/curves-editor';

type ChannelKey = 'master' | 'r' | 'g' | 'b';

@Component({
    selector: 'app-curves-panel',
    standalone: true,
    imports: [CurvesEditorComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="enable-row">
            <label><input type="checkbox" [checked]="op().enabled" (change)="setEnabled($any($event.target).checked)"/> Enable</label>
        </div>
        <app-curves-editor
            [masterCurve]="op().params.master"
            [rCurve]="op().params.r"
            [gCurve]="op().params.g"
            [bCurve]="op().params.b"
            (curveChanged)="onChange($event)"/>
    `,
    styles: [`
        :host { display: flex; flex-direction: column; gap: 10px; }
        .enable-row label { display: flex; gap: 6px; font-size: 12px; cursor: pointer; }
    `],
})
export class CurvesPanelComponent {
    private state = inject(PipelineEditorState);
    protected op = computed(() => this.state.curves());

    setEnabled(enabled: boolean): void {
        this.state.curves.update(o => ({ ...o, enabled }));
    }

    onChange(ev: { channel: ChannelKey; points: CurvePoint[] }): void {
        this.state.curves.update(o => ({
            ...o,
            enabled: true,
            params: { ...o.params, [ev.channel]: ev.points },
        }));
    }
}
