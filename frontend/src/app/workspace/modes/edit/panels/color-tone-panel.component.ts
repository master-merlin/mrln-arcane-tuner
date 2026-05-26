import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { IcoComponent } from '../../../../icons/ico.component';
import { PipelineEditorState } from '../pipeline-editor.state';

@Component({
    selector: 'app-color-tone-panel',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="enable-row">
            <label><input type="checkbox" [checked]="op().enabled" (change)="setEnabled($any($event.target).checked)"/> Enable</label>
        </div>

        <div class="field">
            <div class="row"><label>Hue shift</label><span class="mono">{{ op().params.hue_shift > 0 ? '+' : '' }}{{ op().params.hue_shift }}°</span></div>
            <input type="range" min="-180" max="180" step="1"
                   [value]="op().params.hue_shift" (input)="set('hue_shift', +$any($event.target).value)"/>
        </div>
        <div class="field">
            <div class="row"><label>Saturation</label><span class="mono">{{ op().params.saturation.toFixed(2) }}</span></div>
            <input type="range" min="0" max="3" step="0.01"
                   [value]="op().params.saturation" (input)="set('saturation', +$any($event.target).value)"/>
        </div>
        <div class="field">
            <div class="row"><label>Contrast</label><span class="mono">{{ op().params.contrast.toFixed(2) }}</span></div>
            <input type="range" min="0" max="3" step="0.01"
                   [value]="op().params.contrast" (input)="set('contrast', +$any($event.target).value)"/>
        </div>

        <div class="info">
            <app-ico name="Info" [size]="13"/>
            <p>For per-band hue/saturation/luminance use the <b>HSL</b> tab.</p>
        </div>
    `,
    styles: [`
        :host { display: flex; flex-direction: column; gap: 10px; }
        .enable-row label { display: flex; gap: 6px; font-size: 12px; cursor: pointer; }
        .field { display: flex; flex-direction: column; gap: 4px; }
        .field .row { display: flex; justify-content: space-between; align-items: center; font-size: 11.5px; }
        .field .row .mono { color: var(--color-text-muted); font-size: 11px; }
        .field input[type=range] { width: 100%; accent-color: var(--color-brand); }
        .info {
            display: flex; gap: 8px;
            padding: 10px 12px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
            color: var(--color-text-muted);
            font-size: 11px; line-height: 1.4;
        }
        .info p { margin: 0; }
    `],
})
export class ColorTonePanelComponent {
    private state = inject(PipelineEditorState);
    protected op = computed(() => this.state.colorTone());

    setEnabled(enabled: boolean): void {
        this.state.colorTone.update(o => ({ ...o, enabled }));
    }

    set(field: 'hue_shift' | 'saturation' | 'contrast', value: number): void {
        this.state.colorTone.update(o => ({ ...o, enabled: true, params: { ...o.params, [field]: value } }));
    }
}
