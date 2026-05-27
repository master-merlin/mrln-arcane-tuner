import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { PipelineEditorState } from '../pipeline-editor.state';

@Component({
    selector: 'app-lens-panel',
    standalone: true,
    imports: [],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="enable-row">
            <label><input type="checkbox" [checked]="op().enabled" (change)="setEnabled($any($event.target).checked)"/> Enable</label>
        </div>

        <div class="preview" aria-hidden="true">
            <svg viewBox="0 0 178 100" preserveAspectRatio="none">
                @for (x of [20, 40, 60, 80]; track x) {
                    <path [attr.d]="'M ' + (x * 1.78) + ' 4 Q ' + (x * 1.78 + (x - 50) * (op().params.barrel * 4)) + ' 50 ' + (x * 1.78) + ' 96'"
                          fill="none" stroke="oklch(0.50 0.04 265)" stroke-width="0.4" stroke-opacity="0.6"/>
                }
                @for (y of [20, 40, 60, 80]; track y) {
                    <path [attr.d]="'M 4 ' + y + ' Q 89 ' + (y + (y - 50) * (op().params.barrel * 4)) + ' 174 ' + y"
                          fill="none" stroke="oklch(0.50 0.04 265)" stroke-width="0.4" stroke-opacity="0.6"/>
                }
                <rect x="4" y="4" width="170" height="92" fill="none" stroke="var(--color-brand)" stroke-width="0.8" stroke-opacity="0.7" stroke-dasharray="2 2"/>
            </svg>
        </div>

        <div class="field">
            <div class="row"><label>Barrel / Pincushion</label><span class="mono">{{ op().params.barrel > 0 ? '+' : '' }}{{ op().params.barrel.toFixed(2) }}</span></div>
            <input type="range" min="-1" max="1" step="0.01" [value]="op().params.barrel" (input)="set('barrel', +$any($event.target).value)"/>
        </div>
        <div class="field">
            <div class="row"><label>Vertical Keystone</label><span class="mono">{{ op().params.v_keystone.toFixed(1) }}°</span></div>
            <input type="range" min="-45" max="45" step="0.5" [value]="op().params.v_keystone" (input)="set('v_keystone', +$any($event.target).value)"/>
        </div>
        <div class="field">
            <div class="row"><label>Horizontal Keystone</label><span class="mono">{{ op().params.h_keystone.toFixed(1) }}°</span></div>
            <input type="range" min="-45" max="45" step="0.5" [value]="op().params.h_keystone" (input)="set('h_keystone', +$any($event.target).value)"/>
        </div>

        <label class="toggle-row">
            <input type="checkbox" [checked]="op().params.auto_crop" (change)="setAutoCrop($any($event.target).checked)"/>
            Auto-crop after correction
        </label>
    `,
    styles: [`
        :host { display: flex; flex-direction: column; gap: 10px; }
        .enable-row label, .toggle-row { display: flex; gap: 6px; font-size: 12px; cursor: pointer; }
        .field { display: flex; flex-direction: column; gap: 4px; }
        .field .row { display: flex; justify-content: space-between; align-items: center; font-size: 11.5px; }
        .field .row .mono { color: var(--color-text-muted); font-size: 11px; }
        .field input[type=range] { width: 100%; accent-color: var(--color-brand); }
        .preview {
            aspect-ratio: 1.78;
            background: var(--color-base);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
            padding: 0;
        }
        .preview svg { width: 100%; height: 100%; display: block; }
    `],
})
export class LensPanelComponent {
    private state = inject(PipelineEditorState);
    protected op = computed(() => this.state.lens());

    setEnabled(enabled: boolean): void {
        this.state.lens.update(o => ({ ...o, enabled }));
    }
    set(field: 'barrel' | 'v_keystone' | 'h_keystone', value: number): void {
        this.state.lens.update(o => ({ ...o, enabled: true, params: { ...o.params, [field]: value } }));
    }
    setAutoCrop(auto_crop: boolean): void {
        this.state.lens.update(o => ({ ...o, enabled: true, params: { ...o.params, auto_crop } }));
    }
}
