import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { PipelineEditorState } from '../pipeline-editor.state';
import { VignetteParams } from '../operation-defs';

@Component({
    selector: 'app-vignette-panel',
    standalone: true,
    imports: [],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="enable-row">
            <label><input type="checkbox" [checked]="op().enabled" (change)="setEnabled($any($event.target).checked)"/> Enable</label>
        </div>

        <div class="preview" aria-hidden="true"></div>

        <div class="field">
            <div class="row"><label>Amount</label><span class="mono">{{ op().params.amount.toFixed(2) }}</span></div>
            <input type="range" min="-1" max="1" step="0.01" [value]="op().params.amount" (input)="set('amount', +$any($event.target).value)"/>
        </div>
        <div class="field">
            <div class="row"><label>Midpoint</label><span class="mono">{{ (op().params.midpoint * 100).toFixed(0) }}%</span></div>
            <input type="range" min="0" max="1" step="0.01" [value]="op().params.midpoint" (input)="set('midpoint', +$any($event.target).value)"/>
        </div>
        <div class="field">
            <div class="row"><label>Feather</label><span class="mono">{{ (op().params.feather * 100).toFixed(0) }}%</span></div>
            <input type="range" min="0.01" max="1" step="0.01" [value]="op().params.feather" (input)="set('feather', +$any($event.target).value)"/>
        </div>

        <div class="field">
            <label>Shape</label>
            <div class="seg">
                <button type="button" [class.active]="op().params.shape === 'circular'" (click)="setShape('circular')">Circular</button>
                <button type="button" [class.active]="op().params.shape === 'rectangular'" (click)="setShape('rectangular')">Rectangular</button>
            </div>
        </div>

        <label class="toggle-row">
            <input type="checkbox" [checked]="op().params.apply_before_lut" (change)="setBeforeLut($any($event.target).checked)"/>
            Apply before LUT
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
            background: radial-gradient(ellipse 70% 70% at center, oklch(0.60 0.04 265 / 0.9), oklch(0.10 0.01 265) 90%);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
        }
        .seg { display: flex; gap: 2px; border: 1px solid var(--color-border-subtle); border-radius: var(--radius-theme-sm); overflow: hidden; }
        .seg button {
            flex: 1; padding: 5px 8px;
            background: var(--color-surface-mid);
            color: var(--color-text-muted);
            border: none; cursor: pointer; font-size: 11px;
        }
        .seg button.active { background: color-mix(in oklab, var(--color-brand) 18%, transparent); color: var(--color-brand); }
    `],
})
export class VignettePanelComponent {
    private state = inject(PipelineEditorState);
    protected op = computed(() => this.state.vignette());

    setEnabled(enabled: boolean): void {
        this.state.vignette.update(o => ({ ...o, enabled }));
    }
    set(field: 'amount' | 'midpoint' | 'feather', value: number): void {
        this.state.vignette.update(o => ({ ...o, enabled: true, params: { ...o.params, [field]: value } }));
    }
    setShape(shape: VignetteParams['shape']): void {
        this.state.vignette.update(o => ({ ...o, enabled: true, params: { ...o.params, shape } }));
    }
    setBeforeLut(apply_before_lut: boolean): void {
        this.state.vignette.update(o => ({ ...o, enabled: true, params: { ...o.params, apply_before_lut } }));
    }
}
