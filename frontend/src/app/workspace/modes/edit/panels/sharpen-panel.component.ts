import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { PipelineEditorState } from '../pipeline-editor.state';
import { SharpenParams } from '../operation-defs';

@Component({
    selector: 'app-sharpen-panel',
    standalone: true,
    imports: [],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="enable-row">
            <label><input type="checkbox" [checked]="op().enabled" (change)="setEnabled($any($event.target).checked)"/> Enable</label>
        </div>

        <div class="field">
            <label>Method</label>
            <div class="seg">
                @for (m of methods; track m.key) {
                    <button type="button"
                            [class.active]="op().params.method === m.key"
                            (click)="setMethod(m.key)">{{ m.label }}</button>
                }
            </div>
        </div>

        @switch (op().params.method) {
            @case ('unsharp') {
                <div class="field">
                    <div class="row"><label>Radius</label><span class="mono">{{ op().params.radius.toFixed(1) }} px</span></div>
                    <input type="range" min="0.1" max="10" step="0.1" [value]="op().params.radius" (input)="set('radius', +$any($event.target).value)"/>
                </div>
                <div class="field">
                    <div class="row"><label>Amount</label><span class="mono">{{ op().params.amount }}%</span></div>
                    <input type="range" min="0" max="500" step="1" [value]="op().params.amount" (input)="set('amount', +$any($event.target).value)"/>
                </div>
                <div class="field">
                    <div class="row"><label>Threshold</label><span class="mono">{{ op().params.threshold }}</span></div>
                    <input type="range" min="0" max="20" step="1" [value]="op().params.threshold" (input)="set('threshold', +$any($event.target).value)"/>
                </div>
            }
            @case ('kernel') {
                <div class="field">
                    <div class="row"><label>Strength</label><span class="mono">{{ op().params.strength.toFixed(2) }}</span></div>
                    <input type="range" min="0" max="2" step="0.01" [value]="op().params.strength" (input)="set('strength', +$any($event.target).value)"/>
                </div>
            }
            @case ('high_pass') {
                <div class="field">
                    <div class="row"><label>Radius</label><span class="mono">{{ op().params.radius.toFixed(1) }} px</span></div>
                    <input type="range" min="0.5" max="20" step="0.5" [value]="op().params.radius" (input)="set('radius', +$any($event.target).value)"/>
                </div>
                <div class="field">
                    <div class="row"><label>Strength</label><span class="mono">{{ op().params.strength.toFixed(2) }}</span></div>
                    <input type="range" min="0" max="2" step="0.01" [value]="op().params.strength" (input)="set('strength', +$any($event.target).value)"/>
                </div>
            }
        }
    `,
    styles: [`
        :host { display: flex; flex-direction: column; gap: 10px; }
        .enable-row label { display: flex; gap: 6px; font-size: 12px; cursor: pointer; }
        .field { display: flex; flex-direction: column; gap: 4px; }
        .field .row { display: flex; justify-content: space-between; align-items: center; font-size: 11.5px; }
        .field .row .mono { color: var(--color-text-muted); font-size: 11px; }
        .field input[type=range] { width: 100%; accent-color: var(--color-brand); }
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
export class SharpenPanelComponent {
    private state = inject(PipelineEditorState);
    protected op = computed(() => this.state.sharpen());

    protected methods = [
        { key: 'unsharp' as const,   label: 'Unsharp' },
        { key: 'kernel' as const,    label: 'Kernel' },
        { key: 'high_pass' as const, label: 'High-Pass' },
    ];

    setEnabled(enabled: boolean): void {
        this.state.sharpen.update(o => ({ ...o, enabled }));
    }
    setMethod(method: SharpenParams['method']): void {
        this.state.sharpen.update(o => ({
            ...o,
            enabled: true,
            params: { ...o.params, method } as SharpenParams
        }));
    }
    set(field: keyof SharpenParams, value: number): void {
        this.state.sharpen.update(o => ({
            ...o,
            enabled: true,
            params: { ...o.params, [field]: value } as SharpenParams
        }));
    }
}
