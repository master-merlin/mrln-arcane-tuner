import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { IcoComponent } from '../../../../icons/ico.component';
import { PipelineEditorState } from '../pipeline-editor.state';
import { ColorMatchParams } from '../operation-defs';

@Component({
    selector: 'app-color-match-panel',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="info">
            <app-ico name="Info" [size]="13"/>
            <p>Applies on Save (preview shows the rest of the pipeline only; this op runs server-side first).</p>
        </div>

        <div class="enable-row">
            <label><input type="checkbox" [checked]="op().enabled" (change)="setEnabled($any($event.target).checked)"/> Enable</label>
        </div>

        <div class="field">
            <label>Reference image</label>
            <div class="ref-row">
                <span class="mono ref" [title]="op().params.reference_path || ''">
                    {{ op().params.reference_path || '— none selected —' }}
                </span>
                <button type="button" class="btn sm" (click)="pickReference()">Pick…</button>
                @if (op().params.reference_path) {
                    <button type="button" class="btn sm ghost" (click)="clearReference()" title="Clear">×</button>
                }
            </div>
        </div>

        <div class="field">
            <label>Method</label>
            <div class="seg">
                <button type="button" [class.active]="op().params.method === 'cdf'" (click)="setMethod('cdf')">CDF (histogram)</button>
                <button type="button" [class.active]="op().params.method === 'wavelet'" (click)="setMethod('wavelet')">Wavelet</button>
            </div>
        </div>

        <div class="field">
            <div class="row"><label>Strength</label><span class="mono">{{ (op().params.strength * 100).toFixed(0) }}%</span></div>
            <input type="range" min="0" max="1" step="0.01" [value]="op().params.strength" (input)="setStrength(+$any($event.target).value)"/>
        </div>
    `,
    styles: [`
        :host { display: flex; flex-direction: column; gap: 10px; }
        .info {
            display: flex; gap: 8px;
            padding: 8px 10px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
            color: var(--color-text-muted);
            font-size: 11px; line-height: 1.4;
        }
        .info p { margin: 0; }
        .enable-row label { display: flex; gap: 6px; font-size: 12px; cursor: pointer; }
        .field { display: flex; flex-direction: column; gap: 4px; }
        .field .row { display: flex; justify-content: space-between; align-items: center; font-size: 11.5px; }
        .field .row .mono { color: var(--color-text-muted); font-size: 11px; }
        .field input[type=range] { width: 100%; accent-color: var(--color-brand); }
        .ref-row { display: flex; align-items: center; gap: 6px; }
        .ref-row .ref {
            flex: 1; padding: 5px 8px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-sm);
            font-size: 11px;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
            color: var(--color-text-muted);
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
export class ColorMatchPanelComponent {
    private state = inject(PipelineEditorState);
    protected op = computed(() => this.state.colorMatch());

    setEnabled(enabled: boolean): void { this.state.colorMatch.update(o => ({ ...o, enabled })); }
    setMethod(method: ColorMatchParams['method']): void {
        this.state.colorMatch.update(o => ({ ...o, enabled: true, params: { ...o.params, method } }));
    }
    setStrength(strength: number): void {
        this.state.colorMatch.update(o => ({ ...o, enabled: true, params: { ...o.params, strength } }));
    }
    pickReference(): void {
        // v1: simple prompt() — a dedicated picker modal lands in a follow-up.
        // The prompt accepts a dataset-relative image path.
        const chosen = prompt('Reference image path (dataset-relative):', this.op().params.reference_path ?? '');
        if (chosen != null && chosen.trim() !== '') {
            this.state.colorMatch.update(o => ({ ...o, enabled: true, params: { ...o.params, reference_path: chosen.trim() } }));
        }
    }
    clearReference(): void {
        this.state.colorMatch.update(o => ({ ...o, params: { ...o.params, reference_path: null } }));
    }
}
