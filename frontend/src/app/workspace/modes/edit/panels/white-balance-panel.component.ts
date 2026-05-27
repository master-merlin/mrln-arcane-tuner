import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { IcoComponent } from '../../../../icons/ico.component';
import { PipelineEditorState } from '../pipeline-editor.state';
import { WBParams } from '../operation-defs';

const PRESETS: ReadonlyArray<{ name: string; temp: number; tint: number }> = [
    { name: 'As Shot',     temp: 5500, tint: 0 },
    { name: 'Daylight',    temp: 5500, tint: 0 },
    { name: 'Cloudy',      temp: 6500, tint: 0 },
    { name: 'Shade',       temp: 7500, tint: 0 },
    { name: 'Tungsten',    temp: 3200, tint: 0 },
    { name: 'Fluorescent', temp: 4000, tint: 0 },
];

@Component({
    selector: 'app-wb-panel',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="enable-row">
            <label><input type="checkbox" [checked]="op().enabled" (change)="setEnabled($any($event.target).checked)"/> Enable</label>
        </div>

        <div class="field">
            <div class="row">
                <label>Temperature</label>
                <span class="mono">{{ op().params.temperature }} K</span>
            </div>
            <input type="range" min="2000" max="12000" step="100"
                   [value]="op().params.temperature"
                   (input)="setTemp(+$any($event.target).value)"/>
        </div>

        <div class="field">
            <div class="row">
                <label>Tint</label>
                <span class="mono">{{ formatTint(op().params.tint) }}</span>
            </div>
            <input type="range" min="-100" max="100" step="1"
                   [value]="op().params.tint"
                   (input)="setTint(+$any($event.target).value)"/>
        </div>

        <button type="button" class="btn sm auto" (click)="autoWB()" title="Estimate from image">
            <app-ico name="Wand2" [size]="11"/> Auto white balance
        </button>

        <div class="divider"></div>

        <div class="section-title">PRESETS</div>
        <div class="preset-grid">
            @for (p of presets; track p.name) {
                <button type="button" class="preset"
                        [class.active]="op().params.temperature === p.temp && op().params.tint === p.tint"
                        (click)="applyPreset(p.temp, p.tint)">
                    <span>{{ p.name }}</span>
                    <span class="mono">{{ p.temp }} K</span>
                </button>
            }
        </div>
    `,
    styles: [`
        :host { display: flex; flex-direction: column; gap: 10px; }
        .enable-row label { display: flex; gap: 6px; font-size: 12px; cursor: pointer; }
        .field { display: flex; flex-direction: column; gap: 4px; }
        .field .row { display: flex; justify-content: space-between; align-items: center; font-size: 11.5px; }
        .field .row .mono { color: var(--color-text-muted); font-size: 11px; }
        .field input[type=range] { width: 100%; accent-color: var(--color-brand); }
        .auto { justify-content: center; }
        .divider { height: 1px; background: var(--color-border-subtle); margin: 4px 0; }
        .section-title {
            font-size: 10px; font-weight: 700;
            letter-spacing: 0.10em; text-transform: uppercase;
            color: var(--color-text-subtle);
        }
        .preset-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
        .preset {
            display: flex; align-items: center; justify-content: space-between;
            padding: 6px 8px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-sm);
            font-size: 11px;
            color: var(--color-text-primary);
            cursor: pointer;
        }
        .preset .mono { font-size: 9.5px; color: var(--color-text-muted); }
        .preset.active {
            background: color-mix(in oklab, var(--color-brand) 14%, transparent);
            border-color: color-mix(in oklab, var(--color-brand) 35%, transparent);
            color: var(--color-brand);
        }
    `],
})
export class WhiteBalancePanelComponent {
    private state = inject(PipelineEditorState);
    protected op = computed(() => this.state.whiteBalance());
    protected presets = PRESETS;

    setEnabled(enabled: boolean): void {
        this.state.whiteBalance.update(o => ({ ...o, enabled }));
    }
    setTemp(t: number): void {
        this.state.whiteBalance.update(o => ({ ...o, enabled: true, params: { ...o.params, temperature: t } }));
    }
    setTint(t: number): void {
        this.state.whiteBalance.update(o => ({ ...o, enabled: true, params: { ...o.params, tint: t } }));
    }
    applyPreset(temp: number, tint: number): void {
        this.state.whiteBalance.update(o => ({ ...o, enabled: true, params: { ...o.params, temperature: temp, tint } }));
    }
    autoWB(): void {
        // Backend may expose an auto-WB endpoint; if not yet wired, nudge to 6500/0.
        this.applyPreset(6500, 0);
    }
    formatTint(t: number): string {
        return (t > 0 ? '+' : '') + t.toString();
    }
}
