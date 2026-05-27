import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { IcoComponent } from '../../../../icons/ico.component';
import { PipelineEditorState } from '../pipeline-editor.state';
import { LutEntry } from '../operation-defs';

@Component({
    selector: 'app-lut-panel',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="enable-row">
            <label><input type="checkbox" [checked]="op().enabled" (change)="setEnabled($any($event.target).checked)"/> Enable LUT pipeline</label>
        </div>

        <div class="actions">
            <input type="file" #fileInput hidden accept=".cube" (change)="onFileChosen($event)"/>
            <button type="button" class="btn sm" (click)="fileInput.click()">
                <app-ico name="Plus" [size]="11"/> Import .cube
            </button>
            <button type="button" class="btn sm" (click)="exportStack()" [disabled]="!op().params.luts.length">
                <app-ico name="Download" [size]="11"/> Export
            </button>
        </div>

        <div class="stack">
            @if (op().params.luts.length === 0) {
                <div class="empty">
                    <p>No LUTs in the stack. Import a <code>.cube</code> file above or pick a preset below.</p>
                </div>
            }
            @for (lut of op().params.luts; track $index; let i = $index) {
                <div class="lut-row" [class.disabled]="!lut.enabled">
                    <span class="grip" title="(drag-reorder coming later)">⋮</span>
                    <input type="checkbox" [checked]="lut.enabled" (change)="setLutEnabled(i, $any($event.target).checked)"/>
                    <span class="name mono" [title]="lut.file">{{ lutBasename(lut.file) }}</span>
                    <input class="strength" type="range" min="0" max="1" step="0.01"
                           [value]="lut.strength" (input)="setLutStrength(i, +$any($event.target).value)"/>
                    <span class="pct mono">{{ (lut.strength * 100).toFixed(0) }}%</span>
                    <button type="button" class="del" (click)="removeLut(i)" title="Remove">
                        <app-ico name="X" [size]="11"/>
                    </button>
                </div>
            }
        </div>

        <label class="toggle-row">
            <input type="checkbox" [checked]="op().params.tetrahedral" (change)="setTetrahedral($any($event.target).checked)"/>
            Tetrahedral interpolation (stack-wide)
        </label>

        <div class="divider"></div>

        <div class="section-title">PRESETS</div>
        <div class="presets">
            @for (p of presets; track p.file) {
                <button type="button" class="preset" (click)="addPreset(p.file, p.name)">
                    <span class="name">{{ p.name }}</span>
                    <span class="sub mono">{{ p.file }}</span>
                </button>
            }
        </div>
    `,
    styles: [`
        :host { display: flex; flex-direction: column; gap: 10px; }
        .enable-row label, .toggle-row { display: flex; gap: 6px; font-size: 12px; cursor: pointer; }
        .actions { display: flex; gap: 6px; }
        .actions .btn { flex: 1; justify-content: center; }
        .stack { display: flex; flex-direction: column; gap: 4px; }
        .lut-row {
            display: grid;
            grid-template-columns: 14px 18px 1fr 60px 36px 22px;
            align-items: center; gap: 6px;
            padding: 4px 6px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-sm);
            font-size: 11px;
        }
        .lut-row.disabled { opacity: 0.55; }
        .grip { color: var(--color-text-subtle); cursor: grab; user-select: none; text-align: center; }
        .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .strength { width: 100%; accent-color: var(--color-brand); }
        .pct { color: var(--color-text-muted); font-size: 10px; text-align: right; }
        .del { background: transparent; border: none; cursor: pointer; color: var(--color-text-muted); }
        .del:hover { color: var(--color-danger); }
        .empty {
            padding: 12px;
            border: 1px dashed var(--color-border-subtle);
            border-radius: var(--radius-theme-sm);
            color: var(--color-text-muted); font-size: 11px; text-align: center;
        }
        .empty p { margin: 0; }
        .divider { height: 1px; background: var(--color-border-subtle); margin: 4px 0; }
        .section-title { font-size: 10px; font-weight: 700; letter-spacing: 0.10em; text-transform: uppercase; color: var(--color-text-subtle); }
        .presets { display: flex; flex-direction: column; gap: 4px; }
        .preset {
            display: flex; flex-direction: column; align-items: flex-start;
            padding: 6px 8px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-sm);
            text-align: left; cursor: pointer;
        }
        .preset .name { font-size: 11.5px; font-weight: 500; }
        .preset .sub { font-size: 9.5px; color: var(--color-text-muted); }
    `],
})
export class LutPanelComponent {
    private state = inject(PipelineEditorState);
    protected op = computed(() => this.state.lut());

    protected presets = [
        { name: 'Neutral',         file: 'presets/neutral.cube' },
        { name: 'Cinematic Teal',  file: 'presets/cinematic-teal.cube' },
        { name: 'Warm Portrait',   file: 'presets/warm-portrait.cube' },
        { name: 'Cool Industrial', file: 'presets/cool-industrial.cube' },
    ];

    setEnabled(enabled: boolean): void { this.state.lut.update(o => ({ ...o, enabled })); }
    setTetrahedral(tetrahedral: boolean): void {
        this.state.lut.update(o => ({ ...o, enabled: true, params: { ...o.params, tetrahedral } }));
    }
    setLutEnabled(i: number, enabled: boolean): void { this.patchLut(i, l => ({ ...l, enabled })); }
    setLutStrength(i: number, strength: number): void { this.patchLut(i, l => ({ ...l, strength })); }
    removeLut(i: number): void {
        this.state.lut.update(o => {
            const next = o.params.luts.slice(); next.splice(i, 1);
            return { ...o, params: { ...o.params, luts: next } };
        });
    }
    addPreset(file: string, _name: string): void {
        this.state.lut.update(o => ({ ...o, enabled: true, params: { ...o.params, luts: [...o.params.luts, { file, strength: 1.0, enabled: true }] } }));
    }
    onFileChosen(ev: Event): void {
        const input = ev.target as HTMLInputElement;
        const file = input.files?.[0];
        if (!file) return;
        // Phase 1: just push the filename. A real impl would upload via a dataset endpoint
        // and use the server-relative path that endpoint returns. TODO marker: when the
        // LUT-upload endpoint lands, swap this for the upload flow.
        this.state.lut.update(o => ({ ...o, enabled: true, params: { ...o.params, luts: [...o.params.luts, { file: file.name, strength: 1.0, enabled: true }] } }));
        input.value = '';
    }
    exportStack(): void {
        // Calls DatasetService.exportCube — wire to your existing service method.
        // Leaving as a TODO: actual call goes here once you confirm the service signature
        // by reading frontend/src/app/services/dataset.ts. The method is `exportCube`
        // per [legacy/.../image-editor-modal.ts:344].
        console.warn('LUT export — wire DatasetService.exportCube here');
    }
    private patchLut(i: number, fn: (l: LutEntry) => LutEntry): void {
        this.state.lut.update(o => {
            const next = o.params.luts.slice();
            if (i >= 0 && i < next.length) next[i] = fn(next[i]);
            return { ...o, enabled: true, params: { ...o.params, luts: next } };
        });
    }
    lutBasename(path: string): string {
        const idx = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
        return idx >= 0 ? path.slice(idx + 1) : path;
    }
}
