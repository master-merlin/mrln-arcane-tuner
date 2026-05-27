import { ChangeDetectionStrategy, Component, EventEmitter, inject, input, Output, signal, effect } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { IcoComponent } from '../../../../icons/ico.component';
import { DatasetService } from '../../../../services/dataset';

export type ModelRestoreKind = 'denoise' | 'face' | 'upscale';

export interface ModelEntry {
    name: string;
    path: string;
    size_mb: number;
}

@Component({
    selector: 'app-model-restore-panel',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="enable-row">
            <label><input type="checkbox" [checked]="enabled()" (change)="enableChanged.emit($any($event.target).checked)"/> Enable</label>
        </div>

        <div class="field">
            <label>Models folder</label>
            <div class="folder-row">
                <input class="input mono" [value]="folder()" (input)="folderChanged.emit($any($event.target).value)"/>
                <button type="button" class="btn sm" (click)="scan()">
                    <app-ico name="RefreshCw" [size]="11"/> Scan
                </button>
            </div>
        </div>

        <div class="field">
            <label>Model</label>
            <select class="input" [value]="model() ?? ''" (change)="modelChanged.emit($any($event.target).value || null)">
                <option value="">— select —</option>
                @for (m of models(); track m.path) {
                    <option [value]="m.path">{{ m.name }} ({{ m.size_mb }} MB)</option>
                }
            </select>
            <div class="hint">{{ models().length }} model(s) found</div>
        </div>

        @if (kind() !== 'upscale') {
            <div class="field">
                <div class="row"><label>Strength</label><span class="mono">{{ (strength() * 100).toFixed(0) }}%</span></div>
                <input type="range" min="0" max="1" step="0.01" [value]="strength()" (input)="strengthChanged.emit(+$any($event.target).value)"/>
            </div>
        }

        <div class="field">
            <div class="row"><label>Tile size</label><span class="mono">{{ tileSize() }} px</span></div>
            <input type="range" min="128" max="1024" step="64" [value]="tileSize()" (input)="tileSizeChanged.emit(+$any($event.target).value)"/>
        </div>

        @if (kind() === 'face') {
            <label class="toggle-row">
                <input type="checkbox" [checked]="faceOnly()" (change)="faceOnlyChanged.emit($any($event.target).checked)"/>
                Face-only mask
            </label>
        }

        @if (kind() === 'upscale') {
            <div class="field">
                <label>Target scale</label>
                <div class="seg">
                    @for (s of [1, 2, 4, 8]; track s) {
                        <button type="button" [class.active]="targetScale() === s" (click)="targetScaleChanged.emit(s)">{{ s }}×</button>
                    }
                </div>
            </div>
            <div class="field">
                <label>Resize method</label>
                <select class="input" [value]="resizeMethod()" (change)="resizeMethodChanged.emit($any($event.target).value)">
                    <option value="lanczos">Lanczos</option>
                    <option value="bicubic">Bicubic</option>
                    <option value="bilinear">Bilinear</option>
                    <option value="nearest">Nearest</option>
                </select>
            </div>
        }
    `,
    styles: [`
        :host { display: flex; flex-direction: column; gap: 10px; }
        .enable-row label, .toggle-row { display: flex; gap: 6px; font-size: 12px; cursor: pointer; }
        .field { display: flex; flex-direction: column; gap: 4px; }
        .field .row { display: flex; justify-content: space-between; align-items: center; font-size: 11.5px; }
        .field .row .mono { color: var(--color-text-muted); font-size: 11px; }
        .field input[type=range] { width: 100%; accent-color: var(--color-violet); }
        .input { padding: 6px 8px; background: var(--color-surface-mid); border: 1px solid var(--color-border-subtle); border-radius: var(--radius-theme-sm); font-size: 11.5px; color: var(--color-text-primary); }
        .input.mono { font-family: var(--font-mono); }
        .folder-row { display: flex; gap: 6px; }
        .folder-row .input { flex: 1; }
        .hint { font-size: 10px; color: var(--color-text-muted); margin-top: 2px; }
        .seg { display: flex; gap: 2px; border: 1px solid var(--color-border-subtle); border-radius: var(--radius-theme-sm); overflow: hidden; }
        .seg button {
            flex: 1; padding: 5px 8px;
            background: var(--color-surface-mid);
            color: var(--color-text-muted);
            border: none; cursor: pointer; font-size: 11px;
        }
        .seg button.active { background: color-mix(in oklab, var(--color-violet) 18%, transparent); color: var(--color-violet); }
    `],
})
export class ModelRestorePanelComponent {
    kind = input.required<ModelRestoreKind>();
    enabled = input.required<boolean>();
    folder = input.required<string>();
    model = input.required<string | null>();
    strength = input<number>(0.6);
    tileSize = input.required<number>();
    faceOnly = input<boolean>(false);
    targetScale = input<number>(2);
    resizeMethod = input<string>('lanczos');

    @Output() enableChanged = new EventEmitter<boolean>();
    @Output() folderChanged = new EventEmitter<string>();
    @Output() modelChanged = new EventEmitter<string | null>();
    @Output() strengthChanged = new EventEmitter<number>();
    @Output() tileSizeChanged = new EventEmitter<number>();
    @Output() faceOnlyChanged = new EventEmitter<boolean>();
    @Output() targetScaleChanged = new EventEmitter<number>();
    @Output() resizeMethodChanged = new EventEmitter<string>();

    private datasets = inject(DatasetService);
    protected models = signal<ModelEntry[]>([]);

    constructor() {
        effect(() => { void this.scan(); });  // initial scan
    }

    async scan(): Promise<void> {
        try {
            const category = this.kind() === 'upscale' ? 'upscale' : 'restore';
            const fn = category === 'upscale'
                ? (this.datasets as any).listUpscaleModels
                : (this.datasets as any).listRestoreModels;
            if (typeof fn !== 'function') { this.models.set([]); return; }
            const resp: any = await firstValueFrom(fn.call(this.datasets, this.folder()));
            this.models.set(Array.isArray(resp?.models) ? (resp.models as ModelEntry[]) : []);
        } catch {
            this.models.set([]);
        }
    }
}
