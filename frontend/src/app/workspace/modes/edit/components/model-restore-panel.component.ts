import { ChangeDetectionStrategy, Component, EventEmitter, inject, input, Output, signal, effect, computed } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { IcoComponent } from '../../../../icons/ico.component';
import { DatasetService } from '../../../../services/dataset';
import { ModelDownloadStore } from '../../../../state/model-download.store';

export type ModelRestoreKind = 'denoise' | 'face' | 'upscale';

export interface ModelEntry {
    name: string;
    path: string;
    size_mb: number;
}

export interface RegistryEntry {
    filename: string;
    description: string;
    size_mb: number;
    downloaded: boolean;
}

export interface CombinedEntry {
    key: string;            // 'curated::<filename>'
    filename: string;
    name: string;           // display name (filename for curated)
    path: string | null;    // disk path when installed; null when registry-only
    size_mb: number;
    installed: boolean;
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
            <div class="combo">
                @for (m of combined(); track m.key) {
                    <button type="button" class="combo-row"
                            [class.installed]="m.installed"
                            [class.selected]="model() === m.path"
                            [disabled]="isDownloading() === m.filename"
                            (click)="onPick(m)">
                        @if (activeFor(m.key)(); as dl) {
                            <span class="glyph">⏳</span>
                            <span class="name mono">{{ m.name }}</span>
                            <div class="dl-meta">
                                @if (dl.percent != null) {
                                    {{ dl.percent }}%
                                } @else {
                                    …
                                }
                            </div>
                            <div class="dl-bar">
                                <div class="fill"
                                     [style.width.%]="dl.percent ?? 0"
                                     [class.indeterminate]="dl.percent == null"></div>
                            </div>
                        } @else if (m.installed) {
                            <span class="glyph installed">✓</span>
                            <span class="name mono">{{ m.name }}</span>
                            <span class="size">{{ m.size_mb }} MB</span>
                        } @else {
                            <span class="glyph dl">⬇</span>
                            <span class="name mono">{{ m.name }}</span>
                            <span class="size">{{ m.size_mb }} MB</span>
                        }
                    </button>
                }
            </div>
            <div class="hint">{{ combined().length }} model(s) — disk + curated</div>
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
        .combo {
            display: flex; flex-direction: column;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-sm);
            overflow: hidden;
        }
        .combo-row {
            display: grid;
            grid-template-columns: 14px 1fr auto;
            grid-row-gap: 2px;
            align-items: center; gap: 6px;
            padding: 4px 6px;
            background: transparent; color: var(--color-text-primary);
            border: none; border-bottom: 1px solid var(--color-border-subtle);
            font-size: 11px; cursor: pointer; text-align: left;
        }
        .combo-row:last-child { border-bottom: none; }
        .combo-row:hover { background: color-mix(in oklab, var(--color-violet) 8%, transparent); }
        .combo-row.selected { background: color-mix(in oklab, var(--color-violet) 18%, transparent); color: var(--color-violet); }
        .combo-row[disabled] { opacity: 0.7; cursor: progress; }
        .combo-row .glyph { font-size: 11px; }
        .combo-row .glyph.installed { color: var(--color-success); }
        .combo-row .glyph.dl { color: var(--color-violet); }
        .combo-row .name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .combo-row .size { font-size: 10px; color: var(--color-text-muted); font-family: var(--font-mono); }
        .combo-row .dl-meta { font-size: 10px; color: var(--color-violet); }
        .combo-row .dl-bar {
            grid-column: 1 / 4;
            height: 2px; background: rgba(0,0,0,0.3); border-radius: 1px; overflow: hidden;
        }
        .combo-row .dl-bar .fill { height: 100%; background: var(--color-violet); transition: width 200ms; }
        .combo-row .dl-bar .fill.indeterminate {
            width: 30%;
            animation: slide-combo 1.4s ease-in-out infinite;
        }
        @keyframes slide-combo {
            0%   { transform: translateX(-100%); }
            100% { transform: translateX(330%); }
        }
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
    private downloads = inject(ModelDownloadStore);
    protected models = signal<ModelEntry[]>([]);
    protected registry = signal<RegistryEntry[]>([]);
    protected isDownloading = signal<string | null>(null);

    protected combined = computed<CombinedEntry[]>(() => {
        const disk = this.models();
        const reg = this.registry();
        const seen = new Set<string>();
        const out: CombinedEntry[] = [];
        // Disk-first
        for (const d of disk) {
            seen.add(d.name);
            out.push({
                key: `curated::${d.name}`, filename: d.name, name: d.name,
                path: d.path, size_mb: d.size_mb, installed: true,
            });
        }
        // Then registry entries not yet on disk
        for (const r of reg) {
            if (seen.has(r.filename)) continue;
            out.push({
                key: `curated::${r.filename}`, filename: r.filename, name: r.filename,
                path: null, size_mb: r.size_mb, installed: false,
            });
        }
        return out;
    });

    /** Returns the live download-progress signal for a combined-entry key. */
    activeFor(key: string) {
        return this.downloads.activeForKey(key);
    }

    constructor() {
        effect(() => { void this.scan(); });  // initial scan
    }

    async scan(): Promise<void> {
        try {
            const category = this.kind() === 'upscale' ? 'upscale' : 'restore';
            const fn = category === 'upscale'
                ? (this.datasets as any).listUpscaleModels
                : (this.datasets as any).listRestoreModels;
            const [diskResp, regResp] = await Promise.all([
                typeof fn === 'function'
                    ? firstValueFrom(fn.call(this.datasets, this.folder())) as Promise<any>
                    : Promise.resolve(null),
                firstValueFrom(this.datasets.getModelRegistry(category)) as Promise<any>,
            ]);
            this.models.set(Array.isArray(diskResp?.models) ? (diskResp.models as ModelEntry[]) : []);
            this.registry.set(Array.isArray(regResp?.models) ? (regResp.models as RegistryEntry[]) : []);
        } catch {
            this.models.set([]);
            this.registry.set([]);
        }
    }

    protected async onPick(entry: CombinedEntry): Promise<void> {
        if (entry.installed && entry.path) {
            this.modelChanged.emit(entry.path);
            return;
        }
        // Available — trigger download, then re-scan and auto-select.
        const category = this.kind() === 'upscale' ? 'upscale' : 'restore';
        this.isDownloading.set(entry.filename);
        try {
            await firstValueFrom(
                this.datasets.downloadModel(category, entry.filename, this.folder()),
            );
            await this.scan();
            const installed = this.models().find(m => m.name === entry.filename);
            if (installed) this.modelChanged.emit(installed.path);
        } catch {
            // Toast already surfaced via standard error handling; nothing to do here.
        } finally {
            this.isDownloading.set(null);
        }
    }
}
