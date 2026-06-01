import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';
import { IcoComponent } from '../../icons/ico.component';
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';
import { OverlayStore } from '../../state/overlay.store';

interface CacheModalData {
    datasetName?: string;
    datasetId?: string;
}

interface CacheTypeRow {
    /** Raw cache-type dir name sent to the purge API (`latents`, `te1`, …). */
    key: string;
    label: string;
}

interface CacheVersionRow {
    version: string;
    bytes: number;
    types: CacheTypeRow[];
}

interface CacheModelRow {
    name: string;
    bytes: number;
    tone: string;
    versions: CacheVersionRow[];
}

const TONES = ['brand', 'success', 'violet', 'teal', 'warning'];

/**
 * Cache admin modal — usage breakdown + **per model / version / type**
 * invalidation, mirroring the on-disk layout
 * (`.cache/{model}/{version}/{type}/…`).
 *
 * Restores the legacy `viewer-cache-admin-modal` capability the first redesign
 * port dropped (it could only clear a type aggregated across all models). You
 * can now clear a specific cache type (latents / text-embeddings te1/te2)
 * within a specific model version, a whole version, a whole model, or
 * everything. Drives `purgeCache({models, versions, types})` — the per-model
 * and per-version filters were the missing dimensions (the `versions` filter
 * was also added to the backend for this).
 */
@Component({
    selector: 'app-modal-cache',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">CACHE</div>
                <div class="modal-title">{{ data.datasetName ? data.datasetName + ' · cache' : 'Cache' }}</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body">
            @if (!data.datasetName) {
                <div class="ca-empty">
                    <app-ico name="Info" [size]="20"/>
                    Open a dataset workspace first — cache admin is per-dataset.
                </div>
            } @else if (loading()) {
                <div class="ca-empty">Loading cache info…</div>
            } @else if (totalBytes() === 0) {
                <div class="ca-empty">
                    <app-ico name="Database" [size]="20"/>
                    No cached data for this dataset yet.
                </div>
            } @else {
                <!-- Breakdown by model -->
                <div class="card">
                    <div class="card-head">
                        <div class="card-title">
                            <app-ico name="Database" [size]="11"/> Cache breakdown
                        </div>
                        <span class="mono ca-sub">{{ formatMB(totalBytes()) }} MB</span>
                    </div>
                    <div class="card-body">
                        <div class="ca-stack">
                            @for (m of models(); track m.name) {
                                <div class="ca-stack-seg"
                                     [style.flex]="m.bytes || 1"
                                     [style.background]="toneColor(m.tone)"
                                     [attr.title]="m.name + ' · ' + formatMB(m.bytes) + ' MB'"></div>
                            }
                        </div>
                        <div class="ca-legend">
                            @for (m of models(); track m.name) {
                                <span class="ca-legend-row mono">
                                    <span class="ca-legend-dot" [style.background]="toneColor(m.tone)"></span>
                                    {{ m.name }} {{ formatMB(m.bytes) }} MB
                                </span>
                            }
                        </div>
                    </div>
                </div>

                <!-- Per model → version → type purge -->
                @for (m of models(); track m.name) {
                    <div class="card ca-model">
                        <div class="card-head">
                            <div class="card-title">
                                <span class="ca-legend-dot" [style.background]="toneColor(m.tone)"></span>
                                {{ m.name }}
                            </div>
                            <div class="ca-head-right">
                                <span class="mono ca-sub">{{ formatMB(m.bytes) }} MB</span>
                                <button class="btn sm danger-out" type="button"
                                        [disabled]="purging() !== null"
                                        (click)="purgeModel(m)">Clear all</button>
                            </div>
                        </div>
                        <div class="card-body ca-versions">
                            @for (v of m.versions; track v.version) {
                                <div class="ca-version">
                                    <div class="ca-version-head">
                                        <span class="ca-ver-tag">v{{ v.version }}</span>
                                        <span class="mono ca-sub">{{ formatMB(v.bytes) }} MB</span>
                                        <button class="btn sm danger-out ca-ver-clear" type="button"
                                                [disabled]="purging() !== null"
                                                (click)="purgeVersion(m, v)">Clear version</button>
                                    </div>
                                    <div class="ca-type-row">
                                        @for (t of v.types; track t.key) {
                                            <button class="btn sm" type="button"
                                                    [disabled]="purging() !== null"
                                                    (click)="purgeType(m, v, t)">
                                                <app-ico name="Trash2" [size]="12"/> {{ t.label }}
                                            </button>
                                        }
                                    </div>
                                </div>
                            }
                        </div>
                    </div>
                }
            }
        </div>

        <div class="modal-foot">
            @if (data.datasetName && totalBytes() > 0) {
                <button class="btn danger-out" type="button"
                        style="margin-right: auto;"
                        [disabled]="purging() !== null"
                        (click)="purgeAll()">
                    <app-ico name="Trash2" [size]="13"/> Clear all cache
                </button>
            }
            <button class="btn ghost" type="button" (click)="overlay.closeModal()">Close</button>
        </div>
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .ca-empty {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 28px;
            justify-content: center;
            color: var(--color-text-muted);
            font-size: 13px;
        }
        .ca-sub { font-size: 11px; }
        .ca-stack {
            display: flex;
            height: 14px;
            border-radius: 4px;
            overflow: hidden;
            border: 1px solid var(--color-border-subtle);
            margin-bottom: 8px;
            background: var(--color-surface-mid);
        }
        .ca-stack-seg { height: 100%; }
        .ca-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 14px;
            font-size: 10.5px;
            color: var(--color-text-muted);
        }
        .ca-legend-row { display: flex; align-items: center; gap: 5px; }
        .ca-legend-dot { width: 8px; height: 8px; border-radius: 2px; display: inline-block; flex-shrink: 0; }
        .card { margin-bottom: 14px; }
        .ca-model .card-title { display: flex; align-items: center; gap: 7px; text-transform: none; letter-spacing: 0; font-family: var(--font-mono); }
        .ca-head-right { display: flex; align-items: center; gap: 10px; }
        .ca-versions { display: flex; flex-direction: column; gap: 12px; }
        .ca-version {
            border-left: 2px solid var(--color-border-default);
            padding-left: 12px;
        }
        .ca-version-head { display: flex; align-items: center; gap: 10px; margin-bottom: 7px; }
        .ca-ver-tag {
            font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
            color: var(--color-brand-light); font-family: var(--font-mono);
        }
        .ca-ver-clear { margin-left: auto; }
        .ca-type-row { display: flex; gap: 8px; flex-wrap: wrap; }
    `],
})
export class CacheModalComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    private datasetsApi = inject(DatasetService);
    private toast = inject(ToastService);

    protected loading = signal(false);
    protected purging = signal<string | null>(null);
    protected models = signal<CacheModelRow[]>([]);
    protected totalBytes = computed(() => this.models().reduce((a, m) => a + m.bytes, 0));

    protected data: CacheModalData = (this.overlay.topModal()?.data as CacheModalData) ?? {};

    ngOnInit(): void {
        if (this.data.datasetName) this.load();
    }

    protected formatMB(bytes: number): string {
        return (bytes / (1024 * 1024)).toFixed(1);
    }

    protected toneColor(tone: string): string {
        return `var(--color-${tone})`;
    }

    private typeLabel(key: string): string {
        if (key === 'latents') return 'Latents';
        if (key === 'te1') return 'Embeddings · te1';
        if (key === 'te2') return 'Embeddings · te2';
        if (key.startsWith('te')) return `Embeddings · ${key}`;
        if (key === 'embeddings') return 'Embeddings';
        return key.charAt(0).toUpperCase() + key.slice(1);
    }

    private load(): void {
        const name = this.data.datasetName!;
        this.loading.set(true);
        this.datasetsApi.listCache(name).subscribe({
            next: (res: { cache?: Record<string, Record<string, { size_bytes?: number; types?: Record<string, Record<string, unknown>> }>> }) => {
                const tree = res.cache ?? {};
                const rows: CacheModelRow[] = [];
                let toneIdx = 0;

                for (const [modelName, versions] of Object.entries(tree)) {
                    const verRows: CacheVersionRow[] = [];
                    let modelBytes = 0;
                    for (const [version, verData] of Object.entries(versions)) {
                        const bytes = verData.size_bytes ?? 0;
                        const types = verData.types ?? {};
                        const typeRows = Object.entries(types)
                            .filter(([, variants]) => variants && Object.keys(variants).length > 0)
                            .map(([k]) => ({ key: k, label: this.typeLabel(k) }))
                            .sort((a, b) => a.key.localeCompare(b.key));
                        if (typeRows.length === 0 && bytes === 0) continue;
                        modelBytes += bytes;
                        verRows.push({ version, bytes, types: typeRows });
                    }
                    if (verRows.length === 0) continue;
                    verRows.sort((a, b) => b.version.localeCompare(a.version, undefined, { numeric: true }));
                    rows.push({
                        name: modelName,
                        bytes: modelBytes,
                        tone: TONES[toneIdx++ % TONES.length],
                        versions: verRows,
                    });
                }

                rows.sort((a, b) => b.bytes - a.bytes);
                this.models.set(rows);
                this.loading.set(false);
            },
            error: (err: { error?: { detail?: string }; message?: string }) => {
                this.toast.error('Failed to load cache: ' + (err.error?.detail ?? err.message ?? 'unknown error'));
                this.loading.set(false);
            },
        });
    }

    protected purgeType(m: CacheModelRow, v: CacheVersionRow, t: CacheTypeRow): void {
        this.purge(
            `${m.name} v${v.version} · ${t.label}`,
            { models: [m.name], versions: [v.version], types: [t.key] },
            `${m.name}/${v.version}/${t.key}`,
        );
    }

    protected purgeVersion(m: CacheModelRow, v: CacheVersionRow): void {
        this.purge(
            `${m.name} v${v.version} (all types)`,
            { models: [m.name], versions: [v.version] },
            `${m.name}/${v.version}`,
        );
    }

    protected purgeModel(m: CacheModelRow): void {
        this.purge(`all cache for ${m.name}`, { models: [m.name] }, m.name);
    }

    protected purgeAll(): void {
        this.purge('ALL cache for this dataset', {}, '*');
    }

    private purge(
        label: string,
        options: { models?: string[]; versions?: string[]; types?: string[] },
        token: string,
    ): void {
        const name = this.data.datasetName;
        if (!name) return;
        // eslint-disable-next-line no-alert
        if (!confirm(`Purge ${label}? This cannot be undone.`)) return;
        this.purging.set(token);
        this.datasetsApi.purgeCache(name, options).subscribe({
            next: (res: { deleted?: number; freed_bytes?: number }) => {
                const freed = res?.freed_bytes ?? 0;
                const n = res?.deleted ?? 0;
                this.toast.success(
                    freed > 0
                        ? `Purged ${n} item${n === 1 ? '' : 's'} — freed ${this.formatMB(freed)} MB.`
                        : 'Cache purged.',
                );
                this.purging.set(null);
                this.load();
            },
            error: (err: { error?: { detail?: string }; message?: string }) => {
                this.toast.error('Purge failed: ' + (err.error?.detail ?? err.message ?? 'unknown error'));
                this.purging.set(null);
            },
        });
    }
}
