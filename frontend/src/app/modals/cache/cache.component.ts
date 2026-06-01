import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';
import { IcoComponent } from '../../icons/ico.component';
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';
import { OverlayStore } from '../../state/overlay.store';

interface CacheModalData {
    datasetName?: string;
    datasetId?: string;
}

interface CategoryRow {
    name: 'Latents' | 'Embeddings' | 'Models';
    bytes: number;
    tone: 'brand' | 'success' | 'violet';
    types: string[];
}

/**
 * Cache admin modal — usage breakdown by category + per-category purge.
 *
 * Ports the core flow from `viewer-cache-admin-modal.ts`: list cache via
 * `DatasetService.listCache(name)`, aggregate per-category totals, and
 * trigger `purgeCache` with the chosen type filter when the user clicks
 * one of the per-category purge buttons.
 *
 * Simplified from the orphan modal: we don't expose the per-leaf checkbox
 * tree here — the design spec lists per-category clear buttons as the
 * surface for this modal. Per-leaf selection can be added later if UAT
 * asks for it.
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
                <!-- Totals -->
                <div class="ca-kpis">
                    <div class="kpi compact">
                        <div class="kpi-accent teal"></div>
                        <div class="kpi-label">TOTAL</div>
                        <div class="kpi-value">{{ formatMB(totalBytes()) }}<span class="unit">MB</span></div>
                        <div class="kpi-sub">across {{ categories().length }} categories</div>
                    </div>
                    @for (c of categories(); track c.name) {
                        <div class="kpi compact">
                            <div class="kpi-accent" [class]="c.tone"></div>
                            <div class="kpi-label">{{ c.name.toUpperCase() }}</div>
                            <div class="kpi-value">{{ formatMB(c.bytes) }}<span class="unit">MB</span></div>
                            <div class="kpi-sub">{{ c.types.length }} type{{ c.types.length === 1 ? '' : 's' }}</div>
                        </div>
                    }
                </div>

                <!-- Breakdown bar -->
                <div class="card">
                    <div class="card-head">
                        <div class="card-title">
                            <app-ico name="Database" [size]="11"/> Cache breakdown
                        </div>
                        <span class="mono ca-sub">{{ formatMB(totalBytes()) }} MB</span>
                    </div>
                    <div class="card-body">
                        <div class="ca-stack">
                            @for (c of categories(); track c.name) {
                                <div class="ca-stack-seg"
                                     [style.flex]="c.bytes || 1"
                                     [style.background]="toneColor(c.tone)"
                                     [attr.title]="c.name + ' · ' + formatMB(c.bytes) + ' MB'"></div>
                            }
                        </div>
                        <div class="ca-legend">
                            @for (c of categories(); track c.name) {
                                <span class="ca-legend-row mono">
                                    <span class="ca-legend-dot" [style.background]="toneColor(c.tone)"></span>
                                    {{ c.name }} {{ formatMB(c.bytes) }} MB
                                </span>
                            }
                        </div>
                    </div>
                </div>

                <!-- Per-category clear actions -->
                <div class="ca-actions">
                    @for (c of categories(); track c.name) {
                        <button class="btn danger-out" type="button"
                                [disabled]="purging() !== null || c.bytes === 0"
                                (click)="purgeCategory(c)">
                            <app-ico name="Trash2" [size]="13"/>
                            Clear {{ c.name }} ({{ formatMB(c.bytes) }} MB)
                        </button>
                    }
                </div>
            }
        </div>

        <div class="modal-foot">
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
        .ca-kpis {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-bottom: 14px;
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
            gap: 14px;
            font-size: 10.5px;
            color: var(--color-text-muted);
        }
        .ca-legend-row { display: flex; align-items: center; gap: 5px; }
        .ca-legend-dot { width: 8px; height: 8px; border-radius: 2px; display: inline-block; }
        .ca-actions {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 16px;
        }
        .card { margin-bottom: 14px; }
    `],
})
export class CacheModalComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    private datasetsApi = inject(DatasetService);
    private toast = inject(ToastService);

    protected loading = signal(false);
    protected purging = signal<string | null>(null);
    protected categories = signal<CategoryRow[]>([]);
    protected totalBytes = computed(() => this.categories().reduce((a, c) => a + c.bytes, 0));

    protected data: CacheModalData = (this.overlay.topModal()?.data as CacheModalData) ?? {};

    ngOnInit(): void {
        if (this.data.datasetName) this.load();
    }

    protected formatMB(bytes: number): string {
        return (bytes / (1024 * 1024)).toFixed(1);
    }

    protected toneColor(t: 'brand' | 'success' | 'violet'): string {
        return t === 'brand' ? 'var(--color-brand)'
             : t === 'success' ? 'var(--color-success)'
             : 'var(--color-violet)';
    }

    private load(): void {
        const name = this.data.datasetName!;
        this.loading.set(true);
        this.datasetsApi.listCache(name).subscribe({
            next: (res: { cache?: Record<string, Record<string, { size_bytes?: number; types?: Record<string, Record<string, unknown>> }>> }) => {
                const tree = res.cache ?? {};
                const byType = new Map<string, { bytes: number; typesSeen: Set<string> }>();

                for (const versions of Object.values(tree)) {
                    for (const verData of Object.values(versions)) {
                        const types = verData.types ?? {};
                        const verBytes = verData.size_bytes ?? 0;
                        // Apportion the version's bytes roughly by present types
                        const typeKeys = Object.keys(types).filter(k => Object.keys(types[k] ?? {}).length > 0);
                        if (typeKeys.length === 0) continue;
                        const share = verBytes / typeKeys.length;
                        for (const t of typeKeys) {
                            const entry = byType.get(t) ?? { bytes: 0, typesSeen: new Set<string>() };
                            entry.bytes += share;
                            entry.typesSeen.add(t);
                            byType.set(t, entry);
                        }
                    }
                }

                const rows: CategoryRow[] = [];
                const knownTones: Record<string, 'brand' | 'success' | 'violet'> = {
                    latents: 'brand',
                    embeddings: 'success',
                };
                // Latents
                const lat = byType.get('latents');
                if (lat) rows.push({ name: 'Latents', bytes: lat.bytes, tone: 'brand', types: ['latents'] });
                // Embeddings
                const emb = byType.get('embeddings');
                if (emb) rows.push({ name: 'Embeddings', bytes: emb.bytes, tone: 'success', types: ['embeddings'] });
                // Anything else lumps into "Models"
                const otherTypes = [...byType.entries()].filter(([k]) => !knownTones[k]);
                if (otherTypes.length) {
                    const sum = otherTypes.reduce((a, [, v]) => a + v.bytes, 0);
                    rows.push({ name: 'Models', bytes: sum, tone: 'violet', types: otherTypes.map(([k]) => k) });
                }

                this.categories.set(rows);
                this.loading.set(false);
            },
            error: (err: { error?: { detail?: string }; message?: string }) => {
                this.toast.error('Failed to load cache: ' + (err.error?.detail ?? err.message ?? 'unknown error'));
                this.loading.set(false);
            },
        });
    }

    protected purgeCategory(c: CategoryRow): void {
        const name = this.data.datasetName;
        if (!name) return;
        // eslint-disable-next-line no-alert
        if (!confirm(`Purge all ${c.name.toLowerCase()} cache for ${name}? This cannot be undone.`)) return;
        this.purging.set(c.name);
        this.datasetsApi.purgeCache(name, { types: c.types }).subscribe({
            next: (res: { deleted?: number; freed_bytes?: number }) => {
                const freed = res?.freed_bytes ?? 0;
                const n = res?.deleted ?? 0;
                this.toast.success(
                    freed > 0
                        ? `${c.name} cache purged — freed ${this.formatMB(freed)} MB (${n} item${n === 1 ? '' : 's'}).`
                        : `${c.name} cache purged.`,
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
