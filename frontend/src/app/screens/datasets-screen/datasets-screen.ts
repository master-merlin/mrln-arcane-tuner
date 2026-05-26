import { ChangeDetectionStrategy, Component, computed, effect, inject, signal } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { Dataset, DatasetService } from '../../services/dataset';
import { ProjectService } from '../../services/project.service';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { ToastService } from '../../services/toast';
import { DatasetStore } from '../../state/dataset.store';
import { OverlayStore } from '../../state/overlay.store';
import { ScopeStore } from '../../state/scope.store';
import { IcoComponent } from '../../icons/ico.component';
import { KpiTileComponent } from '../../ui/kpi-tile/kpi-tile.component';
import { ChipTagComponent } from '../../ui/chip-tag/chip-tag.component';
import { StatePillsComponent } from '../../ui/state-pills/state-pills.component';

interface ProjectBadge {
    id: string;
    name: string;
    color: string;
}

/**
 * Datasets screen — KPI rail + scope-aware library grid.
 *
 * Loads the dataset list via {@link DatasetStore.loadAll} the first time
 * the screen mounts (the store de-dupes if already loaded, since `entities`
 * stays warm). Scope filtering reads from {@link ProjectService.getProjectDatasets}
 * when a project scope is active, falling back to the global list otherwise.
 *
 * Backend gap: the `Dataset` shape doesn't carry `mask_coverage` or
 * `cache_size`, so per-dataset readiness flags for the H/C/M pills are
 * derived from `caption_coverage` + `mask_count > 0` + `has_cache`. The
 * KPI rail's MASKED total is summed from `mask_count` rather than a
 * dedicated `mask_coverage_count` field that doesn't exist yet.
 */
@Component({
    selector: 'app-datasets-screen',
    standalone: true,
    imports: [IcoComponent, KpiTileComponent, ChipTagComponent, StatePillsComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './datasets-screen.html',
    styleUrl: './datasets-screen.css',
})
export class DatasetsScreen {
    private datasets = inject(DatasetStore);
    private datasetsApi = inject(DatasetService);
    private projects = inject(ProjectService);
    private rtc = inject(RuntimeConfigService);
    private toast = inject(ToastService);
    protected scope = inject(ScopeStore);
    protected overlay = inject(OverlayStore);

    /** Dataset ids that belong to the active project, when one is scoped. */
    private projectDatasetIds = signal<Set<string>>(new Set());

    constructor() {
        // Load the global dataset list once on mount. Idempotent: re-runs are
        // harmless because setAll replaces the entity map.
        void this.datasets.loadAll().catch(() => {
            // Errors surface as toasts via the entity-store base; nothing to do.
        });

        // Reactively refresh the project-membership filter whenever scope
        // changes. Switching scope from the context-switcher or sidebar must
        // update the grid immediately — ngOnInit fired only on mount.
        effect(() => {
            const pid = this.scope.projectId();
            void this.refreshProjectMembership(pid);
        });
    }

    /** Source of truth — all datasets currently in the entity store. */
    private allDatasets = this.datasets.entities;

    /** Scope-filtered list: project membership when scoped, else everything. */
    protected visibleDatasets = computed<Dataset[]>(() => {
        const all = this.allDatasets() ?? [];
        const pid = this.scope.projectId();
        if (!pid) return all;
        const allowed = this.projectDatasetIds();
        return all.filter(d => allowed.has(d.id));
    });

    /** KPI rail aggregates. */
    protected kpis = computed(() => {
        const list = this.visibleDatasets();
        const images = list.reduce((acc, d) => acc + (d.multimedia_count ?? 0), 0);
        const captioned = list.reduce((acc, d) => acc + (d.caption_count ?? 0), 0);
        const masked = list.reduce((acc, d) => acc + (d.mask_count ?? 0), 0);
        const cached = list.reduce((acc, d) => acc + (d.has_cache ? 1 : 0), 0);
        return { datasets: list.length, images, captioned, masked, cached };
    });

    /**
     * HPS-Median KPI — aggregate quality across visible datasets. The histogram
     * bins each dataset's `median_quality_score` into 10 equal-width buckets
     * across the observed min..max range, mirroring the design's MiniHisto.
     */
    protected hpsStats = computed(() => {
        const scores = this.visibleDatasets()
            .map(d => d.median_quality_score)
            .filter((v): v is number => v != null && !Number.isNaN(v))
            .sort((a, b) => a - b);
        if (scores.length === 0) {
            return { median: null as number | null, min: 0, max: 0, buckets: new Array(10).fill(0) as number[] };
        }
        const n = scores.length;
        const mid = n >> 1;
        const median = n % 2 ? scores[mid] : (scores[mid - 1] + scores[mid]) / 2;
        const min = scores[0];
        const max = scores[n - 1];
        const range = max - min || 1;
        const buckets = new Array(10).fill(0);
        for (const s of scores) {
            let i = Math.floor(((s - min) / range) * 10);
            if (i >= 10) i = 9;
            buckets[i]++;
        }
        return { median, min, max, buckets };
    });

    protected hpsMedianLabel = computed(() => {
        const m = this.hpsStats().median;
        return m == null ? '—' : m.toFixed(4);
    });

    protected hpsRangeLabel = computed(() => {
        const s = this.hpsStats();
        if (s.median == null) return '';
        return `range ${s.min.toFixed(3)} – ${s.max.toFixed(3)}`;
    });

    protected hpsHistoMax = computed(() => {
        let m = 0;
        for (const b of this.hpsStats().buckets) if (b > m) m = b;
        return m || 1;
    });

    /** Project badge lookup for cards (only shown in Global scope). */
    protected projectBadge = computed<ProjectBadge | null>(() => {
        // TODO(backend): no per-dataset project-membership index is exposed
        // for the global view yet. Cards in global scope render without a
        // project badge until /datasets returns a `project_ids: string[]` field.
        return null;
    });

    /** Whether to show the project badge slot in card markup. */
    protected get scopeIsGlobal(): boolean {
        return this.scope.projectId() === null;
    }

    /** Re-fetches the active project's dataset list. Driven by the scope effect in the constructor. */
    private async refreshProjectMembership(pid: string | null): Promise<void> {
        if (!pid) {
            this.projectDatasetIds.set(new Set());
            return;
        }
        try {
            const rows = await firstValueFrom(this.projects.getProjectDatasets(pid));
            this.projectDatasetIds.set(new Set(rows.map(r => r.id)));
        } catch {
            this.projectDatasetIds.set(new Set());
        }
    }

    // ── Card UI helpers ────────────────────────────────────────────────

    /**
     * HPSv2 quality — backend exposes it as `median_quality_score` (median of per-image
     * `quality_score` values, only present once images have been scored). Returns '—' when
     * no image has been scored yet so the chip is hidden in {@link hpsTone}.
     */
    protected hpsLabel(d: Dataset): string {
        const v = d.median_quality_score;
        if (v === undefined || v === null || Number.isNaN(v)) return '—';
        return v.toFixed(4);
    }

    /** Tone for the HPS chip — matches legacy thresholds (≥0.27 good, ≥0.24 warn). */
    protected hpsTone(d: Dataset): 'success' | 'warning' | 'danger' | '' {
        const v = d.median_quality_score;
        if (v === undefined || v === null || Number.isNaN(v)) return '';
        if (v >= 0.27) return 'success';
        if (v >= 0.24) return 'warning';
        return 'danger';
    }

    /**
     * Readiness flags + per-pill coverage tooltips fed into <app-state-pills/>.
     *
     * Denominator is `multimedia_count` (images) since the H/C/M conditions
     * apply per image. Harmonized count is approximated as
     * `round(harmonization_score × images)` because the backend exposes only
     * the aggregate ratio, not a per-file harmonized boolean in the list endpoint.
     */
    protected stateOf(d: Dataset): {
        harmonized: boolean;
        captioned: boolean;
        masked: boolean;
        titles: { harmonized: string; captioned: string; masked: string };
    } {
        const total = d.multimedia_count ?? 0;
        const captioned = d.caption_count ?? 0;
        const masked = d.mask_count ?? 0;
        const harmonScore = d.harmonization_score ?? 0;
        const harmonized = Math.round(harmonScore * total);
        const pct = (n: number) => (total > 0 ? Math.round((n / total) * 100) : 0);
        const fmt = (label: string, n: number, percent: number) =>
            total > 0
                ? `${label} ${n}/${total} files (${percent}%)`
                : `${label}: no images yet`;
        return {
            harmonized: harmonScore > 0,
            captioned: !!d.caption_coverage,
            masked: masked > 0,
            titles: {
                harmonized: fmt('Harmonized', harmonized, Math.round(harmonScore * 100)),
                captioned: fmt('Captioned', captioned, pct(captioned)),
                masked: fmt('Masked', masked, pct(masked)),
            },
        };
    }

    /**
     * Compact relative time for the card's last-scanned badge ("12m ago",
     * "2h ago", "3d ago", "5w ago"). Returns "Never" when the dataset
     * has never been scanned. `last_scanned_at` is unix seconds.
     */
    protected lastScanRelative(d: Dataset): string {
        const ts = d.last_scanned_at;
        if (!ts) return 'Never';
        const diffSec = Math.max(0, Date.now() / 1000 - ts);
        if (diffSec < 60) return 'just now';
        const minutes = Math.floor(diffSec / 60);
        if (minutes < 60) return `${minutes}m ago`;
        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `${hours}h ago`;
        const days = Math.floor(hours / 24);
        if (days < 7) return `${days}d ago`;
        const weeks = Math.floor(days / 7);
        if (weeks < 5) return `${weeks}w ago`;
        const months = Math.floor(days / 30);
        if (months < 12) return `${months}mo ago`;
        const years = Math.floor(days / 365);
        return `${years}y ago`;
    }

    /** Absolute locale date/time for the badge's tooltip (legacy format). */
    protected lastScanAbsolute(d: Dataset): string {
        const ts = d.last_scanned_at;
        if (!ts) return 'Never scanned';
        return `Last scanned ${new Date(ts * 1000).toLocaleString()}`;
    }

    /** Pretty MB/GB. */
    protected sizeLabel(bytes: number | undefined): string {
        if (!bytes) return '0 MB';
        const mb = bytes / (1024 * 1024);
        if (mb < 1024) return `${mb.toFixed(1)} MB`;
        return `${(mb / 1024).toFixed(2)} GB`;
    }

    /** Track function for the dataset grid. */
    protected trackById = (_: number, d: Dataset) => d.id ?? d.name;

    /**
     * URL of the dataset's preview thumbnail, or `null` when none is available
     * (dataset missing on disk, or no preview chosen yet). Matches the pattern
     * used by the legacy dataset-card: `${mediaBaseUrl}/${name}/${preview_image}`.
     */
    protected previewUrl(d: Dataset): string | null {
        if (!d.preview_image || d.missing) return null;
        return `${this.rtc.mediaBaseUrl}/${encodeURIComponent(d.name)}/${d.preview_image}`;
    }

    /**
     * Hide the broken <img> when its src 404s — the @else branch already
     * renders the ImageOff fallback, so we just blank the failed img out
     * by setting display:none. Avoids flashing the browser broken-image
     * glyph over the gradient backdrop on missing-preview datasets.
     */
    protected onPreviewError(event: Event): void {
        const img = event.target as HTMLImageElement;
        img.style.display = 'none';
    }

    // ── Actions ────────────────────────────────────────────────────────

    protected openCard(d: Dataset): void {
        this.overlay.openWorkspace(d.id ?? d.name, 'browse');
    }

    protected openNewDataset(): void {
        this.overlay.openModal('dataset-form');
    }

    protected openRescan(): void {
        this.overlay.openModal('rescan');
    }

    /** Per-dataset rescan from the card action bar. */
    protected rescanDataset(d: Dataset, event: Event): void {
        event.stopPropagation();
        this.overlay.openModal('rescan', { datasetName: d.name });
    }

    /** Per-dataset analyze from the card action bar. */
    protected analyzeDataset(d: Dataset, event: Event): void {
        event.stopPropagation();
        this.overlay.openModal('analyze', { datasetName: d.name });
    }

    /** Per-dataset cache admin from the card action bar. Disabled in template when !has_cache. */
    protected cacheDataset(d: Dataset, event: Event): void {
        event.stopPropagation();
        if (!d.has_cache) return;
        this.overlay.openModal('cache', { datasetName: d.name });
    }

    /** Triggers a browser download of the dataset zip. Mirrors the legacy flow. */
    protected downloadDataset(d: Dataset, event: Event): void {
        event.stopPropagation();
        const url = this.datasetsApi.getDownloadUrl(d.name);
        window.open(url, '_blank');
    }

    /** Per-dataset edit-metadata — opens the dataset-form modal in edit mode. */
    protected editDataset(d: Dataset, event: Event): void {
        event.stopPropagation();
        this.overlay.openModal('dataset-form', { datasetId: d.id ?? d.name });
    }

    /**
     * Scope-aware delete.
     *
     * - **Global scope**: removes the dataset from the library (existing
     *   `DELETE /datasets/{name}` endpoint).
     * - **Project scope**: detaches the dataset from the active project via
     *   {@link ProjectService.removeProjectDataset}. The dataset remains in
     *   the library.
     *
     * A native `confirm()` gates both paths for now; the dedicated `confirm`
     * modal ships in Phase 8 and this call site will switch to
     * `overlay.openModal('confirm', ...)` then.
     */
    protected deleteDataset(d: Dataset, event: Event): void {
        event.stopPropagation();
        const projectId = this.scope.projectId();

        if (projectId) {
            if (!confirm(`Remove "${d.name}" from this project? It will stay in the library.`)) return;
            // TODO(frontend): replace native confirm with overlay.openModal('confirm', ...) when Phase 8 lands.
            this.projects.removeProjectDataset(projectId, d.id ?? d.name).subscribe({
                next: () => {
                    this.toast.success(`Removed "${d.name}" from project.`);
                    // Refresh the scope filter and the global library list.
                    void this.refreshAfterDelete(projectId);
                },
                error: (err: { error?: { detail?: string }; message?: string }) =>
                    this.toast.error('Failed to remove from project: ' + (err?.error?.detail || err?.message)),
            });
            return;
        }

        // Global scope: actual library delete.
        if (!confirm(`Delete "${d.name}" from the library? This cannot be undone.`)) return;
        this.datasetsApi.deleteDataset(d.name).subscribe({
            next: () => {
                this.toast.success(`Deleted "${d.name}".`);
                void this.datasets.loadAll().catch(() => {});
            },
            error: (err: { error?: { detail?: string }; message?: string }) =>
                this.toast.error('Failed to delete dataset: ' + (err?.error?.detail || err?.message)),
        });
    }

    private async refreshAfterDelete(projectId: string): Promise<void> {
        try {
            // Refresh project membership immediately so the scope-filtered grid
            // drops the removed dataset without waiting for the next scope tick.
            const rows = await firstValueFrom(this.projects.getProjectDatasets(projectId));
            this.projectDatasetIds.set(new Set(rows.map(r => r.id)));
        } catch {
            this.projectDatasetIds.set(new Set());
        }
    }
}
