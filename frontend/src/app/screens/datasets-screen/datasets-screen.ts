import { ChangeDetectionStrategy, Component, computed, effect, HostListener, inject, signal } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { Dataset, DatasetService, type MpxDistribution } from '../../services/dataset';
import { DatasetUploadService } from '../../services/dataset-upload.service';
import { ProjectService, type Project } from '../../services/project.service';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { ToastService } from '../../services/toast';
import { DatasetStore } from '../../state/dataset.store';
import { OverlayStore } from '../../state/overlay.store';
import { ScopeStore } from '../../state/scope.store';
import { datasetPreviewUrl } from '../../shared/media-preview';
import { formatBytes } from '../../shared/format-bytes';
import { FormatBytesPipe } from '../../shared/format-bytes.pipe';
import {
    SearchStore,
    type DatasetSearchField,
} from '../../state/search.store';
import { IcoComponent } from '../../icons/ico.component';
import { KpiTileComponent } from '../../ui/kpi-tile/kpi-tile.component';
import { ChipTagComponent } from '../../ui/chip-tag/chip-tag.component';
import { StatePillsComponent, StatePillsState, datasetStatePills } from '../../ui/state-pills/state-pills.component';

interface ProjectBadge {
    id: string;
    name: string;
    color: string;
}

/**
 * Filter-chip identifiers for the screen-level filter bar.
 *
 * - `needs-captioning` / `needs-masking` / `low-hps` / `missing` — four
 *   pre-built smart filters rendered as chips with live counts. Each chip
 *   only renders when its population (computed against the scope+search
 *   baseline) is non-zero — a clean dataset never sees them.
 * - `cat:${classifier}` — dynamic per-classifier chips added via the
 *   "+ Filter" picker popover's Category dimension. No `category` field
 *   exists on `Dataset` (services/dataset.ts), so we group by `classifier`.
 * - `tag:${value}` — dynamic per-tag chips added via the popover's Tags
 *   dimension. Sourced from `Dataset.tags?: string[]` (services/dataset.ts:28).
 */
type FilterKey =
    | 'needs-captioning'
    | 'needs-masking'
    | 'low-hps'
    | 'missing'
    | `cat:${string}`
    | `tag:${string}`;

/** Two-tier "+ Filter" picker popover state. */
type FilterPickerTier = 'dimensions' | 'category' | 'tags';

/** Sort keys for the Sort dropdown — Name / Created / Images / HPS. */
type SortKey = 'name' | 'created' | 'images' | 'hps';


/**
 * Threshold below which a dataset's median HPSv2 quality score is "low".
 * The plan suggested 5; the actual scores on `Dataset.median_quality_score`
 * are in the ~0.2–0.3 range (see {@link DatasetsScreen.hpsTone} which uses
 * 0.27/0.24 thresholds). We use 0.27 here to mirror the existing "warning"
 * boundary — anything not in the success band is "low".
 */
const LOW_HPS_THRESHOLD = 0.27;

/**
 * Shape returned by {@link DatasetService.getCacheStats}. Mirrors the backend
 * `/datasets/cache/stats` response: `total_bytes` (sum of `.cache/` subtrees),
 * `latent_bytes`, `embedding_bytes`, `cached_datasets`, and `dataset_root_bytes`
 * (full on-disk size of every `<dataset>/` folder — images + captions + masks
 * + `.cache/` + everything). Used by the CACHED tile sub-line and the DATASETS
 * tile's corner indicator.
 */
interface CacheStats {
    total_bytes: number;
    latent_bytes: number;
    embedding_bytes: number;
    cached_datasets: number;
    dataset_root_bytes: number;
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
    imports: [IcoComponent, KpiTileComponent, ChipTagComponent, StatePillsComponent, FormatBytesPipe],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './datasets-screen.html',
    styleUrl: './datasets-screen.css',
})
export class DatasetsScreen {
    private datasets = inject(DatasetStore);
    private datasetsApi = inject(DatasetService);
    private upload = inject(DatasetUploadService);
    private projects = inject(ProjectService);
    private rtc = inject(RuntimeConfigService);
    private toast = inject(ToastService);
    protected scope = inject(ScopeStore);
    protected overlay = inject(OverlayStore);
    protected search = inject(SearchStore);

    /** Dataset ids that belong to the active project, when one is scoped. */
    private projectDatasetIds = signal<Set<string>>(new Set());

    /** Dataset name whose project-picker dropdown is currently open. Empty
     *  string = none open. Tracked by name (not by reference) so the picker
     *  survives grid re-renders. */
    protected projectPickerOpenFor = signal<string>('');

    /** Per-dataset picker anchor alignment — computed on open from the trigger
     *  button's viewport position. Default ('right') keeps the panel extending
     *  leftward from the FolderPlus button (existing behavior); 'left' switches
     *  to right-extending so cards near the viewport's LEFT edge don't clip. */
    private pickerAlign = signal<Map<string, 'left' | 'right'>>(new Map());

    /** Projects available to add datasets to. Reads from {@link ProjectService.allProjects}
     *  which is loaded once on app boot by the shell — no fetch needed here. */
    protected availableProjects = computed<Project[]>(() => this.projects.allProjects());

    /** True while the fullscreen workspace overlay is mounted over the library. */
    private workspaceOpen = computed(() => this.overlay.workspace() !== null);

    constructor() {
        // Load the global dataset list once on mount. Idempotent: re-runs are
        // harmless because setAll replaces the entity map.
        void this.datasets.loadAll().catch(() => {
            // Errors surface as toasts via the entity-store base; nothing to do.
        });

        // Fetch global cache stats once on mount so the CACHED tile sub-line
        // can show "L latents · E embeds" sizes. Non-fatal — if the endpoint
        // fails, cacheStats stays null and the sub falls back to just the
        // "D / N datasets" fraction (see cachedSub() below).
        this.datasetsApi.getCacheStats().subscribe({
            next: (s: CacheStats) => this.cacheStats.set(s),
            error: () => undefined,
        });

        // Fetch cross-dataset MPx distribution once on mount so the IMAGES
        // tile can render its sub-line ("<avg> MP · <avg> avg") and the
        // mini-histogram below the count. Non-fatal — mpxStats stays null
        // and the tile falls back to the static "across datasets" sub.
        this.datasetsApi.getMpxDistribution().subscribe({
            next: (s: MpxDistribution) => this.mpxStats.set(s),
            error: () => undefined,
        });

        // Refresh the project-membership filter whenever scope changes OR when
        // the user returns from the workspace overlay. Switching scope from the
        // context-switcher or sidebar must update the grid immediately; and the
        // workspace "Add to project" pill can change membership WITHOUT changing
        // scope. Since this screen stays mounted beneath the overlay, closing
        // the workspace must re-sync membership — otherwise a just-added dataset
        // stays hidden. The fetch is skipped while the overlay is open (the grid
        // is hidden and the pre-add snapshot would be stale anyway).
        effect(() => {
            const pid = this.scope.projectId();
            if (this.workspaceOpen()) return;
            void this.refreshProjectMembership(pid);
        });
    }

    /** Source of truth — all datasets currently in the entity store. */
    private allDatasets = this.datasets.entities;

    // ── Filter chips + sort dropdown (Task 9) ─────────────────────────────
    //
    // The screen-level filter bar sits under the KPI rail and owns the
    // search input (the global topbar search is hidden on /datasets — see
    // topbar.component.html). Active chips AND together: a row must satisfy
    // every predicate to remain visible.

    /** Set of currently-active filter-chip keys. */
    protected activeFilters = signal<Set<FilterKey>>(new Set());

    /** Whether the "+ Filter" picker popover is currently open. */
    protected filterPickerOpen = signal(false);

    /** Which tier of the picker is on screen — root dimensions list or one of
     *  the value lists. Reset to `'dimensions'` every time the picker opens. */
    protected filterPickerTier = signal<FilterPickerTier>('dimensions');

    /** Sort dropdown key — defaults to Name. */
    protected sortKey = signal<SortKey>('name');

    /** Sort direction — clicking the same Sort key toggles asc⇄desc. */
    protected sortDir = signal<'asc' | 'desc'>('asc');

    /**
     * Distinct classifiers present in the loaded rows — drives the Category
     * dimension of the "+ Filter" picker. `Dataset` has no `category` field;
     * we group by `classifier`. Rows with no classifier are bucketed as 'other'.
     */
    protected categories = computed<string[]>(() =>
        Array.from(new Set(
            (this.allDatasets() ?? []).map(d => d.classifier || 'other')
        )).sort()
    );

    /**
     * Distinct tags across the loaded rows — drives the Tags dimension of the
     * "+ Filter" picker. Sourced from `Dataset.tags?: string[]`. Empty arrays
     * and missing fields are skipped; the result is alpha-sorted. The popover
     * suppresses the Tags dimension entirely when this list is empty.
     */
    protected availableTags = computed<string[]>(() => {
        const out = new Set<string>();
        for (const d of this.allDatasets() ?? []) {
            for (const t of d.tags ?? []) {
                if (t) out.add(t);
            }
        }
        return Array.from(out).sort();
    });

    protected isFilterActive(k: FilterKey): boolean {
        return this.activeFilters().has(k);
    }

    protected toggleFilter(k: FilterKey): void {
        this.activeFilters.update(s => {
            const next = new Set(s);
            if (next.has(k)) next.delete(k); else next.add(k);
            return next;
        });
    }

    /**
     * Builds a per-classifier chip key (`cat:${classifier}`) — bridges the
     * template's plain-string concatenation to the template-literal `FilterKey`
     * type so Angular's strict template checking accepts the chip handlers.
     */
    protected catKey(cat: string): FilterKey {
        return `cat:${cat}`;
    }

    /** Builds a per-tag chip key (`tag:${value}`) — same bridging role as `catKey`. */
    protected tagKey(tag: string): FilterKey {
        return `tag:${tag}`;
    }

    /**
     * "+ Filter" button click — toggles the picker open/closed. Always
     * returns to the root dimensions tier on (re-)open so the user starts
     * from the top of the navigation. `stopPropagation` keeps the host's
     * document-click handler (which dismisses the picker) from firing on
     * the same event that opened it.
     */
    protected toggleFilterPicker(event: MouseEvent): void {
        event.stopPropagation();
        const wasOpen = this.filterPickerOpen();
        if (!wasOpen) this.filterPickerTier.set('dimensions');
        this.filterPickerOpen.set(!wasOpen);
    }

    /** Adds a filter from the picker and closes the popover. Re-adding an
     *  already-active filter is a no-op for the Set, which is intentional —
     *  the picker still closes so the user sees the result. */
    protected addFilter(key: FilterKey): void {
        this.activeFilters.update(s => new Set([...s, key]));
        this.filterPickerOpen.set(false);
    }

    /** Removes a single active filter without affecting the others. Called
     *  from the X icon on each active dynamic chip. */
    protected removeFilter(key: FilterKey): void {
        this.activeFilters.update(s => {
            const next = new Set(s);
            next.delete(key);
            return next;
        });
    }

    /**
     * Sort dropdown change handler — pass a key to switch keys (resetting to
     * ascending) or to toggle direction when the current key is re-selected
     * (used by the direction-toggle button which calls `setSort(sortKey())`).
     */
    protected setSort(k: SortKey): void {
        if (this.sortKey() === k) {
            this.sortDir.set(this.sortDir() === 'asc' ? 'desc' : 'asc');
        } else {
            this.sortKey.set(k);
            this.sortDir.set('asc');
        }
    }

    // ── Filter pipeline (scope → search → activeFilters → sort) ────────
    //
    // Each stage is a discrete computed so the smart-filter counts can
    // share the scope+search baseline without re-running the active-chip
    // loop (which would zero the counts the moment a filter activates).

    /** Stage 1 — narrow to the active project's membership when scoped. */
    protected scopedDatasets = computed<Dataset[]>(() => {
        const all = this.allDatasets() ?? [];
        const pid = this.scope.projectId();
        return !pid ? all : all.filter(d => this.projectDatasetIds().has(d.id));
    });

    /** Stage 2 — apply the topbar search across the enabled fields. */
    protected searchedDatasets = computed<Dataset[]>(() => {
        const scoped = this.scopedDatasets();
        const query = this.search.query().trim().toLowerCase();
        if (!query) return scoped;
        const enabled = this.search.fields();
        // Safety net: if the user unchecks everything, fall back to name
        // so the result isn't empty-by-accident.
        const fields: ReadonlySet<DatasetSearchField> =
            enabled.size === 0 ? new Set(['name']) : enabled;
        return scoped.filter(d => this.matchesSearch(d, query, fields));
    });

    /**
     * Stage 3 — apply the active filter chips (AND across chips). Smart
     * filters (`needs-masking` / `low-hps` / `missing`) live alongside
     * dynamic per-value chips (`cat:` / `tag:`).
     */
    private filteredDatasets = computed<Dataset[]>(() => {
        const rows = this.searchedDatasets();
        const active = this.activeFilters();
        if (active.size === 0) return rows;
        return rows.filter(d => {
            for (const k of active) {
                if (!this.matchesFilter(d, k)) return false;
            }
            return true;
        });
    });

    /** Stage 4 — the public list consumed by the grid + every KPI computed. */
    protected visibleDatasets = computed<Dataset[]>(() => {
        const rows = this.filteredDatasets();
        const sk = this.sortKey();
        const sd = this.sortDir() === 'desc' ? -1 : 1;
        const cmp = (a: Dataset, b: Dataset): number => {
            switch (sk) {
                case 'name':
                    return a.name.localeCompare(b.name) * sd;
                case 'created':
                    // created_at is a Unix-seconds NUMBER on Dataset, not ISO.
                    return ((a.created_at || 0) - (b.created_at || 0)) * sd;
                case 'images':
                    return ((a.multimedia_count ?? 0) - (b.multimedia_count ?? 0)) * sd;
                case 'hps':
                    // Treat missing scores as -Infinity so unscored rows sink
                    // to the bottom in ascending order and to the top in
                    // descending order — predictable either way.
                    return (
                        (a.median_quality_score ?? -Infinity)
                        - (b.median_quality_score ?? -Infinity)
                    ) * sd;
            }
            return 0;
        };
        return [...rows].sort(cmp);
    });

    /**
     * Smart-filter populations computed against the scope+search baseline
     * (NOT the already-filtered visible list). Each count reflects what
     * WOULD match if that chip were the only active filter — composing
     * with other chips would otherwise drive the counts to zero the moment
     * another filter activates, which reads as a broken UI.
     */
    protected smartFilterCounts = computed(() => {
        const rows = this.searchedDatasets();
        let needsCaptioning = 0;
        let needsMasking = 0;
        let lowHps = 0;
        let missing = 0;
        for (const d of rows) {
            if ((d.caption_count ?? 0) < (d.multimedia_count ?? 0)) needsCaptioning++;
            if ((d.mask_count ?? 0) < (d.multimedia_count ?? 0)) needsMasking++;
            const v = d.median_quality_score;
            const score = (v == null || Number.isNaN(v)) ? Infinity : v;
            if (score < LOW_HPS_THRESHOLD) lowHps++;
            if (d.missing === true) missing++;
        }
        return { needsCaptioning, needsMasking, lowHps, missing };
    });

    /**
     * Active dynamic filter chips (everything except the three smart
     * filters). Drives the row of removable chips rendered to the left of
     * the "+ Filter" button. Labels strip the prefix for display.
     */
    protected activeDynamicFilters = computed<{ key: FilterKey; label: string }[]>(() => {
        const out: { key: FilterKey; label: string }[] = [];
        for (const k of this.activeFilters()) {
            if (k.startsWith('cat:')) {
                out.push({ key: k, label: k.slice(4) });
            } else if (k.startsWith('tag:')) {
                out.push({ key: k, label: k.slice(4) });
            }
        }
        return out;
    });

    /** Single-key predicate used by both `filteredDatasets` and any future
     *  count-style computeds that need to test rows in isolation. */
    private matchesFilter(d: Dataset, k: FilterKey): boolean {
        if (k === 'needs-captioning') {
            // "Not all images captioned" — caption_count < multimedia_count.
            // Mirrors the needs-masking shape so the two filters compose
            // identically and the chip pair behaves predictably.
            return (d.caption_count ?? 0) < (d.multimedia_count ?? 0);
        }
        if (k === 'needs-masking') {
            // "Not all images masked" — mask_count < multimedia_count.
            return (d.mask_count ?? 0) < (d.multimedia_count ?? 0);
        }
        if (k === 'low-hps') {
            // No score = treat as not-low (Infinity), matching the plan.
            const v = d.median_quality_score;
            const score = (v == null || Number.isNaN(v)) ? Infinity : v;
            return score < LOW_HPS_THRESHOLD;
        }
        if (k === 'missing') return d.missing === true;
        if (k.startsWith('cat:')) {
            return (d.classifier || 'other') === k.slice(4);
        }
        if (k.startsWith('tag:')) {
            return d.tags?.includes(k.slice(4)) ?? false;
        }
        return true;
    }

    private matchesSearch(
        d: Dataset,
        q: string,
        fields: ReadonlySet<DatasetSearchField>,
    ): boolean {
        if (fields.has('name') && d.name?.toLowerCase().includes(q)) return true;
        if (fields.has('classifier') && d.classifier?.toLowerCase().includes(q)) return true;
        if (fields.has('description') && d.description?.toLowerCase().includes(q)) return true;
        if (fields.has('trigger_word') && d.trigger_word?.toLowerCase().includes(q)) return true;
        if (fields.has('notes') && d.notes?.toLowerCase().includes(q)) return true;
        if (fields.has('tags') && d.tags?.some(t => t.toLowerCase().includes(q))) return true;
        return false;
    }

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

    // ── KPI rail enrichment (Task 8) ───────────────────────────────────
    //
    // Folds legacy <app-dataset-stats> content into the existing six tiles
    // (no STORAGE tile is added). The new sub-line computeds enrich
    // DATASETS / CAPTIONED / MASKED / CACHED / HPS without changing the
    // tile count or layout.

    /** Disk-usage / cache snapshot from `DatasetService.getCacheStats()`,
     *  loaded once on mount. `null` while the request is in flight or if
     *  the endpoint errored out — the CACHED sub falls back gracefully. */
    protected cacheStats = signal<CacheStats | null>(null);

    /** Cross-dataset MPx + mean image-size aggregate from
     *  `DatasetService.getMpxDistribution()`. Drives the IMAGES tile's
     *  sub-line and mini-histogram body. `null` while in flight or on
     *  error — the tile falls back to the static "across datasets" sub. */
    protected mpxStats = signal<MpxDistribution | null>(null);

    /** Count of datasets created within the last 7 days. `created_at` is a
     *  Unix-seconds NUMBER on Dataset (services/dataset.ts:12) — not an ISO
     *  string — so the comparison uses Math.floor(Date.now()/1000). */
    protected createdThisWeek = computed(() => {
        const cutoffSec = Math.floor(Date.now() / 1000) - 7 * 86400;
        return this.visibleDatasets().filter(d =>
            Number.isFinite(d.created_at) && d.created_at >= cutoffSec
        ).length;
    });

    /** Count of datasets flagged `missing: true` (folder gone from disk). */
    protected missingCount = computed(() =>
        this.visibleDatasets().filter(d => d.missing).length
    );

    /** Total disk size across every loaded dataset, summed from each row's
     *  `total_size_bytes`. Returns 0 when the list is empty. */
    protected totalSizeBytes = computed(() =>
        this.visibleDatasets().reduce((sum, d) => sum + (d.total_size_bytes || 0), 0)
    );

    /** Sub-line for the DATASETS tile — composes "+N this week · M missing".
     *  Each segment is omitted when zero / unknown. Falls back to "in scope"
     *  (the prior static value) when nothing is known. Total disk size is
     *  rendered separately as a corner indicator (see `.kpi-corner` in the
     *  template) so it doesn't crowd the sub-line. */
    protected datasetsSub = computed(() => {
        const parts: string[] = [];
        const delta = this.createdThisWeek();
        if (delta > 0) parts.push(`+${delta} this week`);
        const missing = this.missingCount();
        if (missing > 0) parts.push(`${missing} missing`);
        return parts.length === 0 ? 'in scope' : parts.join(' · ');
    });

    /** Sub-line for IMAGES — three-segment Hi-Fi format
     *  "<total image bytes> · <avg> MP avg · <avg> avg". The total image
     *  size (sum of each row's `total_size_bytes`) leads the line; the
     *  two averages come from the cross-dataset MPx aggregate. Both
     *  averaged segments carry an explicit "avg" suffix so the middle
     *  value doesn't read as a raw MPx count. If the MPx aggregate
     *  hasn't loaded (or no images have been scanned), we still surface
     *  the total alone when known, else fall back to the prior static
     *  "across datasets" string. */
    protected imagesSub = computed(() => {
        const m = this.mpxStats();
        const totalImageBytes = this.totalSizeBytes();
        if (!m || m.total_images === 0) {
            // No MPx data yet — show just the total if we have it, else fallback.
            return totalImageBytes > 0 ? formatBytes(totalImageBytes) : 'across datasets';
        }
        const mp = m.avg_megapixels.toFixed(1);
        const avgBytes = formatBytes(m.avg_size_bytes);
        const totalStr = formatBytes(totalImageBytes);
        return `${totalStr} · ${mp} MP avg · ${avgBytes} avg`;
    });

    /** Peak bucket count across the MPx histogram — used to normalize bar
     *  heights for the IMAGES tile's mini-histogram. Clamped to 1 so an
     *  empty/all-zero histogram doesn't divide-by-zero in the template. */
    protected mpxHistoMax = computed(() => {
        const buckets = this.mpxStats()?.buckets ?? [];
        return Math.max(1, ...buckets.map(b => b.count));
    });

    /** Sub-line for CAPTIONED — "X / Y ready" fraction or static fallback. */
    protected captionedSub = computed(() => {
        const k = this.kpis();
        if (k.images === 0) return 'ready for training';
        return `${k.captioned} / ${k.images} ready`;
    });

    /** Sub-line for MASKED — "X / Y with overlays" or static fallback. */
    protected maskedSub = computed(() => {
        const k = this.kpis();
        if (k.images === 0) return 'with overlays';
        return `${k.masked} / ${k.images} with overlays`;
    });

    /** Sub-line for CACHED — "D / N datasets · L latents · E embeds". Sizes
     *  pulled from `getCacheStats()`; missing fields render as 0 B. When
     *  cacheStats hasn't loaded yet, returns just the fraction (or the
     *  legacy "latents on disk" string if the entity list is empty too). */
    protected cachedSub = computed(() => {
        const total = this.visibleDatasets().length;
        const frac = total === 0 ? '' : `${this.kpis().cached} / ${total} datasets`;
        const s = this.cacheStats();
        if (!s) return frac || 'latents on disk';
        const latents = formatBytes(s.latent_bytes || 0);
        const embeds  = formatBytes(s.embedding_bytes || 0);
        return `${frac} · ${latents} latents · ${embeds} embeds`;
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
     * Dataset-level H/C/M pills: colored by % completeness (full ≥90% green,
     * mid ≥50% amber, low >0% red, none grey) via the shared
     * {@link datasetStatePills} builder, with per-pill coverage tooltips.
     * Denominator is `multimedia_count` (images) since the conditions apply per
     * image; harmonized count is `round(harmonization_score × images)` because the
     * list endpoint exposes only the aggregate ratio, not a per-file boolean.
     */
    protected stateOf(d: Dataset): StatePillsState {
        return datasetStatePills({
            total: d.multimedia_count ?? 0,
            captioned: d.caption_count ?? 0,
            masked: d.mask_count ?? 0,
            harmonizationScore: d.harmonization_score ?? 0,
        });
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
     * (dataset missing on disk, or no preview chosen yet). Stills resolve to
     * `${mediaBaseUrl}/${name}/${preview_image}`; video clips (mp4/webm/mkv/avi)
     * route through the thumbnail endpoint for a renderable first-frame poster
     * (a raw clip in an `<img>` is what left video-only datasets blank).
     */
    protected previewUrl(d: Dataset): string | null {
        if (!d.preview_image || d.missing) return null;
        return datasetPreviewUrl(this.rtc.apiUrl, this.rtc.mediaBaseUrl, d.name, d.preview_image);
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

    protected clearSearch(): void {
        this.search.query.set('');
    }

    /** Reset all active filter chips. Search query is preserved (separate concern). */
    protected clearFilters(): void {
        this.activeFilters.set(new Set());
    }

    protected openCard(d: Dataset): void {
        this.overlay.openWorkspace(d.id ?? d.name, 'browse');
    }

    /** Inline version-edit pencil on a card. Mirrors the workspace header's
     *  {@link DatasetWorkspaceComponent.editVersion}: opens the shared
     *  `version-edit` modal and reflects the saved version into the local
     *  store row on success. `stopPropagation` keeps the click off
     *  {@link openCard}. */
    protected editVersion(d: Dataset, event: Event): void {
        event.stopPropagation();
        this.overlay.openModal('version-edit', {
            datasetName: d.name,
            currentVersion: d.version,
            onSaved: (newVersion: string) => {
                this.datasets.upsertLocal({ ...d, version: newVersion });
            },
        });
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

    /** Dataset names currently showing the drop overlay. We track by name
     *  rather than by reference so we don't lose state across grid re-renders. */
    protected dragOverDatasets = signal<Set<string>>(new Set());

    protected isDragOver(name: string): boolean {
        return this.dragOverDatasets().has(name);
    }

    protected onCardDragOver(name: string, event: DragEvent): void {
        if (!event.dataTransfer?.types.includes('Files')) return;
        event.preventDefault();
        event.stopPropagation();
        event.dataTransfer.dropEffect = 'copy';
        const next = new Set(this.dragOverDatasets());
        next.add(name);
        this.dragOverDatasets.set(next);
    }

    protected onCardDragLeave(name: string, event: DragEvent): void {
        event.preventDefault();
        event.stopPropagation();
        // Suppress flicker: when the cursor moves to a CHILD of the card,
        // browsers fire dragleave on the card root before the child's
        // dragover fires. Only clear when actually leaving the card.
        const related = event.relatedTarget as Node | null;
        const card = event.currentTarget as Node;
        if (related && card.contains(related)) return;
        const next = new Set(this.dragOverDatasets());
        next.delete(name);
        this.dragOverDatasets.set(next);
    }

    /**
     * Global drop guard. Without these, a file dropped OUTSIDE any card (e.g.
     * the 12px grid gap) navigates the browser tab to the file and loses app
     * state. Card-level handlers call `stopPropagation` and short-circuit
     * before reaching these — only un-handled file drags fire the host guard.
     */
    @HostListener('document:dragover', ['$event'])
    onDocumentDragOver(event: DragEvent): void {
        if (event.dataTransfer?.types.includes('Files')) {
            event.preventDefault();
        }
    }

    @HostListener('document:drop', ['$event'])
    onDocumentDrop(event: DragEvent): void {
        if (event.dataTransfer?.types.includes('Files')) {
            event.preventDefault();
        }
    }

    protected onCardDrop(name: string, event: DragEvent): void {
        event.preventDefault();
        event.stopPropagation();
        const next = new Set(this.dragOverDatasets());
        next.delete(name);
        this.dragOverDatasets.set(next);
        const files = event.dataTransfer?.files;
        if (files && files.length > 0) {
            this.onUploadFiles(name, files);
        }
    }

    /**
     * Route a card drop / picker selection to the upload authority.
     *
     * For an EDIT (paired) dataset a dropped image is ambiguous — it could be a
     * training target ("after") or a control ("before") — so we open the
     * pair-role-chooser to let the user decide and route accordingly. For a
     * standard dataset every file is a target: delegate straight to
     * {@link DatasetUploadService.uploadTargets} (classified optimistic counts +
     * backgrounded rescan live there).
     */
    protected onUploadFiles(name: string, files: FileList | null): void {
        if (!name || !files || files.length === 0) return;
        const ds = this.datasets.entities().find(d => d.name === name);
        if (ds?.kind === 'edit') {
            this.overlay.openModal('pair-role-chooser', {
                datasetName: name,
                files: Array.from(files),
            });
            return;
        }
        this.upload.uploadTargets(name, files);
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

    /** Triggers a browser download of the portable export zip (+ metadata). */
    protected exportDataset(d: Dataset, event: Event): void {
        event.stopPropagation();
        window.open(this.datasetsApi.getExportUrl(d.name), '_blank');
    }

    /** Opens the import-dataset modal to upload a portable export zip. */
    protected openImportDataset(): void {
        this.overlay.openModal('import-dataset');
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
     * Both paths use the themed `confirm` modal. Global-scope delete carries
     * an opt-in "also delete files on disk" checkbox (replacing the legacy
     * chained double-confirm); project-scope is a plain destructive confirm.
     * The action fires only from the modal's `onConfirm`.
     */
    protected deleteDataset(d: Dataset, event: Event): void {
        event.stopPropagation();
        const projectId = this.scope.projectId();

        if (projectId) {
            this.overlay.openModal('confirm', {
                title: 'Remove from project?',
                message: `"${d.name}" will be removed from this project. It stays in the library.`,
                confirmLabel: 'Remove',
                destructive: true,
                onConfirm: () => {
                    this.projects.removeProjectDataset(projectId, d.id ?? d.name).subscribe({
                        next: () => {
                            this.toast.success(`Removed "${d.name}" from project.`);
                            // Refresh the scope filter and the global library list.
                            void this.refreshAfterDelete(projectId);
                            // Keep the scope-switcher / sidebar count in sync.
                            this.projects.bumpDatasetStat(projectId, -1);
                            this.projects.loadProjects();
                        },
                        error: (err: { error?: { detail?: string }; message?: string }) =>
                            this.toast.error('Failed to remove from project: ' + (err?.error?.detail || err?.message)),
                    });
                },
            });
            return;
        }

        // Global scope: one dialog with an opt-in "delete files" checkbox
        // (replaces the legacy chained double-confirm). Unticked = remove from
        // library only (files kept); ticked = permanently delete folder + files.
        this.overlay.openModal('confirm', {
            title: `Delete dataset "${d.name}"?`,
            message: `Removes "${d.name}" from the library. Tick below to also permanently `
                + `delete its folder and all files on disk — this cannot be undone.`,
            confirmLabel: 'Delete',
            destructive: true,
            checkboxLabel: 'Also delete files on disk (permanent)',
            onConfirm: (deleteFiles?: boolean) => {
                const wipe = !!deleteFiles;
                this.datasetsApi.deleteDataset(d.name, wipe).subscribe({
                    next: () => {
                        this.toast.success(wipe
                            ? `Dataset '${d.name}' and its files deleted.`
                            : `Dataset '${d.name}' removed from library.`);
                        void this.datasets.loadAll().catch(() => undefined);
                    },
                    error: (err: { error?: { detail?: string }; message?: string }) =>
                        this.toast.error('Failed to delete dataset: ' + (err?.error?.detail || err?.message)),
                });
            },
        });
    }

    /**
     * Toggle the per-card project-picker dropdown for the given dataset.
     *
     * When OPENING, measure the trigger button's viewport position and pick
     * an alignment for the dropdown panel: right-anchored by default (panel
     * extends LEFTWARD from the FolderPlus button), or left-anchored when the
     * default would overflow the viewport's left edge (panel extends RIGHTWARD
     * instead). This keeps the picker visible for cards on the LEFT side of
     * the grid where right-anchoring used to clip the panel off-screen.
     */
    protected toggleProjectPicker(name: string, event: MouseEvent): void {
        event.stopPropagation();
        const isOpening = this.projectPickerOpenFor() !== name;
        if (isOpening) {
            const trigger = event.currentTarget as HTMLElement | null;
            if (trigger) {
                const rect = trigger.getBoundingClientRect();
                // Choose the side with more room. Center-based test sidesteps
                // having to know the panel's rendered width (the previous
                // panelMinWidth=180 guess was too tight — rows + padding stretch
                // the panel well past 180px, so the check missed cards on the
                // left half of the grid). If the trigger's center sits in the
                // left half of the viewport, anchor the panel to its LEFT edge
                // (extends rightward into more space); otherwise anchor to the
                // RIGHT edge (extends leftward into more space).
                const center = rect.left + rect.width / 2;
                const align: 'left' | 'right' =
                    center < window.innerWidth / 2 ? 'left' : 'right';
                this.pickerAlign.update(m => new Map(m).set(name, align));
            }
        }
        this.projectPickerOpenFor.set(isOpening ? name : '');
    }

    /** Alignment for the picker panel — defaults to 'right' for cards that
     *  haven't been opened yet (matches the historical anchoring). */
    protected pickerAlignFor(name: string): 'left' | 'right' {
        return this.pickerAlign().get(name) ?? 'right';
    }

    protected closeProjectPicker(): void {
        if (this.projectPickerOpenFor() !== '') {
            this.projectPickerOpenFor.set('');
        }
    }

    /**
     * Adds a dataset to the picked project. Closes the picker first so a slow
     * network call doesn't leave it hanging. On success: toast + if the user is
     * currently scoped to the picked project, fold the new id into the in-screen
     * membership set so the dataset becomes visible there immediately. Re-adds
     * to a project the dataset is already in are a backend no-op.
     */
    protected addDatasetToProject(d: Dataset, projectId: string, projectName: string): void {
        this.closeProjectPicker();
        this.projects.addProjectDataset(projectId, d.id ?? d.name).subscribe({
            next: () => {
                this.toast.success(
                    `Dataset '${d.name}' added to project '${projectName}'.`,
                );
                if (this.scope.projectId() === projectId) {
                    this.projectDatasetIds.update(s => new Set([...s, d.id ?? d.name]));
                }
                // Refresh the scope-switcher / sidebar dataset count: optimistic
                // bump now, authoritative reload to reconcile.
                this.projects.bumpDatasetStat(projectId, 1);
                this.projects.loadProjects();
            },
            error: (err: { error?: { detail?: string }; message?: string }) =>
                this.toast.error(
                    'Failed to add dataset to project: ' + (err?.error?.detail || err?.message),
                ),
        });
    }

    /**
     * Closes any open dropdown when the user clicks elsewhere — both the
     * per-card project picker AND the filter-bar "+ Filter" popover.
     *
     * Each panel calls `stopPropagation` on its own clicks (see the
     * template), so picking a row in either panel still fires through
     * to its dedicated handler before this guard can fire.
     */
    @HostListener('document:click')
    onDocumentClick(): void {
        this.closeProjectPicker();
        if (this.filterPickerOpen()) this.filterPickerOpen.set(false);
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
