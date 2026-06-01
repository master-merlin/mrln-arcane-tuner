import {
    ChangeDetectionStrategy,
    Component,
    computed,
    effect,
    inject,
    signal,
} from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { OverlayStore, type WorkspaceMode } from '../state/overlay.store';
import { DatasetStore } from '../state/dataset.store';
import { MediaItemStore, MediaItem } from '../state/media-item.store';
import { CaptionCacheStore, CaptionRow } from '../state/caption-cache.store';
import { DatasetService, Dataset } from '../services/dataset';
import { ScopeStore } from '../state/scope.store';
import { ToastService } from '../services/toast';
import { RuntimeConfigService } from '../services/runtime-config.service';
import { SegmentedComponent, type SegOption } from '../ui/segmented/segmented.component';
import { IconButtonComponent } from '../ui/icon-button/icon-button.component';
import { ContextSwitcherComponent } from '../shell/context-switcher/context-switcher.component';
import { ProjectMembershipPillComponent } from './project-membership-pill/project-membership-pill.component';
import { IcoComponent } from '../icons/ico.component';
import { FilmstripScrubberComponent } from './filmstrip-scrubber/filmstrip-scrubber.component';
import { BrowseMode } from './modes/browse-mode';
import { DetailsMode } from './modes/details-mode';
import { EditMode } from './modes/edit-mode';

/**
 * Fullscreen dataset workspace overlay.
 *
 * Mounted by `<app-workspace-layer>` whenever {@link OverlayStore.workspace}
 * is non-null. Owns:
 *  - the topbar (back · dataset info · mode segmented · scope/actions/close)
 *  - the browse-mode secondary toolbar
 *  - the mode body (browse / details / edit, switched via `@switch`)
 *  - the bottom filmstrip scrubber
 *
 * State architecture:
 *   {@link MediaItemStore} is the SOURCE OF TRUTH for per-image
 *   metadata (enabled, has_mask, has_caption, dimensions, HPS, etc.).
 *   Mutations go through the store's `runOptimistic` methods —
 *   instant apply, rollback + toast on HTTP failure, WS
 *   `entity.changed` broadcast for cross-tab consistency.
 *
 *   Caption TEXT (`caption_content`, `masked_caption_content`) is
 *   too large to push through the WS bus, so it stays in
 *   {@link CaptionCacheStore} — a per-dataset Map keyed by
 *   `media_file`. The workspace's {@link pairs} computed merges the
 *   two sources back into the legacy pair shape that the orphan-tree
 *   grid + detail components still expect.
 *
 *   Mutation handlers below are the SOLE entry points; child
 *   components (browse-mode, details-mode) only emit intent events.
 */
@Component({
    selector: 'app-dataset-workspace',
    standalone: true,
    imports: [
        SegmentedComponent,
        IconButtonComponent,
        ContextSwitcherComponent,
        ProjectMembershipPillComponent,
        IcoComponent,
        FilmstripScrubberComponent,
        BrowseMode,
        DetailsMode,
        EditMode,
    ],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './dataset-workspace.component.html',
    styleUrl: './dataset-workspace.component.css',
})
export class DatasetWorkspaceComponent {
    protected overlay = inject(OverlayStore);
    protected scope = inject(ScopeStore);
    private datasets = inject(DatasetStore);
    private datasetsApi = inject(DatasetService);
    private mediaItems = inject(MediaItemStore);
    private captions = inject(CaptionCacheStore);
    private toast = inject(ToastService);
    private rtc = inject(RuntimeConfigService);

    /** Datasets whose `/pairs` we've fetched at least once (acts as the
     *  load-state marker; the actual rows now live in MediaItemStore). */
    private loadedDatasets = signal<Set<string>>(new Set());
    /** Resolved-on-demand dataset rows (for ids not in the store yet). */
    private extraDatasets = signal<Record<string, Dataset>>({});

    protected ws = computed(() => this.overlay.workspace());

    /** Resolves the current dataset record (store first, fallback fetch). */
    protected dataset = computed<Dataset | null>(() => {
        const w = this.ws();
        if (!w) return null;
        const fromStore = (this.datasets.entities() ?? []).find(
            (d: Dataset) => d.id === w.datasetId || d.name === w.datasetId,
        );
        if (fromStore) return fromStore;
        return this.extraDatasets()[w.datasetId] ?? null;
    });

    /**
     * Pair list for the active dataset, projected from MediaItemStore +
     * the local caption-text cache. Re-emits whenever either source
     * changes — store optimistic updates and WS `entity.changed` events
     * propagate here automatically. Legacy pair shape is preserved so
     * the orphan-tree grid + detail components don't need to know about
     * the store split.
     */
    protected pairs = computed<any[]>(() => {
        const d = this.dataset();
        if (!d) return [];
        const items = this.mediaItems.byDataset(d.name)();
        const captions = this.captions.byDataset()[d.name] ?? new Map();
        return items.map(item => projectPair(item, captions.get(item.media_file)));
    });

    /**
     * Pairs filtered by the active secondary-toolbar filter. Browse mode
     * shows this projection; Details / Edit (and the filmstrip) operate
     * on the full {@link pairs} list so navigation isn't constrained by
     * the active filter.
     */
    protected visiblePairs = computed<any[]>(() => {
        const list = this.pairs();
        switch (this.filter()) {
            case 'enabled':
                return list.filter(p => p?.metadata?.enabled !== false);
            case 'excluded':
                return list.filter(p => p?.metadata?.enabled === false);
            case 'captioned':
                return list.filter(p => !!p?.caption_content?.trim());
            case 'masked':
                return list.filter(p => !!p?.metadata?.has_mask);
            case 'low_hps':
                return list.filter(p => {
                    const q = p?.metadata?.quality_score;
                    return typeof q === 'number' && q < 0.24;
                });
            default:
                return list;
        }
    });

    /** Filmstrip-shaped rows: readiness flags + a thumbnail URL per pair.
     *  Uses the dataset's `/thumbnail` endpoint (256px WebP, generated on
     *  first request) rather than the full media URL — keeps the strip
     *  cheap to load even for big 4K/8K source images. */
    protected filmstripImages = computed(() => {
        const d = this.dataset();
        const name = d?.name ?? '';
        const rev = this.mediaItems.mediaRev();
        return this.pairs().map(p => ({
            harmonized: !!p?.metadata?.has_overlay,
            captioned: !!(p?.caption_content?.trim()),
            masked: !!p?.metadata?.has_mask,
            thumbnailUrl: p?.media_file && name
                ? `${this.datasetsApi.thumbnailUrl(name, p.media_file)}&t=${rev}`
                : undefined,
            mediaType: p?.media_type ?? 'image',
        }));
    });

    /**
     * Session-scoped flag — the first edit (caption save / crop / mask
     * apply / editor bake) triggers a single ``bumpVersion('patch')``.
     * Subsequent edits in the same workspace session no-op. Resets when
     * the workspace is re-opened (component re-mount). Mirrors legacy
     * ``hasBumpedPatchInSession`` (dataset-viewer.ts:514).
     */
    private hasBumpedPatchInSession = false;

    /** Collapsed state for the bottom filmstrip; toggled by the centered arrow. */
    protected filmstripCollapsed = signal<boolean>(false);
    protected toggleFilmstrip(): void {
        this.filmstripCollapsed.update(v => !v);
    }


    protected modeOptions: ReadonlyArray<SegOption<WorkspaceMode>> = [
        { value: 'browse', label: 'Browse', icon: 'Grid3x3' },
        { value: 'details', label: 'Details', icon: 'Image' },
        { value: 'edit', label: 'Edit', icon: 'Sliders' },
    ];

    /** Browse-mode secondary toolbar state. Local for now — wiring the filter
     *  selection through to the grid is a follow-up; today the tabs are
     *  selectable visuals only. */
    protected filter = signal<
        'all' | 'enabled' | 'excluded' | 'captioned' | 'masked' | 'low_hps'
    >('all');
    protected density = signal<number>(5);
    /** Render masked variants of images + masked_caption_content when true. */
    protected showMasked = signal<boolean>(false);
    /** Render edited overlays in place of originals when true (legacy default). */
    protected showOverlay = signal<boolean>(true);

    /** True when at least one pair has a masked-image flag — drives toggle enablement. */
    protected hasMaskedImages = computed<boolean>(() =>
        this.pairs().some(p => !!p?.metadata?.has_masked),
    );
    /** True when at least one pair has an editor-produced overlay file. */
    protected hasOverlayImages = computed<boolean>(() =>
        this.pairs().some(p => !!p?.metadata?.has_overlay),
    );

    /** Training-readiness pair count (enabled + has caption) over total. */
    protected trainingCounts = computed<{ ready: number; total: number }>(() => {
        const list = this.pairs();
        const total = list.length;
        const ready = list.reduce((n, p) =>
            n + (p?.metadata?.enabled !== false && !!p?.caption_content?.trim() ? 1 : 0), 0);
        return { ready, total };
    });

    /** Per-tab counts shown next to the filter chips. */
    protected filterCounts = computed<{
        all: number; enabled: number; excluded: number;
        captioned: number; masked: number; lowHps: number;
    }>(() => {
        const list = this.pairs();
        let enabled = 0, excluded = 0, captioned = 0, masked = 0, lowHps = 0;
        for (const p of list) {
            if (p?.metadata?.enabled === false) excluded++; else enabled++;
            if (p?.caption_content?.trim()) captioned++;
            if (p?.metadata?.has_mask) masked++;
            const q = p?.metadata?.quality_score;
            if (typeof q === 'number' && q < 0.24) lowHps++;
        }
        return { all: list.length, enabled, excluded, captioned, masked, lowHps };
    });

    /** HPS range readout — min/max quality_score over the loaded pairs. */
    protected hpsRangeLabel = computed<string | null>(() => {
        const xs = this.pairs()
            .map(p => p?.metadata?.quality_score)
            .filter((q): q is number => typeof q === 'number');
        if (xs.length === 0) return null;
        const lo = Math.min(...xs);
        const hi = Math.max(...xs);
        return `${lo.toFixed(2)} – ${hi.toFixed(2)}`;
    });

    /** Eyebrow row shows `image N / M` whenever we're not in browse mode. */
    protected showImageIndex = computed<boolean>(() => {
        const m = this.ws()?.mode;
        return m === 'details' || m === 'edit';
    });

    /** Media file of the pair at the active cursor — null when no
     *  pair is loaded. Drives the browse-mode tile highlight + scroll. */
    protected activeMediaFile = computed<string | null>(() => {
        const w = this.ws();
        if (!w) return null;
        const list = this.pairs();
        const idx = w.imageIndex;
        return idx >= 0 && idx < list.length ? (list[idx]?.media_file ?? null) : null;
    });

    constructor() {
        // Ensure the dataset store is hydrated so the lookup above succeeds.
        // No-op if `loadAll` already ran on the Datasets screen.
        void this.datasets.loadAll().catch(() => undefined);

        // Effect 1 — resolve the dataset row.
        //
        // Tracks only `ws()`. When the workspace opens by id BEFORE the
        // dataset store has hydrated, the by-name HTTP fetch will 404
        // (endpoint is keyed by name, not id). That's fine — we silently
        // return, and Effect 2 will pick up the row once `loadAll()`
        // resolves and `dataset()` becomes non-null.
        effect(() => {
            const w = this.ws();
            if (!w) return;
            void this.ensureDatasetRow(w.datasetId);
        });

        // Effect 2 — load pairs once both workspace and dataset are known.
        //
        // Tracks BOTH `ws()` and `dataset()`. This is the key dependency
        // fix: without `dataset()` in the read path the effect would not
        // re-run when `DatasetStore.loadAll()` later populates the
        // store, leaving pairs empty for ever.
        effect(() => {
            const w = this.ws();
            const d = this.dataset();
            if (!w || !d) return;
            void this.ensurePairsLoaded(d.name);
        });

        // Patch-bump trigger: any bytes-changing op (crop, mask apply,
        // editor bake) bumps MediaItemStore.mediaRev. We piggyback on
        // that — first increment per session counts as "first edit".
        // Capture the initial value as the baseline so we don't fire on
        // the constructor-time signal read; only subsequent increments
        // trigger.
        let mediaRevBaseline = this.mediaItems.mediaRev();
        effect(() => {
            const rev = this.mediaItems.mediaRev();
            if (rev !== mediaRevBaseline) {
                mediaRevBaseline = rev;
                this.ensurePatchBump();
            }
        });

    }

    /** Resolve the dataset row (store first, fallback HTTP by id-or-name). */
    private async ensureDatasetRow(idOrName: string): Promise<void> {
        const existing =
            (this.datasets.entities() ?? []).find(
                (d: Dataset) => d.id === idOrName || d.name === idOrName,
            ) ?? this.extraDatasets()[idOrName] ?? null;
        if (existing) return;

        try {
            const row = await firstValueFrom(this.datasetsApi.getDataset(idOrName));
            this.extraDatasets.update(m => ({ ...m, [idOrName]: row }));
        } catch {
            // 404 is expected when the id can't be resolved as a name —
            // `loadAll()` (started in the constructor) will hydrate the
            // store and Effect 2 will then pick the row up via `dataset()`.
            return;
        }
    }

    /**
     * Fetch `/pairs` once per dataset, route metadata into MediaItemStore
     * (the source of truth), and copy caption text into the local cache.
     * Idempotent — re-entry while a fetch is in flight just resolves to
     * the same `loadedDatasets` flip.
     */
    private async ensurePairsLoaded(name: string): Promise<void> {
        if (this.loadedDatasets().has(name)) return;
        // Mark loaded BEFORE the await so re-entries don't double-fetch.
        this.loadedDatasets.update(s => new Set(s).add(name));
        try {
            const pairs = (await firstValueFrom(
                this.datasetsApi.getDatasetPairs(name),
            )) as any[];
            const captions = new Map<string, CaptionRow>();
            for (const p of pairs ?? []) {
                if (!p?.media_file) continue;
                this.mediaItems.upsertFromPair(name, p);
                captions.set(p.media_file, {
                    caption_content: p.caption_content,
                    masked_caption_content: p.masked_caption_content,
                });
            }
            this.captions.seed(name, captions);
        } catch {
            // Leave `loadedDatasets` flipped — the user can manually
            // refresh; further auto-retries would just spam the API.
            this.captions.seed(name, new Map());
        }
    }

    /** Pretty-prints an HTTP error payload's `detail` (or falls back to message). */
    private errMsg(err: any, fallback: string): string {
        return `${fallback}: ${err?.error?.detail || err?.message || 'unknown error'}`;
    }

    protected openMass(kind: 'mass-caption' | 'mass-mask' | 'mass-edit'): void {
        const d = this.dataset();
        this.overlay.openModal(kind, {
            datasetId: this.ws()?.datasetId,
            datasetName: d?.name,
            // Workspace owns the per-session patch-bump flag; mass ops
            // are session-meaningful edits and should count toward it.
            // Idempotent — `ensurePatchBump` itself short-circuits on
            // subsequent calls, so back-to-back mass runs only bump once.
            onCompleted: () => this.ensurePatchBump(),
        });
    }

    /** Per-dataset Analyze / Cache / Rescan actions from the topbar. */
    protected openAction(kind: 'analyze' | 'cache' | 'rescan'): void {
        const d = this.dataset();
        if (!d) return;
        if (kind === 'cache' && !d.has_cache) return;
        this.overlay.openModal(kind, { datasetName: d.name });
    }

    protected setFilter(v: 'all' | 'enabled' | 'excluded' | 'captioned' | 'masked' | 'low_hps'): void {
        this.filter.set(v);
    }

    protected onDensityChange(v: number | string): void {
        const n = typeof v === 'string' ? parseInt(v, 10) : v;
        if (Number.isFinite(n)) this.density.set(Math.max(3, Math.min(7, n as number)));
    }

    protected toggleMaskedView(): void {
        if (this.hasMaskedImages()) this.showMasked.update(v => !v);
    }
    protected toggleOverlayView(): void {
        if (this.hasOverlayImages()) this.showOverlay.update(v => !v);
    }

    /** Re-include every excluded pair. MediaItemStore handles optimistic
     *  apply + rollback + toast. */
    protected enableAll(): void {
        const d = this.dataset();
        if (!d) return;
        void this.mediaItems.enableAll(d.name);
    }

    protected onSeek(idx: number): void {
        this.overlay.setWorkspaceImage(idx);
    }

    protected onModeChange(mode: WorkspaceMode): void {
        this.overlay.setWorkspaceMode(mode);
    }

    /** Toggle a pair's `enabled` flag via the store (cross-tab + rollback). */
    protected onToggleExclusion(event: { media_file: string; enabled: boolean }): void {
        const d = this.dataset();
        if (!d) return;
        void this.mediaItems.toggleEnabled(d.name, event.media_file, event.enabled);
    }

    /**
     * Persist a caption edit. Two-layer optimistic:
     *   - MediaItemStore.saveCaption stamps `caption_file` + `has_caption`
     *     (with its own rollback baked in).
     *   - The local caption-text cache snap-stamps the new content; on
     *     store failure we restore the previous text.
     */
    protected onSaveCaption(
        event: { pair: any; content: string; isMasked: boolean },
    ): void {
        const d = this.dataset();
        if (!d) return;
        const { pair, content, isMasked } = event;
        if (!pair?.media_file) return;
        const mediaFile: string = pair.media_file;
        // Masked captions live under ``masked/<stem>.txt`` on disk — legacy
        // parity with ``dataset-viewer.saveCurrentCaption``. ``pair.caption_file``
        // always points to the PLAIN caption file even when the masked variant
        // exists, so the masked branch must compose its path from the media
        // stem, not the field. Plain branch keeps the existing fallback.
        const stem = mediaFile.substring(0, mediaFile.lastIndexOf('.'));
        const filename = isMasked
            ? `masked/${stem}.txt`
            : (pair.caption_file || `${stem}.txt`);

        // Snapshot + apply via the shared cache (so the grid repaints live).
        // Non-reactive snapshot for optimistic-rollback restore.
        const existingRow = this.captions.get(d.name).get(mediaFile);
        const hadRow = existingRow !== undefined;
        const prevRow = existingRow ?? {};
        this.captions.setCaption(d.name, mediaFile, content, isMasked);

        void this.mediaItems
            .saveCaption(d.name, mediaFile, filename, content)
            .then(result => {
                if (result.ok) {
                    this.ensurePatchBump();
                    this.toast.success('Caption saved.');
                } else {
                    // Roll back the local text cache too.
                    if (hadRow) this.captions.setRow(d.name, mediaFile, prevRow);
                    else this.captions.remove(d.name, mediaFile);
                }
            });
    }

    /**
     * Delete a pair. Optimistic at both layers: MediaItemStore removes
     * the row; local caption cache drops the text entry; the cursor
     * clamps to the new tail or the workspace closes if empty. On
     * failure the store rolls back its row — we restore the caption
     * row and the cursor.
     */
    protected onDeletePairRequested(pair: any): void {
        const d = this.dataset();
        if (!d || !pair?.media_file) return;
        if (!confirm('Delete this entry?')) return;
        const mediaFile: string = pair.media_file;

        // Snapshot caption + cursor before the optimistic apply.
        const w = this.ws();
        const prevIdx = w?.imageIndex ?? 0;
        // Non-reactive snapshot for optimistic-rollback restore.
        const prevRow = this.captions.get(d.name).get(mediaFile);

        // Drop the caption row locally; MediaItemStore.deletePair will
        // remove the metadata row optimistically and roll back on error.
        this.captions.remove(d.name, mediaFile);

        void this.mediaItems.deletePair(d.name, mediaFile).then(result => {
            if (result.ok) {
                // After-removal: clamp cursor / close if empty.
                const remaining = this.mediaItems.byDataset(d.name)().length;
                if (remaining === 0) {
                    this.overlay.closeWorkspace();
                } else if (w && w.imageIndex >= remaining) {
                    this.overlay.setWorkspaceImage(remaining - 1);
                }
                this.toast.success('Entry deleted.');
            } else {
                // Restore caption row + cursor; store rolled itself back.
                if (prevRow) this.captions.setRow(d.name, mediaFile, prevRow);
                if (w) this.overlay.setWorkspaceImage(prevIdx);
            }
        });
    }

    /** Delete the mask for a pair — store handles optimistic flip + rollback. */
    protected onDeleteMaskRequested(pair: any): void {
        const d = this.dataset();
        if (!d || !pair?.metadata?.has_mask) return;
        if (!confirm('Delete the mask for this image?')) return;
        void this.mediaItems
            .deleteMask(d.name, pair.media_file)
            .then(result => {
                if (result.ok) this.toast.success('Mask deleted.');
            });
    }

    /**
     * Re-fetch `/pairs` after a server-driven mutation that we don't
     * model locally — currently only mask generation, where the server
     * produces a new file and the workspace needs the fresh metadata
     * (mask size, dimensions, etc.) to render the preview.
     *
     * Clears both the load marker (so ensurePairsLoaded re-runs) and
     * the caption cache for this dataset. MediaItemStore entries are
     * left in place; upsertFromPair will overlay new values on top
     * (preserving any entries the server might no longer report — see
     * the "stale-on-reload caveat" in MediaItemStore.loadForDataset).
     */
    protected refreshPairs(): void {
        const d = this.dataset();
        if (!d) return;
        this.loadedDatasets.update(s => {
            const n = new Set(s);
            n.delete(d.name);
            return n;
        });
        this.captions.clear(d.name);
        void this.ensurePairsLoaded(d.name);
    }

    /**
     * Fire the per-session patch bump exactly once. Idempotent: subsequent
     * calls are no-ops. On HTTP failure the flag clears so the next call
     * can retry — but failures don't surface as toasts (auto-bump
     * shouldn't interrupt the user's flow); we just log and let the
     * next edit trigger the retry.
     */
    private ensurePatchBump(): void {
        if (this.hasBumpedPatchInSession) return;
        const d = this.dataset();
        if (!d) return;
        this.hasBumpedPatchInSession = true;
        this.datasetsApi.bumpVersion(d.name, 'patch').subscribe({
            next: (res: any) => {
                this.datasets.upsertLocal({ ...d, version: res.version });
            },
            error: (err: any) => {
                console.warn('[workspace] auto patch bump failed', err);
                this.hasBumpedPatchInSession = false;
            },
        });
    }

    /**
     * Manually bump the dataset to the next MAJOR version. Mirrors the
     * legacy ``manualBump`` (dataset-viewer.ts:591) — confirm, POST to
     * /bump?type=major, stamp the new version on the cached dataset row.
     * Auto-bumps for ordinary edits live in {@link ensurePatchBump};
     * this is the explicit-intent MAJOR escalation.
     *
     * Note: deliberately does NOT set ``hasBumpedPatchInSession``. Legacy
     * parity — a manual MAJOR bump followed by an edit in the same
     * session also fires the per-session patch bump, yielding e.g.
     * 1.0.0 → 2.0.0 → 2.0.1. Two distinct user-meaningful operations,
     * two bumps. Mirrors {@link ../components/dataset/dataset-viewer/dataset-viewer.ts}
     * ``manualBump`` which likewise leaves the flag untouched.
     */
    protected async bumpMajor(): Promise<void> {
        const d = this.dataset();
        if (!d) return;
        if (!confirm(`Bump "${d.name}" to the next MAJOR version?`)) return;
        try {
            const res = await firstValueFrom(
                this.datasetsApi.bumpVersion(d.name, 'major'),
            );
            const newVersion = (res as { version: string }).version;
            this.datasets.upsertLocal({ ...d, version: newVersion });
            this.toast.success(`Version bumped to ${newVersion}`);
        } catch (err: any) {
            this.toast.error(this.errMsg(err, 'Failed to bump version'));
        }
    }

    /**
     * Open the version-edit modal — companion to {@link bumpMajor} for
     * cases where an accidental bump pushed the version off. Reuses the
     * same ``upsertLocal`` reconciliation path so Browse / Details stay
     * in sync without an extra HTTP fetch. Per legacy parity (and
     * matching {@link bumpMajor}) does NOT set
     * ``hasBumpedPatchInSession`` — a manual correction is an
     * out-of-band edit, not a user-meaningful bump.
     *
     * The modal owns the HTTP call + toast so future callers (e.g. a
     * mass-version-correction tool) can open it without re-wiring the
     * error path. The workspace just provides the local-cache
     * reconciliation callback.
     */
    protected editVersion(): void {
        const d = this.dataset();
        if (!d) return;
        this.overlay.openModal('version-edit', {
            datasetName: d.name,
            currentVersion: d.version,
            onSaved: (newVersion: string) => {
                const cur = this.dataset();
                if (cur) this.datasets.upsertLocal({ ...cur, version: newVersion });
            },
        });
    }
}

/**
 * Merge a MediaItem (metadata source of truth) with its optional
 * caption row into the legacy pair shape that the orphan-tree
 * grid + detail components expect:
 *
 *   { stem, media_file, media_type, caption_file, caption_content,
 *     masked_caption_content, metadata: { ...all-other-MediaItem-fields } }
 *
 * The pair-level keys (`stem`, `media_file`, `media_type`,
 * `caption_file`) are hoisted out of the MediaItem; everything else
 * (enabled, has_mask, dimensions, HPS, etc.) goes into `metadata`.
 */
function projectPair(item: MediaItem, caption?: CaptionRow): any {
    const {
        id, dataset_name, media_file, stem, media_type, caption_file,
        ...metadata
    } = item;
    return {
        stem: stem ?? stripExt(media_file),
        media_file,
        media_type: media_type ?? (item.is_video ? 'video' : 'image'),
        caption_file,
        caption_content: caption?.caption_content,
        masked_caption_content: caption?.masked_caption_content,
        metadata,
    };
}

function stripExt(path: string): string {
    const dot = path.lastIndexOf('.');
    return dot > 0 ? path.substring(0, dot) : path;
}
