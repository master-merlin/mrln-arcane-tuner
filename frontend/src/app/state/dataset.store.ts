import { Injectable, inject, signal } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { EntityStore } from './entity-store';
import { Dataset, DatasetService } from '../services/dataset';
import { WebSocketService } from '../services/websocket.service';
import { ToastService } from '../services/toast';

/**
 * Per-domain store for Dataset entities.
 *
 * Datasets are a flat list (no archive/history split, unlike jobs), so this
 * store exposes only `loadAll` + the three CRUD mutations. Component code
 * still has to migrate (Task 16); until then the store sits unused.
 *
 * Backend keys datasets by UUID `id`. Most HTTP endpoints are keyed by
 * `name` in their URL path, so mutations look up the current dataset by id
 * and pass `name` to the API.
 */
@Injectable({ providedIn: 'root' })
export class DatasetStore extends EntityStore<Dataset> {
    protected entityName = 'dataset';
    private api = inject(DatasetService);

    /**
     * Tri-state gate for the library grid. `true` from construction until the
     * first {@link loadAll} resolves (success OR error), and again for the
     * duration of any subsequent reload. Lets the datasets screen render
     * skeleton cards instead of flashing the empty-state message while the
     * initial list is still in flight (the "false-empty flash").
     */
    private _loading = signal(true);
    readonly loading = this._loading.asReadonly();

    constructor(ws: WebSocketService, toast: ToastService) {
        super(ws, toast);

        // Live WS bridge: training subprocesses signal `[CACHE_READY:...]`
        // and the backend broadcasts `dataset_cache_ready` with the names of
        // datasets whose cache became available. We flip `has_cache: true`
        // in-place so the CACHED KPI tile and the per-card "Cache" button
        // unlock without requiring a manual reload.
        //
        // Payload is `{ datasets: string[] }` of dataset NAMES (not IDs).
        // Long-lived store ⇒ no explicit teardown required (matches the
        // app-lifetime pattern used elsewhere; `takeUntilDestroyed` requires
        // an injection context not available from a base-class constructor).
        ws.on<{ datasets: string[] }>('dataset_cache_ready').subscribe(({ datasets }) => {
            this.patchHasCacheByName(datasets);
        });

        // Task completion (caption / mask / rescan) broadcasts the coarse
        // `dataset.invalidated`. We consume it from the RxJS stream — NOT the
        // single `entityChanged` signal, whose value coalesces under
        // `eventCoalescing` and gets clobbered by the flood of per-file
        // entity.changed events during a batch (so the dataset's own count
        // update was being dropped, leaving the Library card's C/M pills stale
        // until the next reload — worse while the Task Center is open, since a
        // slower CD widens the coalescing window). Re-fetch the affected
        // dataset's authoritative counts so the H/C/M pills reflect the new
        // status. Only refresh datasets we already hold; skip unknown names.
        ws.on<{ name: string }>('dataset.invalidated').subscribe(({ name }) => {
            if (!name || !this.entities().some(d => d.name === name)) return;
            this.api.getDataset(name).subscribe({
                next: row => this.upsert(row),
                error: () => undefined,
            });
        });
    }

    /**
     * Optimistically flips `has_cache: true` on each row whose `name` is in
     * `names`. No HTTP — the backend already persisted the cache; this is
     * a local reconciliation triggered by the `dataset_cache_ready` WS event.
     * Rows that aren't loaded yet are silently skipped (loaded later via
     * a normal `loadAll()` will reflect the server-side flag). Rows that
     * already have `has_cache: true` are NOT re-upserted (object identity
     * is preserved so downstream `computed` consumers don't churn).
     */
    patchHasCacheByName(names: string[]): void {
        if (names.length === 0) return;
        const wanted = new Set(names);
        for (const ds of this.entities()) {
            if (wanted.has(ds.name) && !ds.has_cache) {
                this.upsert({ ...ds, has_cache: true });
            }
        }
    }

    /**
     * Optimistically reflects a finished upload batch on the named row so the
     * Library card surfaces the new files immediately, rather than waiting for
     * the post-upload rescan (hashing, thumbnailing, HPS scoring can take many
     * seconds). Counts are classified by `kind` so caption files don't inflate
     * the image count — the bug this replaces bumped `multimedia_count` for
     * EVERY dropped file, so 10 images + 10 caption `.txt` files surfaced as
     * "20 images / 0 captioned" until a rescan landed.
     *
     * - `media`   → added to `multimedia_count`
     * - `caption` → added to `caption_count`
     * - both      → added to `file_count` (every uploaded file is a file)
     *
     * `previewCandidate` (a just-uploaded image's filename) seeds `preview_image`
     * ONLY when the row has none yet, so a freshly-populated card shows a
     * thumbnail instead of the empty-state glyph. The follow-up rescan's
     * `dataset.invalidated` broadcast re-fetches the row (see constructor),
     * resetting every field to backend-authoritative values — including stem
     * dedupe and the backend's own preview choice. No-op if the dataset isn't
     * loaded into the store, or if nothing was uploaded.
     */
    applyOptimisticUpload(
        name: string,
        counts: { media: number; caption: number },
        previewCandidate?: string,
    ): void {
        const fileDelta = counts.media + counts.caption;
        if (fileDelta === 0) return;
        for (const ds of this.entities()) {
            if (ds.name === name) {
                this.upsert({
                    ...ds,
                    multimedia_count: (ds.multimedia_count ?? 0) + counts.media,
                    caption_count: (ds.caption_count ?? 0) + counts.caption,
                    file_count: (ds.file_count ?? 0) + fileDelta,
                    preview_image: ds.preview_image || previewCandidate,
                });
                return;
            }
        }
    }

    /** Shared promise for the request currently in flight, if any. */
    private inFlight: Promise<void> | null = null;
    /** Bumped per request so a slow response cannot clobber a newer one. */
    private generation = 0;

    /**
     * Re-fetch the whole library. Concurrent callers share one request.
     *
     * The sidebar hydrates the store for its nav badge counts and the datasets
     * screen loads it for the grid, so mounting the library fired
     * `GET /datasets` TWICE — the app's largest response, fetched and parsed
     * twice for one page load. Same coalescing DatasetSyncService already does
     * per dataset for `refreshDataset`.
     *
     * `force: true` bypasses the join and always issues a fresh request. Use it
     * after a mutation: a request that STARTED before the mutation landed can
     * still be in flight, and joining it would hand back pre-mutation rows.
     */
    public override async loadAll(opts: { force?: boolean } = {}): Promise<void> {
        if (this.inFlight && !opts.force) return this.inFlight;

        const gen = ++this.generation;
        this._loading.set(true);
        const run = (async () => {
            try {
                const datasets = await firstValueFrom(this.api.listDatasets());
                // A `force` call may have overtaken us; letting this older
                // response land would put the pre-mutation list back.
                if (this.generation === gen) this.setAll(datasets);
            } finally {
                this._loading.set(false);
                if (this.generation === gen) this.inFlight = null;
            }
        })();
        this.inFlight = run;
        return run;
    }

    /**
     * Creates a dataset server-side. No optimistic insert — the backend
     * assigns the UUID, so we wait for the HTTP response to surface the
     * row. The `entity.changed:created` WebSocket event would do the same
     * if we ever moved to a fire-and-forget pattern.
     */
    async createDataset(
        name: string,
        description: string = '',
        classifier: string = '',
        extra: { trigger_word?: string; tags?: string[]; notes?: string; kind?: string } = {},
    ): Promise<Dataset | null> {
        try {
            const created = await firstValueFrom(
                this.api.createDataset(name, description, classifier, extra),
            );
            // Surface the row immediately for the caller; the WS broadcast
            // will upsert again (idempotent).
            this.upsert(created);
            return created;
        } catch {
            this.toast.error(`Couldn't create dataset.`);
            return null;
        }
    }

    async deleteDataset(id: string, deleteFiles: boolean = false): Promise<void> {
        const current = this.byId(id)();
        if (!current) return;
        const name = current.name;
        await this.runOptimistic({
            apply: m => { const n = new Map(m); n.delete(id); return n; },
            request: () => firstValueFrom(this.api.deleteDataset(name, deleteFiles)),
            errorMessage: `Couldn't delete dataset — restored.`,
        });
    }

    /**
     * Public counterpart to the protected ``upsert`` — used by external
     * mutation paths (e.g. workspace manual version bump) that already
     * have the full row in hand and want to reconcile the local cache
     * without an extra HTTP fetch. The WS broadcast will idempotently
     * upsert again.
     */
    upsertLocal(row: Dataset): void {
        this.upsert(row);
    }

    /**
     * Patches dataset metadata. Updates merge into the locally-cached
     * dataset, so callers can pass partial diffs (e.g. `{ description }`).
     *
     * Note: the HTTP API requires a full payload (name + description +
     * classifier). We fill missing fields from the current cached entity.
     */
    async updateDataset(id: string, updates: Partial<Dataset>): Promise<void> {
        const current = this.byId(id)();
        if (!current) return;
        const next: Dataset = { ...current, ...updates };
        await this.runOptimistic({
            apply: m => new Map(m).set(id, next),
            request: () => firstValueFrom(
                this.api.updateDataset(
                    current.name,
                    next.name,
                    next.description,
                    next.classifier ?? '',
                    {
                        trigger_word: next.trigger_word ?? '',
                        tags: next.tags ?? [],
                        notes: next.notes ?? '',
                        kind: next.kind,
                    },
                ),
            ),
            errorMessage: `Couldn't update dataset — reverted.`,
        });
    }
}
