import { Injectable, Signal, computed, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { EntityStore, OptimisticResult } from './entity-store';
import { DatasetService } from '../services/dataset';
import { WebSocketService } from '../services/websocket.service';
import { ToastService } from '../services/toast';

/**
 * Frontend view of a backend media item. The backend keeps these as a
 * dict under ``Dataset.media_metadata`` keyed by relative path — there's
 * no dedicated MediaItem Pydantic model — so this interface mirrors the
 * fields the persist/emit codepath actually surfaces, PLUS the
 * pair-level fields (`media_type`, `stem`, `caption_file`) that live
 * one level up in the `/pairs` endpoint's response.
 *
 * The store-level ``id`` is the composite ``{dataset_name}/{media_file}``,
 * which the backend also broadcasts via ``entity.changed``. The composite
 * is required because media items live within datasets (the bare
 * ``media_file`` isn't globally unique across datasets).
 *
 * NOTE: caption TEXT (`caption_content`, `masked_caption_content`) is
 * intentionally not on MediaItem — those fields are heavy, are not
 * broadcast over WS, and live in a thin per-workspace local cache
 * instead. The store carries only `caption_file` / `has_caption` so
 * metadata-driven UI (e.g. "(New File)" vs filename, captioned-count
 * filter) can react via the store.
 */
export interface MediaItem {
    /** Composite key: `${dataset_name}/${media_file}` */
    id: string;
    /** Dataset this item belongs to. */
    dataset_name: string;
    /** Forward-slash relative path within the dataset (e.g. `subdir/img.png`). */
    media_file: string;
    /** Filename without extension; used as the @for tracking key. */
    stem?: string;
    /** Coarse type: `'image'` or `'video'` (or other backend types). */
    media_type?: string;
    /** Caption sidecar path relative to the dataset, when one exists. */
    caption_file?: string;
    enabled?: boolean;
    has_caption?: boolean;
    has_mask?: boolean;
    has_masked?: boolean;
    has_masked_caption?: boolean;
    has_overlay?: boolean;
    width?: number;
    height?: number;
    target_width?: number;
    target_height?: number;
    aspect_ratio?: number;
    orientation?: string;
    size_bytes?: number;
    quality_score?: number | null;
    is_video?: boolean;
    is_majority_ar?: boolean;
    [extra: string]: unknown;
}

/** Shape returned by ``DatasetService.getDatasetPairs`` (subset we use). */
interface DatasetPair {
    media_file?: string;
    media_type?: string;
    stem?: string;
    caption_file?: string;
    metadata?: Partial<MediaItem> | null;
    [extra: string]: unknown;
}

export function mediaKey(datasetName: string, mediaFile: string): string {
    return `${datasetName}/${mediaFile}`;
}

/**
 * Per-domain store for MediaItem entities.
 *
 * Media items aren't loaded globally — they belong to a dataset — so the
 * default ``loadAll`` is a no-op. Callers use ``loadForDataset(name)``
 * which fetches the pairs endpoint and upserts each row keyed by the
 * composite id. The workspace prefers ``upsertFromPair`` so it can do
 * one `/pairs` fetch and route the result through both the store and
 * its local caption-text cache.
 */
@Injectable({ providedIn: 'root' })
export class MediaItemStore extends EntityStore<MediaItem> {
    protected entityName = 'media_item';
    private api = inject(DatasetService);

    private _byDatasetCache = new Map<string, Signal<MediaItem[]>>();

    constructor(ws: WebSocketService, toast: ToastService) {
        super(ws, toast);
    }

    public override async loadAll(): Promise<void> {
        // Media items aren't globally listable; callers must use
        // loadForDataset(name). The WS reconnect handler still calls this
        // (per EntityStore contract) — making it a no-op means a reconnect
        // doesn't wipe per-dataset state that components have loaded.
    }

    /**
     * Returns a memoized computed signal yielding every MediaItem
     * belonging to `datasetName`, in insertion order (which mirrors the
     * `/pairs` response order). Same Signal across calls so template
     * bindings don't allocate per change-detection cycle.
     */
    readonly byDataset = (datasetName: string): Signal<MediaItem[]> => {
        const cached = this._byDatasetCache.get(datasetName);
        if (cached) return cached;
        const c = computed(() =>
            this.entities().filter(e => e.dataset_name === datasetName),
        );
        this._byDatasetCache.set(datasetName, c);
        return c;
    };

    /**
     * Fetches all media items for a dataset and upserts them into the
     * store. Items belonging to OTHER datasets are preserved, so multiple
     * datasets can coexist in the store concurrently.
     *
     * Note: items previously held for THIS dataset that the server no
     * longer reports are NOT removed (a stale-on-reload caveat). The
     * server emits ``entity.changed:deleted`` for actual deletions, so
     * in practice the only stale rows are ones that vanished while the
     * client was disconnected — acceptable for the MVP.
     */
    async loadForDataset(datasetName: string): Promise<void> {
        const pairs = (await firstValueFrom(
            this.api.getDatasetPairs(datasetName),
        )) as DatasetPair[];
        for (const p of pairs) {
            this.upsertFromPair(datasetName, p);
        }
    }

    /**
     * Upserts a single pair-shaped row from the `/pairs` endpoint.
     * Exposed so the workspace can do one fetch, route the metadata
     * through the store, and still pull caption text into its local
     * cache — avoiding a second HTTP round trip.
     */
    upsertFromPair(datasetName: string, p: DatasetPair): void {
        if (!p.media_file) return;
        const meta = (p.metadata ?? {}) as Partial<MediaItem>;
        const item: MediaItem = {
            ...meta,
            id: mediaKey(datasetName, p.media_file),
            dataset_name: datasetName,
            media_file: p.media_file,
            stem: p.stem ?? meta.stem,
            media_type: p.media_type ?? meta.media_type,
            caption_file: p.caption_file ?? meta.caption_file,
            has_caption: !!(p.caption_file ?? meta.caption_file),
        };
        this.upsert(item);
    }

    /**
     * Optimistically toggles the ``enabled`` flag for a single media item.
     * Rolls back + toasts on HTTP failure. If the item isn't in the store
     * (e.g. caller toggling before loadForDataset), falls through to a
     * plain HTTP call and lets the server-emitted entity.changed event
     * reconcile.
     *
     * Returns an {@link OptimisticResult} so callers maintaining their own
     * optimistic projections (e.g. `dataset-viewer.ts` keeps a richer
     * `pairs` snapshot) can authoritatively roll back on failure
     * regardless of whether the row was pre-seeded in the store.
     */
    async toggleEnabled(
        datasetName: string,
        mediaFile: string,
        enabled: boolean,
    ): Promise<OptimisticResult<unknown>> {
        const key = mediaKey(datasetName, mediaFile);
        const current = this.byId(key)();
        if (!current) {
            try {
                const value = await firstValueFrom(
                    this.api.toggleImageEnabled(datasetName, mediaFile, enabled),
                );
                return { ok: true, value };
            } catch (error) {
                this.toast.error(`Couldn't update — reverted.`);
                return { ok: false, error };
            }
        }
        return this.runOptimistic({
            apply: m => new Map(m).set(key, { ...current, enabled }),
            request: () =>
                firstValueFrom(
                    this.api.toggleImageEnabled(datasetName, mediaFile, enabled),
                ),
            errorMessage: `Couldn't update — reverted.`,
        });
    }

    /**
     * Optimistically re-includes every excluded item in a dataset.
     * Single API call covers the bulk; the local apply flips
     * `enabled: true` on every store row scoped to this dataset.
     */
    async enableAll(datasetName: string): Promise<OptimisticResult<unknown>> {
        return this.runOptimistic({
            apply: m => {
                const next = new Map(m);
                for (const [id, item] of next) {
                    if (item.dataset_name === datasetName && item.enabled === false) {
                        next.set(id, { ...item, enabled: true });
                    }
                }
                return next;
            },
            request: () => firstValueFrom(this.api.enableAllImages(datasetName)),
            errorMessage: `Couldn't enable all — reverted.`,
        });
    }

    /**
     * Optimistically removes a pair from the store. Caller is
     * responsible for cleaning up companion local state (e.g. caption
     * text caches keyed by media_file).
     */
    async deletePair(
        datasetName: string,
        mediaFile: string,
    ): Promise<OptimisticResult<unknown>> {
        const key = mediaKey(datasetName, mediaFile);
        return this.runOptimistic({
            apply: m => {
                const next = new Map(m);
                next.delete(key);
                return next;
            },
            request: () =>
                firstValueFrom(this.api.deletePair(datasetName, mediaFile)),
            errorMessage: `Couldn't delete entry — reverted.`,
        });
    }

    /**
     * Optimistically deletes the mask for a pair (flips `has_mask: false`).
     * No bulk "set true" path — mask creation is async (Meta SAM 3 etc.)
     * and the workspace pulls the resulting metadata via a re-fetch.
     */
    async deleteMask(
        datasetName: string,
        mediaFile: string,
    ): Promise<OptimisticResult<unknown>> {
        const key = mediaKey(datasetName, mediaFile);
        const current = this.byId(key)();
        if (!current) {
            try {
                const value = await firstValueFrom(
                    this.api.deleteMask(datasetName, mediaFile),
                );
                return { ok: true, value };
            } catch (error) {
                this.toast.error(`Couldn't delete mask — reverted.`);
                return { ok: false, error };
            }
        }
        return this.runOptimistic({
            apply: m => new Map(m).set(key, { ...current, has_mask: false }),
            request: () =>
                firstValueFrom(this.api.deleteMask(datasetName, mediaFile)),
            errorMessage: `Couldn't delete mask — reverted.`,
        });
    }

    /**
     * Persists caption text AND optimistically stamps the metadata
     * (`caption_file`, `has_caption: true`). The text itself is NOT
     * stored here — it lives in the caller's caption cache because
     * it's heavy and not broadcast over WS.
     *
     * The caller still needs to maintain its own rollback for the text
     * field (snapshot pre-call, restore on `!result.ok`).
     */
    async saveCaption(
        datasetName: string,
        mediaFile: string,
        captionFile: string,
        content: string,
    ): Promise<OptimisticResult<unknown>> {
        const key = mediaKey(datasetName, mediaFile);
        const current = this.byId(key)();
        if (!current) {
            try {
                const value = await firstValueFrom(
                    this.api.saveCaption(datasetName, captionFile, content),
                );
                return { ok: true, value };
            } catch (error) {
                this.toast.error(`Couldn't save caption — reverted.`);
                return { ok: false, error };
            }
        }
        return this.runOptimistic({
            apply: m =>
                new Map(m).set(key, {
                    ...current,
                    caption_file: captionFile,
                    has_caption: true,
                }),
            request: () =>
                firstValueFrom(
                    this.api.saveCaption(datasetName, captionFile, content),
                ),
            errorMessage: `Couldn't save caption — reverted.`,
        });
    }
}
