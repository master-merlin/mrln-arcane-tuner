import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { EntityStore, OptimisticResult } from './entity-store';
import { DatasetService } from '../services/dataset';
import { WebSocketService } from '../services/websocket.service';
import { ToastService } from '../services/toast';

/**
 * Frontend view of a backend media item. The backend keeps these as a
 * dict under ``Dataset.media_metadata`` keyed by relative path — there's
 * no dedicated MediaItem Pydantic model — so this interface mirrors the
 * fields the persist/emit codepath actually surfaces.
 *
 * The store-level ``id`` is the composite ``{dataset_name}/{media_file}``,
 * which the backend also broadcasts via ``entity.changed``. The composite
 * is required because media items live within datasets (the bare
 * ``media_file`` isn't globally unique across datasets).
 */
export interface MediaItem {
    /** Composite key: `${dataset_name}/${media_file}` */
    id: string;
    /** Dataset this item belongs to. */
    dataset_name: string;
    /** Forward-slash relative path within the dataset (e.g. `subdir/img.png`). */
    media_file: string;
    enabled?: boolean;
    has_caption?: boolean;
    has_mask?: boolean;
    has_masked?: boolean;
    has_masked_caption?: boolean;
    width?: number;
    height?: number;
    aspect_ratio?: number;
    orientation?: string;
    size_bytes?: number;
    quality_score?: number | null;
    is_video?: boolean;
    [extra: string]: unknown;
}

/** Shape returned by ``DatasetService.getDatasetPairs`` (subset we use). */
interface DatasetPair {
    media_file?: string;
    metadata?: Partial<MediaItem> | null;
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
 * composite id.
 *
 * Components are not yet migrated to use the store — Task 16 of the
 * optimistic-ui-mutations plan covers that.
 */
@Injectable({ providedIn: 'root' })
export class MediaItemStore extends EntityStore<MediaItem> {
    protected entityName = 'media_item';
    private api = inject(DatasetService);

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
            if (!p.media_file) continue;
            const meta = p.metadata ?? {};
            const item: MediaItem = {
                ...meta,
                id: mediaKey(datasetName, p.media_file),
                dataset_name: datasetName,
                media_file: p.media_file,
            };
            this.upsert(item);
        }
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
}
