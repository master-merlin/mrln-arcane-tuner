import { Injectable, Signal, computed, effect, inject, signal } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { EntityStore, OptimisticResult } from './entity-store';
import { DatasetService, DatasetPair } from '../services/dataset';
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
    /** True for audio media rows (C0). Duration/sample_rate/channels/has_lyrics
     *  flow through via the index signature below, same convention as the
     *  video-only fields (fps, duration_s, ...) that aren't re-declared here. */
    is_audio?: boolean;
    is_majority_ar?: boolean;
    /** Present once a mask exists; mirrors `PairMetadata.mask_info`. */
    mask_info?: { width?: number; height?: number; size_bytes?: number; [k: string]: unknown };
    /** Paired edit datasets — physical control slot rel-paths (slot order). */
    control_files?: string[];
    /** Logical role order; null/absent = default (root image is target). */
    role_order?: string[] | null;
    /** Resolved logical target/control rel-paths from the `/pairs` row. */
    effective_target?: string;
    effective_controls?: string[];
    [extra: string]: unknown;
}

export function mediaKey(datasetName: string, mediaFile: string): string {
    return `${datasetName}/${mediaFile}`;
}

/**
 * Per-domain store for MediaItem entities.
 *
 * Media items aren't loaded globally — they belong to a dataset — so the
 * default ``loadAll`` is a no-op. Callers sync a dataset's rows via
 * ``DatasetSyncService.refreshDataset(name)``, which fetches the pairs
 * endpoint and reconciles this store's slice via ``reconcileDataset``.
 * The workspace prefers ``upsertFromPair`` so it can do one `/pairs`
 * fetch and route the result through both the store and its local
 * caption-text cache.
 */
@Injectable({ providedIn: 'root' })
export class MediaItemStore extends EntityStore<MediaItem> {
    protected entityName = 'media_item';
    private api = inject(DatasetService);

    private _byDatasetCache = new Map<string, Signal<MediaItem[]>>();

    /**
     * Monotonic counter bumped whenever the original image bytes for some
     * media file MAY have changed under the same URL. Bake-in is the
     * concrete trigger today — backend replaces the original file in
     * place. Consumers (Details / Browse / filmstrip) append this to
     * `<img>` URLs as a cache-bust query param to force a re-fetch.
     *
     * Global rather than per-item because Bake is rare and user-initiated;
     * the cost of a few extra refreshes across the grid is negligible
     * compared to per-item bookkeeping. Revert also bumps (we can't
     * distinguish Bake from Revert by WS event alone) — the resulting
     * re-fetch is harmless since the bytes are unchanged.
     */
    readonly mediaRev = signal<number>(0);

    constructor(ws: WebSocketService, toast: ToastService) {
        super(ws, toast);

        // Bridge overlay WS events to media-item state. Save / Revert /
        // Bake-in mutate an overlay and the backend emits
        // entity.changed:overlay:created|updated|deleted, but does NOT
        // reliably emit a follow-up media_item event for the has_overlay
        // flag. Without this bridge, Browse / Details / Edit panes show
        // stale state until the user navigates away and back.
        //
        // Overlay id format is `${dataset_name}/${media_file}` — same
        // composite as MediaItem.id — so we can route the event directly.
        effect(() => {
            const msg = ws.entityChanged();
            if (!msg || msg.entity !== 'overlay') return;
            if (msg.op === 'bulk_deleted') return;
            // ANY single-overlay event (created/updated/deleted) means
            // bytes at a URL the grid/detail render changed:
            //   - `deleted` (bake-in / revert) rewrites the original at
            //     /media/<dataset>/<file>.
            //   - `created`/`updated` writes the overlay PNG at
            //     /api/datasets/<dataset>/overlay/<file>.
            // Both URLs carry ``?t=mediaRev`` so a bump cache-busts the
            // visible <img>. Previously only ``deleted`` bumped, so a
            // re-save left the grid serving stale cached bytes.
            this.mediaRev.update(r => r + 1);
            const current = this.byId(msg.id)();
            if (!current) return;
            const hasOverlay = msg.op !== 'deleted';
            if ((current.has_overlay ?? false) === hasOverlay) return;
            this.upsert({ ...current, has_overlay: hasOverlay });
        });
    }

    public override async loadAll(): Promise<void> {
        // Media items aren't globally listable; callers must use
        // DatasetSyncService.refreshDataset(name). The WS reconnect handler
        // still calls this (per EntityStore contract) — making it a no-op
        // means a reconnect doesn't wipe per-dataset state that components
        // have loaded.
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
     * Upserts a single pair-shaped row from the `/pairs` endpoint.
     * Exposed so the workspace can do one fetch, route the metadata
     * through the store, and still pull caption text into its local
     * cache — avoiding a second HTTP round trip.
     */
    upsertFromPair(datasetName: string, p: DatasetPair): void {
        const item = this.buildItem(datasetName, p);
        if (item) this.upsert(item);
    }

    /**
     * Reconcile the store's slice for `datasetName` to EXACTLY the rows the
     * server reports — upserting each pair AND evicting any existing row for
     * the dataset that the server no longer lists. This is the authoritative
     * "the file set changed on disk" path (harmonize rename, rescan, mass
     * caption), invoked via {@link DatasetSyncService.refreshDataset}; it
     * drops ghost rows whose underlying file was renamed away, so the grid
     * stops showing stale filenames + 404ing on their renamed-away captions.
     *
     * Other datasets' rows are untouched.
     */
    reconcileDataset(datasetName: string, pairs: DatasetPair[]): void {
        const rows: MediaItem[] = [];
        for (const p of pairs) {
            const item = this.buildItem(datasetName, p);
            if (item) rows.push(item);
        }
        this.replaceWhere(e => e.dataset_name === datasetName, rows);
    }

    /** Map a `/pairs` row to a {@link MediaItem}; null if it has no media file. */
    private buildItem(datasetName: string, p: DatasetPair): MediaItem | null {
        if (!p.media_file) return null;
        const meta = (p.metadata ?? {}) as Partial<MediaItem>;
        return {
            ...meta,
            id: mediaKey(datasetName, p.media_file),
            dataset_name: datasetName,
            media_file: p.media_file,
            stem: p.stem ?? meta.stem,
            media_type: p.media_type ?? meta.media_type,
            caption_file: p.caption_file ?? meta.caption_file,
            has_caption: !!(p.caption_file ?? meta.caption_file),
            // Pair-level control fields live on the /pairs ROW (not in the
            // metadata blob) — copy them explicitly or they'd be dropped.
            control_files: p.control_files ?? [],
            role_order: p.role_order ?? null,
            effective_target: p.effective_target ?? p.media_file,
            effective_controls: p.effective_controls ?? [],
        };
    }

    /**
     * Optimistically toggles the ``enabled`` flag for a single media item.
     * Rolls back + toasts on HTTP failure. If the item isn't in the store
     * (e.g. caller toggling before the dataset has been synced via
     * DatasetSyncService.refreshDataset), falls through to a plain HTTP
     * call and lets the server-emitted entity.changed event reconcile.
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
     * Bake a mask onto an image: backend composites `<image>` with its
     * `<mask>` at the given opacity, writes the result to
     * `masked/<stem>.jpg`, and flips `has_masked: true` server-side.
     *
     * Not optimistic — the bake is a real image-compositing step that
     * takes meaningful time, and flipping `has_masked: true` before the
     * file is on disk would let the masked-view URL race the bake and
     * 404. We wait for HTTP success, then patch the local row.
     */
    async applyMask(
        datasetName: string,
        mediaFile: string,
        opacity: number,
    ): Promise<OptimisticResult<unknown>> {
        try {
            const value = await firstValueFrom(
                this.api.applyMask(datasetName, mediaFile, opacity),
            );
            const key = mediaKey(datasetName, mediaFile);
            const current = this.byId(key)();
            if (current) {
                this.upsert({ ...current, has_masked: true });
            }
            return { ok: true, value };
        } catch (error) {
            this.toast.error(`Couldn't bake mask.`);
            return { ok: false, error };
        }
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

    /**
     * Stamp caption metadata after a caption was saved out-of-band (e.g. by a
     * mass-caption run that writes the text into CaptionCacheStore). No HTTP —
     * the save already happened; this only reconciles the store flags so the
     * grid shows the filename + captioned state live. No-op if the item isn't
     * loaded into the store.
     */
    stampCaption(datasetName: string, mediaFile: string, captionFile: string): void {
        const key = mediaKey(datasetName, mediaFile);
        const current = this.byId(key)();
        if (!current) return;
        if (current.has_caption === true && current.caption_file === captionFile) return;
        this.upsert({ ...current, caption_file: captionFile, has_caption: true });
    }

    /**
     * Flag that a mask now exists for an image (mass-mask Generate). No HTTP —
     * optimistic; the authoritative reconcile is a follow-up
     * DatasetSyncService.refreshDataset. No-op if the item isn't loaded or
     * already flagged.
     */
    markMaskGenerated(datasetName: string, mediaFile: string): void {
        const key = mediaKey(datasetName, mediaFile);
        const current = this.byId(key)();
        if (!current || current.has_mask === true) return;
        this.upsert({ ...current, has_mask: true });
    }

    /**
     * Flag that a masked-target caption now exists for an image (mass-caption
     * masked target / mass-mask Caption tab). No HTTP — optimistic; the
     * authoritative reconcile is a follow-up DatasetSyncService.refreshDataset.
     * No-op if the item isn't loaded or already flagged.
     */
    markMaskedCaptioned(datasetName: string, mediaFile: string): void {
        const key = mediaKey(datasetName, mediaFile);
        const current = this.byId(key)();
        if (!current || current.has_masked_caption === true) return;
        this.upsert({ ...current, has_masked_caption: true });
    }

    /**
     * Force a cache-bust across grid/detail/filmstrip `<img>` URLs after ops
     * that rewrite image bytes under the same URL (mask apply, overlay render).
     * Consumers append `mediaRev()` to their URLs.
     */
    bumpMedia(): void {
        this.mediaRev.update(r => r + 1);
    }
}
