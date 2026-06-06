import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { MediaItemStore } from './media-item.store';
import { CaptionCacheStore, CaptionRow } from './caption-cache.store';
import { DatasetService } from '../services/dataset';
import { WebSocketService } from '../services/websocket.service';

/**
 * The single "a dataset's files changed — re-sync it from disk" entry point.
 *
 * The frontend caches a dataset's file list across two app-wide singletons:
 * {@link MediaItemStore} (filenames + metadata) and {@link CaptionCacheStore}
 * (heavy caption text). They live for the whole browser session and are NOT
 * reset by navigation, so any op that renames/adds/removes files on disk would
 * otherwise leave ghost rows (stale filenames, 404ing captions) until a hard
 * reload. {@link refreshDataset} re-fetches `/pairs` (the disk-truth endpoint)
 * and reconciles both stores to exactly that list — replace, not merge.
 *
 * Every file-changing op (harmonize, mass-caption, rescan, mask bake) funnels
 * through here. It also listens for the backend's coarse `dataset.invalidated`
 * broadcast so OTHER tabs/windows self-correct, and re-reconciles loaded
 * datasets on WS reconnect (catching up on events missed while disconnected).
 */
@Injectable({ providedIn: 'root' })
export class DatasetSyncService {
    private mediaItems = inject(MediaItemStore);
    private captions = inject(CaptionCacheStore);
    private api = inject(DatasetService);
    private ws = inject(WebSocketService);

    /** Per-dataset in-flight refresh, so a burst of invalidations coalesces. */
    private inFlight = new Map<string, Promise<void>>();

    constructor() {
        // Cross-tab / multi-window: backend broadcasts `dataset.invalidated`
        // after any structural mutation. Only re-sync datasets this client is
        // actually holding — an untracked dataset has nothing stale to fix.
        this.ws.on<{ name: string }>('dataset.invalidated').subscribe(({ name }) => {
            if (name && this.isLoaded(name)) void this.refreshDataset(name);
        });

        // On reconnect we may have missed `dataset.invalidated` events while
        // the socket was down. MediaItemStore.loadAll() is a deliberate no-op,
        // so reconnect does NOT otherwise re-sync media items — re-reconcile
        // every loaded dataset here.
        this.ws.reconnected$.subscribe(() => {
            for (const name of this.loadedDatasetNames()) void this.refreshDataset(name);
        });
    }

    /**
     * Re-fetch `/pairs` for `name` and reconcile both stores to it (evicting
     * ghosts). Idempotent; concurrent calls for the same dataset share one
     * fetch. Errors are swallowed (a transient refresh failure shouldn't crash
     * the UI — the next op or reconnect retries).
     */
    refreshDataset(name: string): Promise<void> {
        const existing = this.inFlight.get(name);
        if (existing) return existing;
        const run = this.doRefresh(name).finally(() => this.inFlight.delete(name));
        this.inFlight.set(name, run);
        return run;
    }

    private async doRefresh(name: string): Promise<void> {
        try {
            const pairs = await firstValueFrom(this.api.getDatasetPairs(name));
            const captions = new Map<string, CaptionRow>();
            for (const p of pairs ?? []) {
                if (!p?.media_file) continue;
                captions.set(p.media_file, {
                    caption_content: p.caption_content,
                    masked_caption_content: p.masked_caption_content ?? undefined,
                });
            }
            // Reconcile media items (replace + evict ghosts) and replace the
            // caption map wholesale (`seed` already overwrites).
            this.mediaItems.reconcileDataset(name, pairs ?? []);
            this.captions.seed(name, captions);
        } catch {
            // Leave existing state in place; a later op/reconnect will retry.
        }
    }

    /** True when this client currently holds any media rows for the dataset. */
    private isLoaded(name: string): boolean {
        return this.mediaItems.byDataset(name)().length > 0;
    }

    /** Distinct dataset names currently present in the media store. */
    private loadedDatasetNames(): string[] {
        return [...new Set(this.mediaItems.entities().map(e => e.dataset_name))];
    }
}
