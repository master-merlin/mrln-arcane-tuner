import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { EntityStore } from './entity-store';
import { DatasetService, PipelineBlock } from '../services/dataset';
import { WebSocketService } from '../services/websocket.service';
import { ToastService } from '../services/toast';

/**
 * Frontend view of a backend overlay — the result of running the
 * non-destructive editor pipeline on a single media file.
 *
 * Overlays are one-per-media-file (keyed by ``${dataset_name}/${media_file}``),
 * so the store-level ``id`` matches the composite used by the
 * ``media_item`` events. The backend doesn't expose a shared Pydantic
 * model for overlays — the shape here mirrors the payload emitted by
 * ``overlay_routes.render_pipeline`` on ``entity.changed:updated``.
 */
export interface Overlay {
    /** Composite key: `${dataset_name}/${media_file}` */
    id: string;
    /** Dataset this overlay belongs to. */
    dataset_name: string;
    /** Forward-slash relative path within the dataset. */
    media_file: string;
    /** Server-relative path of the rendered overlay PNG. */
    overlay_file?: string;
    /** [width, height] of the rendered overlay. */
    dimensions?: [number, number] | number[];
    /** SHA-256 of the rendered overlay PNG. */
    hash?: string;
    /** Pipeline operations that produced this overlay. */
    operations?: Array<{ type: string; enabled?: boolean; params?: Record<string, unknown> }>;
}

export function overlayKey(datasetName: string, mediaFile: string): string {
    return `${datasetName}/${mediaFile}`;
}

/**
 * Per-domain store for image-editor overlays.
 *
 * The backend has no "list all overlays for a dataset" endpoint, only a
 * per-file recipe endpoint. That makes a global ``loadAll`` impossible
 * (and a bulk ``loadForDataset`` impractical — components only ever
 * care about the overlay for the image currently open in the editor).
 * The store's entry point is therefore ``loadFor(datasetName, mediaFile)``
 * which fetches a single overlay recipe and upserts it.
 *
 * Components are not yet migrated to use the store — Task 16 of the
 * optimistic-ui-mutations plan covers that.
 */
@Injectable({ providedIn: 'root' })
export class OverlayStore extends EntityStore<Overlay> {
    protected entityName = 'overlay';
    private api = inject(DatasetService);

    constructor(ws: WebSocketService, toast: ToastService) {
        super(ws, toast);
    }

    public override async loadAll(): Promise<void> {
        // No bulk list endpoint exists for overlays. The WS-reconnect
        // re-hydrate hook is intentionally a no-op: previously-loaded
        // overlays stay in the store, and any server-side mutations
        // during the disconnect are replayed via entity.changed once
        // the client reconnects.
    }

    /**
     * Fetches a single overlay's recipe and upserts it into the store.
     * If the server returns 404 (no overlay for this media file), the
     * row is removed locally — matches the "no overlay" UI state.
     */
    async loadFor(datasetName: string, mediaFile: string): Promise<void> {
        const id = overlayKey(datasetName, mediaFile);
        try {
            const resp = await firstValueFrom(
                this.api.getOverlayRecipe(datasetName, mediaFile),
            );
            const recipe = (resp?.recipe ?? {}) as {
                overlay_file?: string;
                operations?: Overlay['operations'];
            };
            this.upsert({
                id,
                dataset_name: datasetName,
                media_file: mediaFile,
                overlay_file: recipe.overlay_file,
                operations: recipe.operations ?? [],
            });
        } catch {
            // 404 (or any other failure) — treat as "no overlay".
            this.remove(id);
        }
    }

    /**
     * Optimistically upserts an overlay row keyed by the composite, then
     * POSTs render-pipeline to the server. The render is heavy and runs
     * server-side; the optimistic row just records that we expect an
     * overlay to exist with the given operations. The authoritative
     * payload (with hash/dimensions) lands via ``entity.changed:updated``.
     *
     * Rolls back + toasts on HTTP failure.
     */
    async renderPipeline(
        datasetName: string,
        mediaFile: string,
        blocks: PipelineBlock[],
        tileSize: number = 512,
        tilePad: number = 32,
        replaceRecipe: boolean = false,
    ): Promise<void> {
        const id = overlayKey(datasetName, mediaFile);
        const next: Overlay = {
            id,
            dataset_name: datasetName,
            media_file: mediaFile,
            operations: blocks
                .filter(b => b.enabled)
                .map(b => ({ type: b.type, enabled: b.enabled, params: b.params })),
        };
        await this.runOptimistic({
            apply: m => new Map(m).set(id, next),
            request: () =>
                firstValueFrom(
                    this.api.renderPipeline(
                        datasetName,
                        mediaFile,
                        blocks,
                        tileSize,
                        tilePad,
                        replaceRecipe,
                    ),
                ),
            errorMessage: `Couldn't render overlay — reverted.`,
        });
    }

    /**
     * Optimistically removes the overlay row, then POSTs
     * /overlay/commit (flattens overlay into the original — server-side
     * the overlay file is removed). Rolls back + toasts on HTTP failure.
     */
    async commitOverlay(datasetName: string, mediaFile: string): Promise<void> {
        const id = overlayKey(datasetName, mediaFile);
        await this.runOptimistic({
            apply: m => { const n = new Map(m); n.delete(id); return n; },
            request: () => firstValueFrom(this.api.commitOverlay(datasetName, mediaFile)),
            errorMessage: `Couldn't commit overlay — restored.`,
        });
    }

    /**
     * Optimistically removes the overlay row, then DELETEs the overlay
     * (revert to original). Rolls back + toasts on HTTP failure.
     */
    async deleteOverlay(datasetName: string, mediaFile: string): Promise<void> {
        const id = overlayKey(datasetName, mediaFile);
        await this.runOptimistic({
            apply: m => { const n = new Map(m); n.delete(id); return n; },
            request: () => firstValueFrom(this.api.deleteOverlay(datasetName, mediaFile)),
            errorMessage: `Couldn't revert overlay — restored.`,
        });
    }
}
