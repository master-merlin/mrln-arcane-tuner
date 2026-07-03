import { Injectable, computed, inject, signal } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { EntityStore, OptimisticResult } from './entity-store';
import { DatasetService, PipelineBlock } from '../services/dataset';
import { WebSocketService } from '../services/websocket.service';
import { ToastService } from '../services/toast';

// ── Shell-overlay extension ─────────────────────────────────────────────
// Phase 2 of the frontend overhaul adds workspace + modal-stack state to
// this store so the shell's overlay layers can be driven from a single
// injectable. These concepts are unrelated to the per-image `Overlay`
// entity below — they happen to share the name "overlay" because both
// describe things layered above the page content.

export type WorkspaceMode = 'browse' | 'details' | 'edit';

export interface WorkspaceState {
    datasetId: string;
    mode: WorkspaceMode;
    imageIndex: number;
}

export type ModalKind =
    | 'mass-caption' | 'mass-mask' | 'mass-edit'
    | 'dataset-form' | 'rescan' | 'analyze' | 'cache'
    | 'project-dialog' | 'similar-images' | 'mask-preview' | 'crop-preview'
    | 'confirm' | 'input' | 'version-edit' | 'pair-order' | 'pair-health'
    | 'pair-role-chooser'
    | 'templates-library' | 'template-edit' | 'template-json' | 'job-config'
    | 'import-dataset' | 'export-options' | 'import-archive' | 'resume-job'
    | 'config-help' | 'model-source-config' | 'scene-detect' | 'cutlist-import';

export interface ModalEntry {
    kind: ModalKind;
    data?: unknown;
}

/**
 * Shape returned by ``POST /datasets/{name}/render-pipeline``. Mirrors
 * the dict returned by ``overlay_routes.render_pipeline`` server-side.
 */
export interface RenderPipelineResponse {
    status: string;
    file: string;
    overlay: string;
    dimensions: [number, number] | number[];
    hash: string;
}

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
 * (and a bulk per-dataset sync, à la ``DatasetSyncService.refreshDataset``,
 * impractical — components only ever care about the overlay for the
 * image currently open in the editor). The store's entry point is
 * therefore ``loadFor(datasetName, mediaFile)`` which fetches a single
 * overlay recipe and upserts it.
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
     *
     * Returns an {@link OptimisticResult} so callers can surface
     * response fields (e.g. ``dimensions``) in their own success toast
     * or use the raw HTTP error for per-item failure reporting in loops.
     */
    async renderPipeline(
        datasetName: string,
        mediaFile: string,
        blocks: PipelineBlock[],
        tileSize: number = 512,
        tilePad: number = 32,
        replaceRecipe: boolean = false,
    ): Promise<OptimisticResult<RenderPipelineResponse>> {
        const id = overlayKey(datasetName, mediaFile);
        const next: Overlay = {
            id,
            dataset_name: datasetName,
            media_file: mediaFile,
            operations: blocks
                .filter(b => b.enabled)
                .map(b => ({ type: b.type, enabled: b.enabled, params: b.params })),
        };
        return this.runOptimistic<RenderPipelineResponse>({
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
                ) as Promise<RenderPipelineResponse>,
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
     * Materialize the current overlay render into a control slot for an
     * edit dataset (pair production). Unlike {@link commitOverlay}, this is
     * NON-destructive: the original and its overlay row are left intact — we
     * only copy the rendered result into `control/` (slot 1) etc. The paired
     * target's `control_info` updates via the media_item entity.changed event
     * the backend emits, so no optimistic store mutation is needed here.
     */
    async saveOverlayToControl(
        datasetName: string,
        mediaFile: string,
        target: 'control' | 'control_2' | 'control_3',
    ): Promise<void> {
        try {
            await firstValueFrom(
                this.api.commitOverlay(datasetName, mediaFile, target),
            );
            this.toast.success(`Saved render to ${target}.`);
        } catch {
            this.toast.error(`Couldn't save render to ${target}.`);
        }
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

    // ── Shell-overlay layer (workspace + modal stack) ───────────────────
    // These signals drive the dataset workspace overlay and the modal
    // stack mounted by `app-shell`. Stack semantics (not single slot) so
    // a modal can open another modal (e.g. confirm-on-delete).

    readonly workspace = signal<WorkspaceState | null>(null);
    readonly modalStack = signal<ModalEntry[]>([]);
    readonly topModal = computed<ModalEntry | null>(() => this.modalStack().at(-1) ?? null);

    openWorkspace(datasetId: string, mode: WorkspaceMode = 'browse'): void {
        this.workspace.set({ datasetId, mode, imageIndex: 0 });
    }

    closeWorkspace(): void {
        this.workspace.set(null);
    }

    setWorkspaceMode(mode: WorkspaceMode): void {
        const w = this.workspace();
        if (w) this.workspace.set({ ...w, mode });
    }

    setWorkspaceImage(imageIndex: number): void {
        const w = this.workspace();
        if (w) this.workspace.set({ ...w, imageIndex });
    }

    openModal(kind: ModalKind, data?: unknown): void {
        this.modalStack.update(s => [...s, { kind, data }]);
    }

    closeModal(): void {
        this.modalStack.update(s => s.slice(0, -1));
    }

    closeAllModals(): void {
        this.modalStack.set([]);
    }

    /**
     * Patch the data payload of a modal at `depth` from the top
     * (0 = topmost). Used by modals that need to persist UI state
     * across child-modal pushes — when a child modal is closed, the
     * parent is re-instantiated (it lives behind `@if (last)`) and
     * re-reads its `data`. Mutating in place keeps the rest of the
     * stack stable.
     */
    patchModalData(patch: Record<string, unknown>, depth = 0): void {
        this.modalStack.update(s => {
            if (!s.length) return s;
            const idx = s.length - 1 - depth;
            if (idx < 0 || idx >= s.length) return s;
            const cur = s[idx];
            const next: ModalEntry = {
                ...cur,
                data: { ...(cur.data as Record<string, unknown> ?? {}), ...patch },
            };
            const out = s.slice();
            out[idx] = next;
            return out;
        });
    }
}
