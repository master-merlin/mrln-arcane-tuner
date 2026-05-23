import { Injectable, inject } from '@angular/core';
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

    constructor(ws: WebSocketService, toast: ToastService) {
        super(ws, toast);
    }

    public override async loadAll(): Promise<void> {
        const datasets = await firstValueFrom(this.api.listDatasets());
        this.setAll(datasets);
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
    ): Promise<Dataset | null> {
        try {
            const created = await firstValueFrom(
                this.api.createDataset(name, description, classifier),
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
                ),
            ),
            errorMessage: `Couldn't update dataset — reverted.`,
        });
    }
}
