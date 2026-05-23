import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { EntityStore } from './entity-store';
import { ModelService, ModelSourceOverride } from '../services/model.service';
import { WebSocketService } from '../services/websocket.service';
import { ToastService } from '../services/toast';

/**
 * Per-definition model source override, wrapped with an `id` field so it
 * fits the {@link EntityStore} `HasId` contract.
 *
 * The backend persists these in `settings.json` under
 * `models.overrides.<definition_id>`. The store's `id` is exactly that
 * `definition_id`, matching what the backend broadcasts on
 * `entity.changed:registry_model`.
 */
export interface KeyedOverride extends ModelSourceOverride {
    /** Equal to the model definition id. */
    id: string;
}

/**
 * Per-domain store for model source overrides.
 *
 * The registry has no list endpoint — overrides are fetched one-at-a-time
 * via `loadFor(definitionId)`. As a result `loadAll` is a no-op, and the
 * store only holds rows the user has explicitly looked at (or that the
 * server has broadcast about). This is enough to power optimistic UI in
 * the `model-source-config` component, which is the only consumer.
 *
 * The broader model catalog (`model-registry.yaml`, `ModelDefinition`) is
 * a separate, read-mostly system not covered here.
 *
 * Component migration is deferred to Task 16 of the optimistic-ui
 * mutations plan.
 */
@Injectable({ providedIn: 'root' })
export class RegistryStore extends EntityStore<KeyedOverride> {
    protected entityName = 'registry_model';
    private api = inject(ModelService);

    constructor(ws: WebSocketService, toast: ToastService) {
        super(ws, toast);
    }

    public override async loadAll(): Promise<void> {
        // No bulk list endpoint exists for per-definition overrides, so
        // the WS-reconnect re-hydrate hook is intentionally a no-op:
        // re-fetching every previously-loaded override would race with
        // the component's own `loadFor` calls and offer no benefit (the
        // backend re-emits `entity.changed` for anything that mutated
        // while disconnected once the client reconnects).
    }

    /**
     * Fetches a single override from the server and upserts it. Use
     * before reading via `byId` to guarantee the store reflects the
     * persisted state (e.g., on component init).
     */
    async loadFor(definitionId: string): Promise<void> {
        const override = await firstValueFrom(
            this.api.getModelSource(definitionId),
        );
        this.upsert({ ...override, id: definitionId });
    }

    /**
     * Optimistically upserts an override and PUTs it to the server.
     * Rolls back + toasts on HTTP failure.
     */
    async setOverride(
        definitionId: string,
        override: ModelSourceOverride,
    ): Promise<void> {
        const next: KeyedOverride = { ...override, id: definitionId };
        await this.runOptimistic({
            apply: m => new Map(m).set(definitionId, next),
            request: () =>
                firstValueFrom(this.api.setModelSource(definitionId, override)),
            errorMessage: `Couldn't save model source — reverted.`,
        });
    }

    /**
     * Optimistically removes an override and DELETEs it on the server.
     * Rolls back + toasts on HTTP failure. Safe to call when the
     * override isn't currently in the store (still fires the HTTP).
     */
    async clearOverride(definitionId: string): Promise<void> {
        await this.runOptimistic({
            apply: m => { const n = new Map(m); n.delete(definitionId); return n; },
            request: () =>
                firstValueFrom(this.api.deleteModelSource(definitionId)),
            errorMessage: `Couldn't clear model source — restored.`,
        });
    }
}
