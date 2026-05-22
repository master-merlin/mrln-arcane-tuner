import { computed, effect, signal, Signal } from '@angular/core';
import { WebSocketService } from '../services/websocket.service';
import { ToastService } from '../services/toast';
import { EntityChangedMessage, isBulkDeletedPayload } from './entity-events';

export interface HasId { id: string; }

/**
 * Base class for per-domain stores.
 *
 * - `_entities` is the canonical Map<id, T>.
 * - User-initiated mutations go through `runOptimistic`: snapshot -> apply ->
 *   HTTP. On HTTP failure, snapshot restores and a toast surfaces.
 * - Server-pushed `entity.changed` events flow through `applyServerEvent`,
 *   which is idempotent (safe before, during, or after an optimistic apply).
 * - On WS reconnect, `loadAll()` re-hydrates from the authoritative source.
 */
export abstract class EntityStore<T extends HasId> {
    protected abstract entityName: string;
    protected _entities = signal<Map<string, T>>(new Map());

    readonly entities: Signal<T[]> = computed(() => Array.from(this._entities().values()));
    readonly byId = (id: string): Signal<T | undefined> =>
        computed(() => this._entities().get(id));

    constructor(
        protected ws: WebSocketService,
        protected toast: ToastService,
    ) {
        effect(() => {
            const msg = this.ws.entityChanged();
            if (msg && msg.entity === this.entityName) {
                this.applyServerEvent(msg);
            }
        });
        effect(() => {
            const n = this.ws.reconnected();
            if (n > 0) {
                void this.loadAll();
            }
        });
    }

    protected async runOptimistic<R>(args: {
        apply: (m: Map<string, T>) => Map<string, T>;
        request: () => Promise<R>;
        errorMessage: string;
    }): Promise<R | null> {
        const snapshot = this._entities();
        this._entities.set(args.apply(snapshot));
        try {
            return await args.request();
        } catch (err) {
            this._entities.set(snapshot);
            this.toast.error(args.errorMessage);
            return null;
        }
    }

    private applyServerEvent(msg: EntityChangedMessage): void {
        this._entities.update(m => {
            const next = new Map(m);
            if (msg.op === 'deleted') {
                next.delete(msg.id);
            } else if (msg.op === 'bulk_deleted') {
                if (isBulkDeletedPayload(msg.payload)) {
                    for (const id of msg.payload.ids) next.delete(id);
                }
            } else {
                const row = msg.payload as T | null;
                if (row && row.id) next.set(row.id, row);
            }
            return next;
        });
    }

    protected abstract loadAll(): Promise<void>;
}
