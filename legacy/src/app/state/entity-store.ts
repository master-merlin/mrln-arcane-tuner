import { computed, effect, signal, Signal } from '@angular/core';
import { WebSocketService } from '../services/websocket.service';
import { ToastService } from '../services/toast';
import { EntityChangedMessage, isBulkDeletedPayload } from './entity-events';

export interface HasId { id: string; }

export type OptimisticResult<R> =
    | { ok: true; value: R }
    | { ok: false; error: unknown };

/**
 * Base class for per-domain stores.
 *
 * - `_entities` is the canonical Map<id, T> and is PRIVATE. Subclasses mutate
 *   it only via the protected primitives `setAll` / `upsert` / `remove` /
 *   `bulkRemove`, or via `runOptimistic` for user-initiated mutations.
 * - User-initiated mutations go through `runOptimistic`: snapshot -> apply ->
 *   HTTP. On HTTP failure, snapshot restores, a toast surfaces, and the
 *   discriminated result preserves the original error for the caller.
 * - Server-pushed `entity.changed` events flow through `applyServerEvent`,
 *   which is idempotent (safe before, during, or after an optimistic apply).
 * - On WS reconnect, `loadAll()` re-hydrates from the authoritative source.
 */
export abstract class EntityStore<T extends HasId> {
    protected abstract entityName: string;
    private _entities = signal<Map<string, T>>(new Map());
    private _byIdCache = new Map<string, Signal<T | undefined>>();

    readonly entities: Signal<T[]> = computed(() => Array.from(this._entities().values()));

    /**
     * Returns a memoized computed signal for the entity with the given id.
     * The same Signal instance is returned for the same id across calls, so
     * template bindings like `[input]="store.byId(id)()"` do not allocate a
     * new computed per change-detection cycle.
     */
    readonly byId = (id: string): Signal<T | undefined> => {
        const cached = this._byIdCache.get(id);
        if (cached) return cached;
        const c = computed(() => this._entities().get(id));
        this._byIdCache.set(id, c);
        return c;
    };

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

    // --- Protected mutation primitives (for subclasses) ---

    protected setAll(rows: T[]): void {
        this._entities.set(new Map(rows.map(r => [r.id, r])));
    }

    protected upsert(row: T): void {
        this._entities.update(m => {
            const n = new Map(m);
            n.set(row.id, row);
            return n;
        });
    }

    protected remove(id: string): void {
        this._entities.update(m => {
            const n = new Map(m);
            n.delete(id);
            return n;
        });
    }

    protected bulkRemove(ids: string[]): void {
        this._entities.update(m => {
            const n = new Map(m);
            for (const id of ids) n.delete(id);
            return n;
        });
    }

    // --- Optimistic mutation runner ---

    protected async runOptimistic<R>(args: {
        apply: (m: Map<string, T>) => Map<string, T>;
        request: () => Promise<R>;
        errorMessage: string;
    }): Promise<OptimisticResult<R>> {
        const snapshot = this._entities();
        this._entities.set(args.apply(snapshot));
        try {
            const value = await args.request();
            return { ok: true, value };
        } catch (error) {
            this._entities.set(snapshot);
            this.toast.error(args.errorMessage);
            return { ok: false, error };
        }
    }

    // --- Server-event dispatcher (private; subclasses use primitives) ---

    private applyServerEvent(msg: EntityChangedMessage): void {
        if (msg.op === 'bulk_deleted') {
            if (isBulkDeletedPayload(msg.payload)) {
                this.bulkRemove(msg.payload.ids);
            }
            return;
        }
        if (msg.op === 'deleted') {
            this.remove(msg.id);
            return;
        }
        // created or updated
        const row = msg.payload as T | null;
        if (row && row.id) this.upsert(row);
    }

    protected abstract loadAll(): Promise<void>;
}
