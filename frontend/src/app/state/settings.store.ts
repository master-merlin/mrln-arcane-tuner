import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { EntityStore } from './entity-store';
import { SettingsService } from '../services/settings.service';
import { WebSocketService } from '../services/websocket.service';
import { ToastService } from '../services/toast';

/**
 * One module's settings, wrapped with an `id` field so it fits the
 * {@link EntityStore} `HasId` contract.
 *
 * The backend persists settings under `settings.json` as a flat
 * `Record<module, Record<string, unknown>>`. The store's `id` is the
 * module name, matching what the backend broadcasts on
 * `entity.changed:settings`.
 */
export interface ModuleSettings {
    /** Equal to the module name (e.g. `application`, `models`). */
    id: string;
    module: string;
    settings: Record<string, unknown>;
}

/**
 * Per-domain store for module-keyed application settings.
 *
 * Like {@link RegistryStore}, there is no list endpoint — settings are
 * fetched one module at a time via `loadModule(module)`. As a result
 * `loadAll` is a no-op and the store only holds modules the user has
 * explicitly looked at (or that the server has broadcast about).
 *
 * Deletion is intentionally absent: settings are key-overwrite, not
 * deletable. The backend never emits `entity.changed:deleted` for
 * `entity='settings'`, so the EntityStore base class's delete handling
 * is dormant here.
 *
 * Component migration is deferred to Task 16 of the optimistic-ui
 * mutations plan.
 */
@Injectable({ providedIn: 'root' })
export class SettingsStore extends EntityStore<ModuleSettings> {
    protected entityName = 'settings';
    private api = inject(SettingsService);

    constructor(ws: WebSocketService, toast: ToastService) {
        super(ws, toast);
    }

    public override async loadAll(): Promise<void> {
        // No bulk list endpoint exists. The WS-reconnect re-hydrate hook
        // is intentionally a no-op for the same reasons as RegistryStore:
        // re-fetching every previously-loaded module would race with
        // the component's own `loadModule` calls and offer no benefit
        // (the backend re-emits `entity.changed` for anything that
        // mutated while disconnected once the client reconnects).
    }

    /**
     * Fetches a single module's settings and upserts. Call before
     * reading via `byId` to guarantee the store reflects the persisted
     * state (e.g., on component init).
     */
    async loadModule(module: string): Promise<void> {
        const settings = await firstValueFrom(this.api.getModule(module));
        this.upsert({ id: module, module, settings });
    }

    /**
     * Optimistically upserts a module's settings and PUTs to the
     * server. Rolls back + toasts on HTTP failure.
     *
     * The backend merges partial payloads into the existing module
     * dict, so callers may pass diffs. To mirror that behaviour
     * locally, the optimistic apply merges into the cached row (if
     * present); otherwise it inserts a fresh entry containing only the
     * delta.
     */
    async updateModule(
        module: string,
        settings: Record<string, unknown>,
    ): Promise<void> {
        await this.runOptimistic({
            apply: m => {
                const n = new Map(m);
                const current = n.get(module);
                // TODO(post-merge): backend replaces non-dict modules outright (settings_manager.py:107-111).
                // All current modules are dicts so merge-here matches; revisit if a non-dict module is added.
                const mergedSettings = {
                    ...(current?.settings ?? {}),
                    ...settings,
                };
                n.set(module, {
                    id: module,
                    module,
                    settings: mergedSettings,
                });
                return n;
            },
            request: () => firstValueFrom(this.api.updateModule(module, settings)),
            errorMessage: `Couldn't save ${module} settings — reverted.`,
        });
    }
}
