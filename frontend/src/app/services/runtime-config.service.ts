import { Injectable } from '@angular/core';

/**
 * Shape of `/runtime-config.json`.
 *
 * `backendPort` / `frontendPort` are **deprecated**. Nothing reads them: every
 * URL this service hands out is origin-relative, because the FastAPI backend
 * serves the built SPA and proxies `/api` itself, so a port in a config file
 * could only ever disagree with the port the page was actually loaded from.
 *
 * They are still accepted, and `runtime-config.json` is still shipped and still
 * fetched. It is a user-editable deployment artifact — an install may already
 * have one on disk with these keys set, and removing the keys would turn a
 * working file into a rejected one. Deprecate, never drop (ARCHITECTURE D2).
 * They may be removed a release after they stop appearing in shipped configs.
 */
export interface RuntimeConfig {
    /** @deprecated Unused; URLs are origin-relative. Accepted for compatibility. */
    backendPort?: number;
    /** @deprecated Unused; URLs are origin-relative. Accepted for compatibility. */
    frontendPort?: number;
}

const DEFAULT_CONFIG: RuntimeConfig = {};

/** Ports are 1-65535; 0 is "any" and never a valid target. */
function isValidPort(value: unknown): value is number {
    return typeof value === 'number' && Number.isInteger(value) && value >= 1 && value <= 65535;
}

/**
 * Loads runtime configuration from `/runtime-config.json`.
 *
 * Hooked into Angular via `APP_INITIALIZER` in `app.config.ts` — the app does
 * not bootstrap until this resolves, so it must never reject.
 */
@Injectable({ providedIn: 'root' })
export class RuntimeConfigService {
    private config: RuntimeConfig = DEFAULT_CONFIG;

    get apiUrl(): string {
        return '/api';
    }

    get wsUrl(): string {
        const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
        return `${proto}://${window.location.host}/api/ws`;
    }

    get mediaBaseUrl(): string {
        return '/media';
    }

    /**
     * Validate and adopt a parsed config body.
     *
     * Unknown keys are ignored rather than merged: the previous
     * `{ ...DEFAULT_CONFIG, ...data }` spread copied whatever the file
     * contained — including a non-object, which would have produced a config
     * of indexed characters — straight onto the service's state.
     */
    private adopt(data: unknown): void {
        if (data === null || typeof data !== 'object' || Array.isArray(data)) {
            console.warn('[RuntimeConfig] ignoring malformed config (not an object)');
            this.config = DEFAULT_CONFIG;
            return;
        }

        const source = data as Record<string, unknown>;
        const next: RuntimeConfig = {};

        for (const key of ['backendPort', 'frontendPort'] as const) {
            if (!(key in source)) continue;
            if (isValidPort(source[key])) {
                next[key] = source[key] as number;
            } else {
                console.warn(`[RuntimeConfig] ignoring out-of-range ${key}:`, source[key]);
            }
        }

        this.config = next;
    }

    /**
     * Called by APP_INITIALIZER. Must return a Promise so Angular waits
     * for it to resolve before bootstrapping the app.
     */
    load(): Promise<void> {
        return fetch(`/runtime-config.json?t=${Date.now()}`)
            .then(r => {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.json();
            })
            .then(data => this.adopt(data))
            .catch(err => {
                // A missing or unreadable config is not an error: every value in
                // it is optional and deprecated. Bootstrapping must continue.
                console.warn('[RuntimeConfig] failed to load, using defaults', err);
                this.config = DEFAULT_CONFIG;
            });
    }
}
