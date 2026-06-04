import { Injectable } from '@angular/core';

export interface RuntimeConfig {
    backendPort: number;
    frontendPort: number;
}

const DEFAULT_CONFIG: RuntimeConfig = {
    backendPort: 8000,
    frontendPort: 4200,
};

/**
 * Loads runtime configuration from `/runtime-config.json` (served by the
 * Angular dev server from `public/`).  This allows the backend to change
 * ports and have the frontend discover them on next load — no rebuild needed.
 *
 * Hooked into Angular via `APP_INITIALIZER` in `app.config.ts`.
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

    get backendPort(): number {
        return this.config.backendPort;
    }

    get frontendPort(): number {
        return this.config.frontendPort;
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
            .then((data: RuntimeConfig) => {
                this.config = { ...DEFAULT_CONFIG, ...data };
                console.log('[RuntimeConfig] loaded', this.config);
            })
            .catch(err => {
                console.warn('[RuntimeConfig] failed to load, using defaults', err);
                this.config = DEFAULT_CONFIG;
            });
    }
}
