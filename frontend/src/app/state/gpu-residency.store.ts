import { DestroyRef, Injectable, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Observable, tap } from 'rxjs';
import { RuntimeConfigService } from '../services/runtime-config.service';
import { WebSocketService } from '../services/websocket.service';

/** One GPU-plugin service's residency. Mirrors the backend `GpuServiceState`.
 *  `model` is display-only — the three services own their key formats
 *  independently, so nothing here may branch on its shape. */
export interface GpuServiceState {
    service: string;
    label: string;
    loaded: boolean;
    model: string | null;
}

export interface GpuLoadedResponse {
    any_loaded: boolean;
    services: GpuServiceState[];
}

export interface GpuUnloadResponse extends GpuLoadedResponse {
    unloaded: string[];
    skipped: { service: string; reason: string }[];
}

/**
 * How often to re-ask whether anything holds the GPU.
 *
 * There is no event for "a model finished loading" — loads happen inside
 * request handlers and batch workers that broadcast task/dataset events, not
 * GPU-residency ones — so this is a poll. 15s is chosen against what the poll
 * is FOR: a control that appears when a model becomes resident. The user's own
 * actions (captioning, masking, a rescan) already take seconds, so a quarter
 * minute of lag on the button appearing is invisible, while the request itself
 * reads three class attributes and returns.
 */
const POLL_INTERVAL_MS = 15_000;

/**
 * Shared "does anything hold the GPU right now?" state, backed by
 * `GET /api/system/gpu/loaded`.
 *
 * **The poll stops while the tab is hidden.** A studio is left open in a
 * background tab for hours; a timer that kept firing there would hold a poll
 * loop (and a backend request every 15s) open for a window nobody is looking
 * at. `visibilitychange` tears the interval down and, on return, refreshes
 * immediately — so the first thing a returning user sees is current, not one
 * interval stale.
 */
@Injectable({ providedIn: 'root' })
export class GpuResidencyStore {
    private http = inject(HttpClient);
    private rtc = inject(RuntimeConfigService);
    private ws = inject(WebSocketService);

    /** Precomputed by the backend, not derived here: a client that does not
     *  yet know about a newly added service still gates correctly. */
    readonly anyLoaded = signal(false);
    readonly services = signal<GpuServiceState[]>([]);
    /** True while an unload is in flight — the button's busy state. */
    readonly unloading = signal(false);

    private timerId?: ReturnType<typeof setInterval>;
    private readonly onVisibility = () => this.syncToVisibility();

    constructor() {
        // A reconnect means the backend may be a fresh process, which by
        // definition holds nothing — the cached "loaded" would be a lie.
        this.ws.reconnected$.pipe(takeUntilDestroyed()).subscribe(() => this.refresh());

        document.addEventListener('visibilitychange', this.onVisibility);
        inject(DestroyRef).onDestroy(() => {
            document.removeEventListener('visibilitychange', this.onVisibility);
            this.stopPolling();
        });
    }

    /** Begin (or resume) the bounded poll. Idempotent; call from app init. */
    start(): void {
        this.syncToVisibility();
    }

    /** Re-read residency once. Safe to call at any time. */
    refresh(): void {
        this.http.get<GpuLoadedResponse>(`${this.rtc.apiUrl}/system/gpu/loaded`).subscribe({
            next: r => this.apply(r),
            // A backend that is down or too old holds nothing we can free; the
            // positive-only button must hide rather than offer a dead action.
            error: () => this.apply({ any_loaded: false, services: [] }),
        });
    }

    /**
     * Free every GPU-plugin model that is not busy.
     *
     * The response is authoritative state read AFTER the unload, so it is
     * applied directly rather than triggering another `refresh()` — one
     * producer for the residency fact, no window where the button lingers on
     * stale state.
     */
    unloadAll(): Observable<GpuUnloadResponse> {
        this.unloading.set(true);
        return this.http
            .post<GpuUnloadResponse>(`${this.rtc.apiUrl}/system/gpu/unload`, {})
            .pipe(
                tap({
                    next: r => {
                        this.apply(r);
                        this.unloading.set(false);
                    },
                    error: () => {
                        this.unloading.set(false);
                        this.refresh();
                    },
                }),
            );
    }

    private apply(r: GpuLoadedResponse): void {
        this.services.set(r.services ?? []);
        this.anyLoaded.set(!!r.any_loaded);
    }

    private syncToVisibility(): void {
        if (document.hidden) {
            this.stopPolling();
            return;
        }
        this.refresh();
        if (this.timerId === undefined) {
            this.timerId = setInterval(() => this.refresh(), POLL_INTERVAL_MS);
        }
    }

    private stopPolling(): void {
        if (this.timerId !== undefined) {
            clearInterval(this.timerId);
            this.timerId = undefined;
        }
    }
}
