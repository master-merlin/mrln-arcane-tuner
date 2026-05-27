import { Injectable, Signal, computed, inject, signal } from '@angular/core';
import { WebSocketService } from '../services/websocket.service';

export interface DownloadProgress {
    source: 'curated' | 'hf';
    model_id: string;
    category: string;
    status: 'starting' | 'downloading' | 'complete' | 'error';
    current_bytes: number;
    total_bytes: number | null;
    percent: number | null;
    error: string | null;
}

export interface RecentDownload extends DownloadProgress {
    finishedAt: number;
}

export const RECENT_CAP = 5;
export const RECENT_TTL_MS = 5 * 60 * 1000;

function keyOf(p: { source: string; model_id: string }): string {
    return `${p.source}::${p.model_id}`;
}

/**
 * Root-scoped store consuming `model.download_progress` WS events.
 *
 * Active downloads are keyed by `${source}::${model_id}` and surfaced via
 * `active()` / `activeCount()`. Completions and errors move into a capped
 * `recent` list (5 entries, 5 min TTL pruned lazily on read).
 */
@Injectable({ providedIn: 'root' })
export class ModelDownloadStore {
    private _active = signal<Map<string, DownloadProgress>>(new Map());
    private _recent = signal<RecentDownload[]>([]);
    private _activeForKeyCache = new Map<string, Signal<DownloadProgress | undefined>>();

    readonly active = computed<DownloadProgress[]>(() =>
        Array.from(this._active().values()),
    );
    readonly activeCount = computed(() => this._active().size);

    /** Recents pruned lazily — drops entries older than RECENT_TTL_MS. */
    readonly recent = computed<RecentDownload[]>(() => {
        const cutoff = Date.now() - RECENT_TTL_MS;
        return this._recent().filter(r => r.finishedAt >= cutoff);
    });

    /** Weighted by total_bytes when every active has a known total; null otherwise. */
    readonly aggregatePercent = computed<number | null>(() => {
        const xs = this.active();
        if (xs.length === 0) return null;
        if (xs.some(x => x.total_bytes == null || x.percent == null)) return null;
        const total = xs.reduce((a, x) => a + (x.total_bytes ?? 0), 0);
        const done = xs.reduce((a, x) => a + (x.current_bytes ?? 0), 0);
        return total > 0 ? Math.round((done * 100) / total) : null;
    });

    /** Memoized signal for one download's live progress (subscription source for panel UI). */
    activeForKey(key: string): Signal<DownloadProgress | undefined> {
        const cached = this._activeForKeyCache.get(key);
        if (cached) return cached;
        const c = computed(() => this._active().get(key));
        this._activeForKeyCache.set(key, c);
        return c;
    }

    constructor(ws: WebSocketService) {
        ws.on<DownloadProgress>('model.download_progress').subscribe(msg => this.apply(msg));
    }

    private apply(msg: DownloadProgress): void {
        const key = keyOf(msg);
        if (msg.status === 'starting' || msg.status === 'downloading') {
            this._active.update(m => new Map(m).set(key, msg));
            return;
        }
        // complete | error
        this._active.update(m => { const n = new Map(m); n.delete(key); return n; });
        this._recent.update(r => [
            { ...msg, finishedAt: Date.now() },
            ...r,
        ].slice(0, RECENT_CAP));
    }
}
