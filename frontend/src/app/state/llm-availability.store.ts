import { Injectable, inject, signal } from '@angular/core';
import { DatasetService } from '../services/dataset';
import { WebSocketService } from '../services/websocket.service';

/** Shared LLM-endpoint (Ollama/LM Studio) availability, backed by GET /api/llm-refine/models. */
@Injectable({ providedIn: 'root' })
export class LlmAvailabilityStore {
    private api = inject(DatasetService);
    private ws = inject(WebSocketService);
    readonly available = signal<boolean>(false);
    readonly installed = signal<string[]>([]);
    readonly checked = signal<boolean>(false);

    constructor() {
        // The Ollama sidecar restarts with the backend (the container launches
        // it from the same entrypoint), so a reconnect is exactly when this
        // answer goes stale. Callers only refresh on their own init, which
        // leaves an app that started against a dead backend showing
        // "unavailable" for the whole session.
        this.ws.reconnected$.subscribe(() => this.refresh());
    }

    /** Re-probe the endpoint. Safe to call on app init and after settings save. */
    refresh(): void {
        this.api.listRefineModels().subscribe({
            next: r => { this.available.set(!!r.available); this.installed.set(r.installed ?? []); this.checked.set(true); },
            error: () => { this.available.set(false); this.installed.set([]); this.checked.set(true); },
        });
    }
}
