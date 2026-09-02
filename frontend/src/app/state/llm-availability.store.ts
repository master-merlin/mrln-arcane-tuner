import { Injectable, computed, inject, signal } from '@angular/core';
import { DatasetService } from '../services/dataset';
import { WebSocketService } from '../services/websocket.service';

/** `host:port` of a served endpoint URL for a caption — the scheme and any
 *  path are noise beside a button. Falls back to the raw string for a value
 *  `URL` cannot parse, so the caption never hides what the backend said. */
export function refineEndpointHost(url: string | null | undefined): string | null {
    if (!url) return null;
    try { return new URL(url).host || url; } catch { return url; }
}

/** Shared LLM-endpoint (Ollama/LM Studio) availability, backed by GET /api/llm-refine/models. */
@Injectable({ providedIn: 'root' })
export class LlmAvailabilityStore {
    private api = inject(DatasetService);
    private ws = inject(WebSocketService);
    readonly available = signal<boolean>(false);
    readonly installed = signal<string[]>([]);
    readonly checked = signal<boolean>(false);
    /** LANE-76: WHAT a model-less refine is served with — the default model
     *  the probe judged and the endpoint it probed, as served; null until the
     *  probe answers (or when it failed to). The Refine button names these so
     *  it cannot be read as using the caption provider above it. */
    readonly model = signal<string | null>(null);
    readonly endpoint = signal<string | null>(null);
    readonly endpointHost = computed(() => refineEndpointHost(this.endpoint()));
    /** The backend's own sentence for why a refine cannot start (the same
     *  text `POST /captions/refine-batch` refuses with — LANE-57); null when
     *  it may, or when the probe itself failed to answer. */
    readonly reason = signal<string | null>(null);

    /** A refine that names NO model (the detail sidebar's) may not start:
     *  the endpoint is down, the configured default model is not installed
     *  there, or the probe has not answered yet. The ONE gate every
     *  model-less Refine trigger reads — mirrors `captionStartBlocked`
     *  (LANE-65); a probe still out never passes it (LANE-57, LANE-70). */
    readonly blocked = computed(() => !this.available() || this.reason() != null);
    /** The sentence beside a blocked Refine: the backend's own (the one
     *  `POST /captions/refine-batch` refuses with), else why there is none. */
    readonly blockedReason = computed(() => this.reason()
        ?? (this.checked()
            ? 'LLM endpoint unreachable — configure it in Server settings'
            : 'Checking the LLM refine endpoint…'));

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
            next: r => {
                this.available.set(!!r.available); this.installed.set(r.installed ?? []); this.reason.set(r.unavailable_reason ?? null);
                this.model.set(r.model || null); this.endpoint.set(r.endpoint || null);
                this.checked.set(true);
            },
            error: () => {
                this.available.set(false); this.installed.set([]); this.reason.set(null);
                this.model.set(null); this.endpoint.set(null);
                this.checked.set(true);
            },
        });
    }
}
