import { Injectable, inject, signal } from '@angular/core';
import { DatasetService } from '../services/dataset';

/** Shared LLM-endpoint (Ollama/LM Studio) availability, backed by GET /api/llm-refine/models. */
@Injectable({ providedIn: 'root' })
export class LlmAvailabilityStore {
    private api = inject(DatasetService);
    readonly available = signal<boolean>(false);
    readonly installed = signal<string[]>([]);
    readonly checked = signal<boolean>(false);

    /** Re-probe the endpoint. Safe to call on app init and after settings save. */
    refresh(): void {
        this.api.listRefineModels().subscribe({
            next: r => { this.available.set(!!r.available); this.installed.set(r.installed ?? []); this.checked.set(true); },
            error: () => { this.available.set(false); this.installed.set([]); this.checked.set(true); },
        });
    }
}
