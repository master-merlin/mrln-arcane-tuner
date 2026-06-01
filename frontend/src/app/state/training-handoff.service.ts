import { Injectable, signal } from '@angular/core';

export interface TrainingHandoff {
    /** The config snapshot to apply to the training form. */
    config: Record<string, unknown>;
    /**
     * `reload` patches the form via loadExternalConfig (auto-template
     * suppressed — matches legacy "Reload into Settings"); `template`
     * imports a named template via importTemplate.
     */
    mode: 'reload' | 'template';
    templateName?: string;
    definitionId?: string;
}

/**
 * One-shot bus for handing a job's config from the Jobs screen to the
 * Training screen across a route change. The Jobs screen sets `pending` and
 * navigates to /training; the training screen consumes it once its config
 * editor is ready, then clears it.
 *
 * Mirrors the legacy app-shell `pendingConfig` signal + effect, adapted for
 * the routed redesign (no shared parent component to hold the state).
 */
@Injectable({ providedIn: 'root' })
export class TrainingHandoffService {
    readonly pending = signal<TrainingHandoff | null>(null);

    set(handoff: TrainingHandoff): void {
        this.pending.set(handoff);
    }

    consume(): TrainingHandoff | null {
        const h = this.pending();
        if (h) this.pending.set(null);
        return h;
    }
}
