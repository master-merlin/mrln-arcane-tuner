import { Injectable, signal } from '@angular/core';

export interface TrainingHandoff {
    /** The config snapshot to apply to the training form. */
    config: Record<string, unknown>;
    /**
     * `reload` patches the form via loadExternalConfig (auto-template
     * suppressed — matches legacy "Reload into Settings"); `template`
     * selects/edits an existing template (`templateId`) and patches the form
     * with `config`, recreating the template only if it was deleted. A bare
     * `templateName` (no id) falls back to the legacy create path.
     */
    mode: 'reload' | 'template';
    templateName?: string;
    /**
     * The existing template this handoff edits/reloads. Present for the
     * Projects "Edit" action (Bug A) and for a job whose stored config recorded
     * the template it was built from (Bug B). When set, the Training screen
     * selects this exact template as the save-target instead of cloning a copy.
     */
    templateId?: string;
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
