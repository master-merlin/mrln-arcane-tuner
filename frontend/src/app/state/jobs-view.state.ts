import { Injectable, computed, signal } from '@angular/core';
import { Job, JobStatus } from '../services/job';

/**
 * Thin shared bus for the Jobs screen's 3-pane layout.
 *
 * The `training-job-queue` component remains the owner of all live job state
 * (WS-accumulated logs/warnings, auto-queue, sampling, archive, optimistic
 * delete). It *publishes* its live lists + the current selection here so the
 * sibling center detail pane can render the LIVE selected job (streaming logs)
 * without duplicating the WS machinery or coupling to the queue component.
 */
@Injectable({ providedIn: 'root' })
export class JobsViewState {
    /** Live active queue (pending/running/paused), WS-accumulated. */
    readonly activeJobs = signal<Job[]>([]);
    /** Live archive (completed/failed/stopped). */
    readonly archivedJobs = signal<Job[]>([]);
    /** Explicit user selection; null falls back to the running job. */
    readonly selectedId = signal<string | null>(null);

    /** Currently focused job: explicit selection, else the running job. */
    readonly selectedJob = computed<Job | null>(() => {
        const active = this.activeJobs();
        const id = this.selectedId();
        if (id) {
            const all = [...active, ...this.archivedJobs()];
            const explicit = all.find((j) => j.id === id);
            if (explicit) return explicit;
        }
        return active.find((j) => j.status === JobStatus.RUNNING) ?? null;
    });

    select(id: string): void {
        this.selectedId.set(id);
    }
}
