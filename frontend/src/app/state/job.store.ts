import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { EntityStore } from './entity-store';
import { Job, JobService } from '../services/job';
import { WebSocketService } from '../services/websocket.service';
import { ToastService } from '../services/toast';

/**
 * Per-domain store for Job entities.
 *
 * `loadAll` is widened to public (override-widens-visibility) because it is
 * part of the store's external contract — bootstrap code calls it on app
 * start, and tests invoke it directly without needing a wrapper.
 */
@Injectable({ providedIn: 'root' })
export class JobStore extends EntityStore<Job> {
    protected entityName = 'job';
    private api = inject(JobService);

    constructor(ws: WebSocketService, toast: ToastService) {
        super(ws, toast);
    }

    public override async loadAll(): Promise<void> {
        const jobs = await firstValueFrom(this.api.listJobs());
        this.setAll(jobs);
    }

    /**
     * Loads historical/archived jobs and MERGES them into the store
     * (without clearing active jobs already loaded via `loadAll`). The
     * archive page seeds these so `deleteJob` can optimistically remove
     * an archived row.
     */
    async loadHistory(): Promise<void> {
        const jobs = await firstValueFrom(this.api.listJobHistory());
        for (const job of jobs) {
            this.upsert(job);
        }
    }

    async deleteJob(id: string, force = false, deleteFiles = false): Promise<void> {
        const result = await this.runOptimistic({
            apply: m => { const n = new Map(m); n.delete(id); return n; },
            request: () => firstValueFrom(this.api.deleteJob(id, force, deleteFiles)),
            errorMessage: `Couldn't delete job — restored.`,
        });
        // The record went; the files are a SEPARATE outcome and can fail on
        // their own (a locked file, a folder already moved). Saying nothing
        // here would put us back where DECISION-29 started — the user is told
        // files are gone and they are not. Only speak when they asked for it.
        if (result.ok && deleteFiles && !result.value.files_deleted) {
            this.toast.error(
                `Job deleted, but its files are still on disk — ${
                    result.value.files_error ?? 'the folder could not be removed'
                }`,
            );
        }
    }
}
