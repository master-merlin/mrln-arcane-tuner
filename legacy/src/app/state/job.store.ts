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

    async deleteJob(id: string): Promise<void> {
        await this.runOptimistic({
            apply: m => { const n = new Map(m); n.delete(id); return n; },
            request: () => firstValueFrom(this.api.deleteJob(id)),
            errorMessage: `Couldn't delete job — restored.`,
        });
    }
}
