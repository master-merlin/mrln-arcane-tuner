import { ChangeDetectionStrategy, Component, Injector, computed, effect, inject, signal, untracked } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';

import { IcoComponent } from '../../../icons/ico.component';
import { FormatBytesPipe } from '../../../shared/format-bytes.pipe';
import { DatasetService, LegacyThumbnailSurvey } from '../../../services/dataset';
import { ToastService } from '../../../services/toast';
import { TaskStore } from '../../../state/task.store';

/**
 * "N datasets still hold stale thumbnail caches" — the migration affordance
 * for the `a5003618` relayout (LANE-40).
 *
 * The relayout moved every rendition to `.thumbnails/<edge>/<stem>.webp` and
 * purged the flat one from the scan path, which does nothing until a scan
 * runs. Covers are NOT affected — the backend regenerates from source, so the
 * pixels have been right all along; what is left is unreachable, unreclaimable
 * bytes. That is why this offers a *cleanup*, not a rescan, and why the copy
 * says "reclaim" rather than anything about stale images: telling the user
 * their covers are wrong when they are not would buy a pointless full rescan
 * of the library.
 *
 * Renders NOTHING when there is nothing to reclaim, so it costs a mature
 * install one GET and no pixels.
 */
@Component({
    selector: 'app-thumbnail-migration-banner',
    standalone: true,
    imports: [IcoComponent, FormatBytesPipe],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './thumbnail-migration-banner.html',
    styleUrl: './thumbnail-migration-banner.css',
})
export class ThumbnailMigrationBanner {
    private api = inject(DatasetService);
    private toast = inject(ToastService);
    private injector = inject(Injector);
    private tasks: TaskStore | null = null;

    /** TaskStore is resolved LAZILY, on the first sweep this banner starts.
     *  It is a root singleton that opens a WebSocket subscription and
     *  re-syncs the task list over HTTP the moment it is constructed, and
     *  this component renders nothing at all on a library with nothing to
     *  migrate — which is every library, permanently, once the migration has
     *  run. Injecting it eagerly would make a component that is invisible
     *  99% of the time spin up the task machinery on every visit to the
     *  Datasets screen. */
    private taskStore(): TaskStore {
        return (this.tasks ??= this.injector.get(TaskStore));
    }

    /** Latest survey, or null before the first response / after a failed one. */
    protected readonly survey = signal<LegacyThumbnailSurvey | null>(null);
    /** Id of the sweep this banner started, while it is still in flight. */
    private readonly taskId = signal<string | null>(null);
    protected readonly busy = signal(false);

    /** Gates on the server's precomputed total, never on `datasets.length` —
     *  the count and the list are one payload but only one of them is the
     *  question being asked (ARCHITECTURE D10). */
    protected readonly visible = computed(() => (this.survey()?.total_files ?? 0) > 0);

    private readonly liveTask = computed(() => {
        const id = this.taskId();
        return id ? this.taskStore().byId(id)() : undefined;
    });

    constructor() {
        this.refresh();
        // Re-survey once the sweep leaves the active set. The banner's own
        // state is derived from disk, so it is re-measured rather than
        // predicted: a cancelled or partially failed sweep must leave the
        // banner showing what is genuinely still there.
        effect(() => {
            const task = this.liveTask();
            if (!task || task.status === 'pending' || task.status === 'running') return;
            untracked(() => {
                this.taskId.set(null);
                this.busy.set(false);
                if (task.status === 'failed') {
                    this.toast.error(`Thumbnail cache repair failed: ${task.error ?? 'unknown error'}`);
                }
                this.refresh();
            });
        });
    }

    /** Re-measure. A failure leaves the previous value alone rather than
     *  clearing the banner — an unreachable backend is not evidence that the
     *  files are gone. */
    protected refresh(): void {
        this.api.getLegacyThumbnailSurvey().subscribe({
            next: s => this.survey.set(s),
            error: () => { /* keep the last known survey */ },
        });
    }

    protected migrate(): void {
        if (this.busy()) return;
        this.busy.set(true);
        this.api.startThumbnailMigration().subscribe({
            next: started => {
                this.taskId.set(started.task_id);
                this.toast.info(
                    `Reclaiming ${started.files} stale thumbnails across ${started.dataset_count} datasets…`,
                );
            },
            error: (err: HttpErrorResponse) => {
                this.busy.set(false);
                // 409 = already running, or nothing left to do; either way the
                // truth is on disk, so re-survey instead of guessing.
                this.toast.warning(
                    err.status === 409
                        ? (err.error?.detail ?? 'A thumbnail cleanup is already running.')
                        : 'Could not start the thumbnail cleanup.',
                );
                this.refresh();
            },
        });
    }
}
