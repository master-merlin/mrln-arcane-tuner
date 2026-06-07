import { ChangeDetectionStrategy, Component, computed, inject, input } from '@angular/core';
import { IcoComponent } from '../../icons/ico.component';
import { Task, TaskStore } from '../../state/task.store';

/**
 * Queued-task banner for background-task modals.
 *
 * Heavy dataset ops (caption / mask / rescan / crop / mass-edit / harmonize) all
 * share ONE serialized `gpu` lane, so launching a second one while another runs
 * leaves it `pending` until the lane drains. The modals flip to their progress
 * view immediately on launch, which made a queued task look like it had started
 * (stuck at 0%). This banner renders ONLY while the bound task is `pending` and
 * spells out that it's waiting, plus its 1-based position in the queue. It
 * disappears the moment the task starts (status → running).
 *
 * Bind the modal's live task signal: `<app-task-queue-hint [task]="task()"/>`.
 */
@Component({
    selector: 'app-task-queue-hint',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @if (queued()) {
            <div class="tq-hint" data-testid="task-queue-hint" role="status">
                <app-ico name="Clock" [size]="14"/>
                <span>
                    Another task is running — this job is <b>queued</b>
                    (#{{ position() }} in queue) and will start automatically.
                </span>
            </div>
        }
    `,
    styles: [`
        .tq-hint {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 10px 14px;
            margin-bottom: 12px;
            background: color-mix(in oklab, var(--color-warning) 8%, transparent);
            border: 1px solid color-mix(in oklab, var(--color-warning) 25%, transparent);
            border-radius: var(--radius-theme-md);
            font-size: 11.5px;
            color: var(--color-text-secondary);
            line-height: 1.5;
        }
        .tq-hint app-ico { color: var(--color-warning); flex-shrink: 0; margin-top: 1px; }
        .tq-hint b { color: var(--color-text-primary); font-weight: 600; }
    `],
})
export class TaskQueueHintComponent {
    private tasks = inject(TaskStore);

    /** The modal's live task (TaskStore.byId(...)()), or null/undefined before launch. */
    task = input<Task | null | undefined>(undefined);

    protected queued = computed(() => this.task()?.status === 'pending');

    /**
     * 1-based position of this task in the shared run queue: every active task
     * that executes before it — the one running now plus any task enqueued
     * earlier (lower `created_at`) — counts as "ahead", then +1 for itself. All
     * user-visible tasks share the gpu lane, so `active()` IS the lane order;
     * silent `background`-lane tasks are filtered out of the store already.
     */
    protected position = computed(() => {
        const me = this.task();
        if (!me) return 0;
        const ahead = this.tasks.active().filter(t =>
            t.id !== me.id &&
            (t.status === 'running' || (t.status === 'pending' && t.created_at < me.created_at)),
        ).length;
        return ahead + 1;
    });
}
