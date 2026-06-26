import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';
import type { JobCheckpointMeta } from '../../services/job';

export interface ResumeJobDialogData {
    jobId: string;
    /** Resumable checkpoints only (filtered by the opener). */
    checkpoints: JobCheckpointMeta[];
    onRestart: (wipe: boolean) => void;
    onContinue: (checkpointDir: string) => void;
}

type ResumeMode = 'restart' | 'continue';

/**
 * Resume-job modal — choose how to relaunch a stopped/terminal training job
 * that has at least one resumable checkpoint:
 *   • Restart from 0 (with an optional "wipe previous output" checkbox), or
 *   • Continue from a selected checkpoint.
 *
 * Presentational only: it collects the choice and invokes the `onRestart` /
 * `onContinue` callbacks passed via `overlay.topModal()?.data`, then closes.
 * The Jobs screen owns the actual service calls + reload.
 */
@Component({
    selector: 'app-modal-resume-job',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">RESUME</div>
                <div class="rj-title">Resume training</div>
            </div>
            <button class="icon-btn" type="button" (click)="close()" aria-label="Close">×</button>
        </div>

        <div class="modal-body">
            <label class="rj-opt">
                <input type="radio" name="rj-mode" [checked]="mode() === 'continue'"
                       (change)="mode.set('continue')"/>
                <span>
                    <span class="rj-opt-title">Continue from checkpoint</span>
                    <span class="rj-opt-desc">Resume optimizer, scheduler and step count from a saved checkpoint.</span>
                </span>
            </label>

            @if (mode() === 'continue') {
                <select class="input rj-select" [value]="selectedDir() ?? ''"
                        (change)="selectedDir.set($any($event.target).value)">
                    @for (c of checkpoints(); track c.checkpoint_dir) {
                        <option [value]="c.checkpoint_dir">{{ label(c) }}</option>
                    }
                </select>
            }

            <label class="rj-opt rj-mt">
                <input type="radio" name="rj-mode" [checked]="mode() === 'restart'"
                       (change)="mode.set('restart')"/>
                <span>
                    <span class="rj-opt-title">Restart from 0</span>
                    <span class="rj-opt-desc">Begin a fresh run from step 0.</span>
                </span>
            </label>

            @if (mode() === 'restart') {
                <label class="rj-check">
                    <input type="checkbox" [checked]="wipe()"
                           (change)="wipe.set($any($event.target).checked)"/>
                    Wipe previous output (checkpoints, samples, logs)
                </label>
            }
        </div>

        <div class="modal-foot">
            <button class="btn ghost" type="button" (click)="close()">Cancel</button>
            <button class="btn primary" type="button"
                    [disabled]="mode() === 'continue' && !selectedDir()"
                    (click)="confirm()">
                {{ mode() === 'continue' ? 'Continue' : 'Restart' }}
            </button>
        </div>
    `,
    styles: [`
        .rj-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .rj-opt { display: flex; gap: 10px; align-items: flex-start; cursor: pointer; }
        .rj-opt input { margin-top: 3px; }
        .rj-opt-title { display: block; font-weight: 600; font-size: 13px; color: var(--color-text-primary); }
        .rj-opt-desc { display: block; font-size: 11px; color: var(--color-text-muted); }
        .rj-mt { margin-top: 14px; }
        .rj-select { margin: 8px 0 4px 26px; width: calc(100% - 26px); }
        .rj-check { display: flex; gap: 8px; align-items: center; font-size: 12px; margin: 8px 0 0 26px; cursor: pointer; }
    `],
})
export class ResumeJobModalComponent {
    protected overlay = inject(OverlayStore);

    protected data: ResumeJobDialogData =
        (this.overlay.topModal()?.data as ResumeJobDialogData);

    /** Resumable checkpoints, newest first (highest step / final). */
    protected checkpoints = signal<JobCheckpointMeta[]>(
        [...(this.data?.checkpoints ?? [])].sort((a, b) => b.step - a.step),
    );

    mode = signal<ResumeMode>('continue');
    wipe = signal<boolean>(true);
    selectedDir = signal<string | null>(
        this.checkpoints()[0]?.checkpoint_dir ?? null,
    );

    protected hasCheckpoints = computed(() => this.checkpoints().length > 0);

    protected label(c: JobCheckpointMeta): string {
        const step = c.is_final ? 'final' : `step ${c.step}`;
        const when = c.created_at ? new Date(c.created_at * 1000).toLocaleString() : '';
        return when ? `${step} — ${when}` : step;
    }

    confirm(): void {
        if (this.mode() === 'restart') {
            this.data.onRestart(this.wipe());
        } else {
            const dir = this.selectedDir();
            if (!dir) return;
            this.data.onContinue(dir);
        }
        this.close();
    }

    protected close(): void {
        this.overlay.closeModal();
    }
}
