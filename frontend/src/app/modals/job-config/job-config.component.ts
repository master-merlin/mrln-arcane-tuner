import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';
import { ToastService } from '../../services/toast';
import { JobService, Job } from '../../services/job';
import { IcoComponent } from '../../icons/ico.component';
import { JsonEditorComponent } from '../../ui/json-editor/json-editor.component';

/** Modal payload — set via `overlay.openModal('job-config', JobConfigData)`. */
export interface JobConfigData {
    job: Job;
    /** Fired after a successful save so the opener can refresh its list. */
    onSaved?: () => void;
}

/**
 * Raw JSON editor for a single job's training config. A second route to the
 * same edit the Run Config panel offers inline: loads the job's config as
 * pretty JSON in a CodeMirror editor; Save parses + validates and PUTs via
 * `updateJobConfig`. The backend rejects running/paused jobs, but the opener
 * only surfaces this for editable (pending/terminal) jobs. Invalid JSON blocks
 * Save.
 */
@Component({
    selector: 'app-modal-job-config',
    standalone: true,
    imports: [IcoComponent, JsonEditorComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">EDIT JOB CONFIG · {{ data()?.job?.status }}</div>
                <div class="modal-title">{{ data()?.job?.config?.['lora_name'] || data()?.job?.id }}</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body">
            <div class="jc-editor">
                <app-json-editor
                    [value]="initial()"
                    (valueChange)="text.set($event)"
                    (validChange)="valid.set($event)"/>
            </div>
            @if (!valid()) {
                <div class="jc-error">⚠ Invalid JSON — fix it to enable Save.</div>
            }
            @if (data()?.job?.status === 'pending') {
                <div class="jc-note">This job is pending — saved changes will be used when it runs.</div>
            }
        </div>

        <div class="modal-foot">
            <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
            <button class="btn cta" type="button" [disabled]="!valid() || saving()" (click)="save()">
                <app-ico name="Save" [size]="12"/> {{ saving() ? 'Saving…' : 'Save config' }}
            </button>
        </div>
    `,
    styles: [`
        .jc-editor { height: 460px; max-height: 62vh; border: 1px solid var(--color-border-subtle); border-radius: var(--radius-theme-md); overflow: hidden; }
        .jc-error { margin-top: 8px; font-size: 11.5px; color: var(--color-danger); font-weight: 500; }
        .jc-note { margin-top: 8px; font-size: 11.5px; color: var(--color-text-muted); }
    `],
})
export class JobConfigModalComponent {
    protected overlay = inject(OverlayStore);
    private jobs = inject(JobService);
    private toast = inject(ToastService);

    protected data = computed(() => this.overlay.topModal()?.data as JobConfigData | undefined);
    protected initial = computed(() => {
        const j = this.data()?.job;
        return j ? JSON.stringify(j.config ?? {}, null, 2) : '{}';
    });
    protected text = signal<string>('');
    protected valid = signal(true);
    protected saving = signal(false);

    protected save(): void {
        const d = this.data();
        if (!d) return;
        let parsed: Record<string, unknown>;
        try {
            parsed = JSON.parse(this.text() || this.initial());
        } catch {
            this.toast.error('Invalid JSON — cannot save.');
            return;
        }
        this.saving.set(true);
        this.jobs.updateJobConfig(d.job.id, parsed).subscribe({
            next: () => {
                this.saving.set(false);
                this.toast.success('Job config saved.');
                d.onSaved?.();
                this.overlay.closeModal();
            },
            error: (err: { error?: { detail?: string }; message?: string }) => {
                this.saving.set(false);
                this.toast.error('Save failed: ' + (err?.error?.detail || err?.message));
            },
        });
    }
}
