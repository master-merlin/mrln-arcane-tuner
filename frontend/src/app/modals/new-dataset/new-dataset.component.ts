import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { IcoComponent } from '../../icons/ico.component';
import { DatasetStore } from '../../state/dataset.store';
import { OverlayStore } from '../../state/overlay.store';

const STANDARD_CLASSIFIERS = ['vehicle', 'person', 'style', 'object', 'landscape'] as const;
const NAME_FORBIDDEN = /[<>:"/\\|?*]/;

/**
 * New Dataset modal — wraps the design's "NewDataset" modal in the
 * new modal shell (`.modal-head` / `.modal-body` / `.modal-foot`).
 *
 * Form fields mirror the existing orphaned `dataset-form-modal`
 * (name + category + description). The "path" field from the design
 * source is dropped because the backend's `POST /datasets` payload
 * doesn't accept a path — the dataset folder is created under the
 * server-configured datasets root, derived from the dataset name.
 */
@Component({
    selector: 'app-modal-new-dataset',
    standalone: true,
    imports: [ReactiveFormsModule, IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">CREATE</div>
                <div class="modal-title">New Dataset</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body">
            <form [formGroup]="form" (ngSubmit)="submit()">
                <label class="field-label">Name</label>
                <input class="input" type="text" formControlName="name" autocomplete="off" placeholder="My_New_Concept" autofocus/>
                @if (nameInvalid()) {
                    <div class="field-error">Name contains forbidden characters (&lt; &gt; : " / \\ | ? *)</div>
                }

                <label class="field-label nd-mt">Category</label>
                <select class="select" formControlName="classifier">
                    <option value="">None / Uncategorized</option>
                    @for (s of standardClassifiers; track s) {
                        <option [value]="s">{{ titlecase(s) }}</option>
                    }
                </select>

                <label class="field-label nd-mt">Description</label>
                <textarea class="input nd-textarea" rows="3" formControlName="description"
                          placeholder="Optional description for this dataset"></textarea>
            </form>
        </div>

        <div class="modal-foot">
            <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
            <button class="btn primary" type="button"
                    [disabled]="form.invalid || nameInvalid() || submitting()"
                    (click)="submit()">
                <app-ico name="Plus" [size]="14"/>
                {{ submitting() ? 'Creating…' : 'Create Dataset' }}
            </button>
        </div>
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .nd-mt { margin-top: 12px; }
        .nd-textarea { min-height: 64px; resize: vertical; font-family: var(--font-sans); }
        .field-error {
            color: var(--color-danger);
            font-size: 11px;
            margin-top: 4px;
        }
    `],
})
export class NewDatasetModalComponent {
    private fb = inject(FormBuilder);
    private datasets = inject(DatasetStore);
    protected overlay = inject(OverlayStore);

    protected standardClassifiers = STANDARD_CLASSIFIERS;
    protected submitting = signal(false);

    protected form = this.fb.nonNullable.group({
        name: ['', [Validators.required]],
        classifier: [''],
        description: [''],
    });

    /** Reactive validity for the forbidden-chars check. */
    protected nameInvalid = computed<boolean>(() => {
        const v = this.form.controls.name.value ?? '';
        return NAME_FORBIDDEN.test(v);
    });

    protected titlecase(s: string): string {
        return s.charAt(0).toUpperCase() + s.slice(1);
    }

    async submit(): Promise<void> {
        if (this.form.invalid || this.nameInvalid() || this.submitting()) return;
        const { name, description, classifier } = this.form.getRawValue();
        this.submitting.set(true);
        const created = await this.datasets.createDataset(
            (name ?? '').trim(),
            description ?? '',
            classifier ?? '',
        );
        this.submitting.set(false);
        if (created) {
            this.overlay.closeModal();
        }
        // Errors surface as a toast from DatasetStore.createDataset; leave the modal open.
    }
}
