import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';
import { ToastService } from '../../services/toast';
import { TemplateService, Template } from '../../services/template.service';
import { IcoComponent } from '../../icons/ico.component';
import {
    DatasetCaptionSettingsComponent,
    CaptionSettingsState,
} from '../../components/dataset/dataset-caption-settings/dataset-caption-settings';
import {
    DatasetMaskingSettingsComponent,
    MaskingSettingsState,
} from '../../components/dataset/dataset-masking-settings/dataset-masking-settings';

/** Modal payload — set via `overlay.openModal('template-edit', TemplateEditData)`. */
export interface TemplateEditData {
    domain: 'captioning' | 'masking';
    template: Template;
    /** Fired after a successful save so the opener can refresh its list. */
    onSaved?: () => void;
}

/**
 * Edit a captioning/masking template with the familiar settings dialog, but
 * locked to the one template (no internal switcher), with auto-save disabled
 * and an explicit Save. Reuses `dataset-caption-settings` / `dataset-masking-settings`
 * in their embedded mode (`presetTemplate` + `hideTemplateBar` + `autoSave=false`).
 */
@Component({
    selector: 'app-modal-template-edit',
    standalone: true,
    imports: [IcoComponent, DatasetCaptionSettingsComponent, DatasetMaskingSettingsComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">EDIT {{ data()?.domain === 'masking' ? 'MASKING' : 'CAPTION' }} TEMPLATE</div>
                <div class="modal-title">{{ data()?.template?.name }}</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body">
            @if (data(); as d) {
                @if (d.domain === 'captioning') {
                    <app-dataset-caption-settings
                        [presetTemplate]="d.template"
                        [projectId]="d.template.project_id"
                        [hideTemplateBar]="true"
                        [autoSave]="false"
                        (settingsChanged)="state.set($event)"/>
                } @else {
                    <app-dataset-masking-settings
                        [presetTemplate]="d.template"
                        [projectId]="d.template.project_id"
                        [hideTemplateBar]="true"
                        [autoSave]="false"
                        (settingsChanged)="state.set($event)"/>
                }
            }
        </div>

        <div class="modal-foot">
            <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
            <button class="btn cta" type="button" [disabled]="!state() || saving()" (click)="save()">
                <app-ico name="Save" [size]="12"/> {{ saving() ? 'Saving…' : 'Save template' }}
            </button>
        </div>
    `,
})
export class TemplateEditModalComponent {
    protected overlay = inject(OverlayStore);
    private templates = inject(TemplateService);
    private toast = inject(ToastService);

    protected data = computed(() => this.overlay.topModal()?.data as TemplateEditData | undefined);
    protected state = signal<CaptionSettingsState | MaskingSettingsState | null>(null);
    protected saving = signal(false);

    protected save(): void {
        const d = this.data();
        const s = this.state();
        if (!d || !s) return;

        const payload: Partial<Template> = d.domain === 'captioning'
            ? {
                config: (s as CaptionSettingsState).params,
                system_prompt: (s as CaptionSettingsState).systemPrompt,
                model_id: (s as CaptionSettingsState).modelId,
            }
            : {
                config: (s as MaskingSettingsState).params,
                model_id: (s as MaskingSettingsState).modelId,
            };

        this.saving.set(true);
        this.templates.updateTemplate(d.domain, d.template.id, payload).subscribe({
            next: () => {
                this.saving.set(false);
                this.toast.success(`Template '${d.template.name}' saved.`);
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
