import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';
import { ToastService } from '../../services/toast';
import { TemplateService, Template, TemplateDomain } from '../../services/template.service';
import { IcoComponent } from '../../icons/ico.component';
import { JsonEditorComponent } from '../../ui/json-editor/json-editor.component';

/** Modal payload — set via `overlay.openModal('template-json', TemplateJsonData)`. */
export interface TemplateJsonData {
    domain: TemplateDomain;
    template: Template;
    onSaved?: () => void;
}

/**
 * Build the editable JSON payload for a template — the fields a user may
 * safely change, with server-managed columns (id, project_id, timestamps,
 * readonly/is_default, used_count, branched_from) excluded. Exported for
 * unit testing.
 */
export function buildEditablePayload(t: Template): Record<string, unknown> {
    const out: Record<string, unknown> = { name: t.name, config: t.config ?? {} };
    if (t.system_prompt !== undefined) out['system_prompt'] = t.system_prompt;
    // Captioning sibling of system_prompt: the {wildcard} substitution value.
    // Must be surfaced here too, else the JSON editor silently drops it on save
    // (the structured modal keeps it, hence the asymmetry the user saw).
    if (t.wildcard !== undefined) out['wildcard'] = t.wildcard;
    if (t.model_id !== undefined) out['model_id'] = t.model_id;
    if (t.definition_id !== undefined) out['definition_id'] = t.definition_id;
    return out;
}

/**
 * Raw JSON editor for any template (all three domains). Loads the editable
 * payload as pretty JSON in a CodeMirror editor; Save parses + validates and
 * PUTs the result via `updateTemplate`. Invalid JSON blocks Save.
 */
@Component({
    selector: 'app-modal-template-json',
    standalone: true,
    imports: [IcoComponent, JsonEditorComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">EDIT TEMPLATE JSON · {{ data()?.domain }}</div>
                <div class="modal-title">{{ data()?.template?.name }}</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body">
            <div class="tj-editor">
                <app-json-editor
                    [value]="initial()"
                    (valueChange)="text.set($event)"
                    (validChange)="valid.set($event)"/>
            </div>
            @if (!valid()) {
                <div class="tj-error">⚠ Invalid JSON — fix it to enable Save.</div>
            }
        </div>

        <div class="modal-foot">
            <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
            <button class="btn cta" type="button" [disabled]="!valid() || saving()" (click)="save()">
                <app-ico name="Save" [size]="12"/> {{ saving() ? 'Saving…' : 'Save JSON' }}
            </button>
        </div>
    `,
    styles: [`
        .tj-editor { height: 440px; max-height: 60vh; border: 1px solid var(--color-border-subtle); border-radius: var(--radius-theme-md); overflow: hidden; }
        .tj-error { margin-top: 8px; font-size: 11.5px; color: var(--color-danger); font-weight: 500; }
    `],
})
export class TemplateJsonEditModalComponent {
    protected overlay = inject(OverlayStore);
    private templates = inject(TemplateService);
    private toast = inject(ToastService);

    protected data = computed(() => this.overlay.topModal()?.data as TemplateJsonData | undefined);
    protected initial = computed(() => {
        const t = this.data()?.template;
        return t ? JSON.stringify(buildEditablePayload(t), null, 2) : '{}';
    });
    protected text = signal<string>('');
    protected valid = signal(true);
    protected saving = signal(false);

    protected save(): void {
        const d = this.data();
        if (!d) return;
        let parsed: Partial<Template>;
        try {
            parsed = JSON.parse(this.text() || this.initial());
        } catch {
            this.toast.error('Invalid JSON — cannot save.');
            return;
        }
        this.saving.set(true);
        this.templates.updateTemplate(d.domain, d.template.id, parsed).subscribe({
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
