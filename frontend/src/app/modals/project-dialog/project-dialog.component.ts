import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators, type AbstractControl } from '@angular/forms';
import { Router } from '@angular/router';
import { IcoComponent } from '../../icons/ico.component';
import { ProjectService } from '../../services/project.service';
import { ToastService } from '../../services/toast';
import { OverlayStore } from '../../state/overlay.store';
import { ScopeStore } from '../../state/scope.store';

interface ProjectDialogData {
    mode?: 'create' | 'edit';
    projectId?: string;
}

const COLOR_PRESETS = [
    '#ef4444', '#f97316', '#f59e0b', '#84cc16',
    '#10b981', '#06b6d4', '#3b82f6', '#8b5cf6',
    '#d946ef', '#f43f5e',
] as const;

/** Rejects null/empty/whitespace-only values. */
function nonEmptyTrimmed(c: AbstractControl): { required: true } | null {
    const v = c.value as string | null;
    return (v ?? '').trim().length === 0 ? { required: true } : null;
}

/**
 * Project dialog modal — create or edit a project.
 *
 * Mode + (optional) projectId arrive via `overlay.topModal()?.data`. In
 * edit mode the form is pre-populated from {@link ProjectService.allProjects}
 * and the submit hits `updateProject`; in create mode it hits `createProject`.
 *
 * Form fields mirror the orphaned `project-dialog.ts` component (name,
 * description, color picker). Dataset selection is intentionally deferred
 * to the project detail's Datasets tab — keeping the modal narrow.
 */
@Component({
    selector: 'app-modal-project-dialog',
    standalone: true,
    imports: [ReactiveFormsModule, IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">{{ isEdit() ? 'EDIT' : 'CREATE' }}</div>
                <div class="pd-title">{{ isEdit() ? 'Edit project' : 'New project' }}</div>
            </div>
            <button class="icon-btn" type="button" (click)="close()" aria-label="Close">×</button>
        </div>

        <div class="modal-body">
            <form [formGroup]="form" (ngSubmit)="submit()">
                <!-- Preview card -->
                <div class="pd-preview">
                    <div class="pd-preview-badge"
                         [style.background]="'linear-gradient(135deg, ' + color() + ', color-mix(in oklab, ' + color() + ' 70%, black))'"
                         [style.box-shadow]="'0 4px 12px ' + color() + '40'">
                        {{ initials() }}
                    </div>
                    <div class="pd-preview-text">
                        <div class="pd-preview-name">{{ form.get('name')?.value || 'New project' }}</div>
                        <div class="pd-preview-desc">{{ form.get('description')?.value || 'no description yet' }}</div>
                    </div>
                </div>

                <label class="field-label pd-mt">Project name *</label>
                <input class="input" type="text" formControlName="name" autocomplete="off"
                       placeholder="e.g. CivitAI Car" autofocus/>

                <label class="field-label pd-mt">Description</label>
                <textarea class="input pd-textarea" rows="3" formControlName="description"
                          placeholder="What is this project about?"></textarea>

                <label class="field-label pd-mt">Project color</label>
                <div class="pd-colors">
                    @for (preset of colorPresets; track preset) {
                        <button type="button" class="pd-color-swatch"
                                [class.active]="color() === preset"
                                [style.background]="preset"
                                [title]="preset"
                                (click)="pickColor(preset)"></button>
                    }
                </div>
            </form>
        </div>

        <div class="modal-foot">
            @if (isEdit()) {
                <button class="btn danger-out" type="button" (click)="deleteProject()" style="margin-right: auto;">
                    <app-ico name="Trash2" [size]="13"/> Delete project
                </button>
            }
            <button class="btn ghost" type="button" (click)="close()">Cancel</button>
            <button class="btn primary" type="button"
                    [disabled]="form.invalid || submitting()"
                    (click)="submit()">
                <app-ico name="Check" [size]="14"/>
                {{ submitting() ? 'Saving…' : (isEdit() ? 'Save changes' : 'Create project') }}
            </button>
        </div>
    `,
    styles: [`
        .pd-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .pd-mt { margin-top: 12px; }
        .pd-textarea { min-height: 64px; resize: vertical; font-family: var(--font-sans); }
        .pd-preview {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 14px 16px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
        }
        .pd-preview-badge {
            width: 44px;
            height: 44px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 800;
            font-size: 16px;
            flex-shrink: 0;
        }
        .pd-preview-text { flex: 1; min-width: 0; }
        .pd-preview-name {
            font-size: 14px;
            font-weight: 700;
            color: var(--color-text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .pd-preview-desc {
            font-size: 11px;
            color: var(--color-text-muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .pd-colors {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 4px;
        }
        .pd-color-swatch {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid transparent;
            transition: transform 120ms, border-color 120ms;
            padding: 0;
        }
        .pd-color-swatch.active {
            border-color: var(--color-text-primary);
            transform: scale(1.05);
        }
    `],
})
export class ProjectDialogComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    private scope = inject(ScopeStore);
    private router = inject(Router);
    private projects = inject(ProjectService);
    private toast = inject(ToastService);
    private fb = inject(FormBuilder);

    protected readonly colorPresets = COLOR_PRESETS;

    protected data: ProjectDialogData = (this.overlay.topModal()?.data as ProjectDialogData) ?? {};
    protected isEdit = signal<boolean>(false);

    protected color = signal<string>('#6366f1');
    protected submitting = signal(false);

    protected form = this.fb.nonNullable.group({
        name: this.fb.nonNullable.control('', [Validators.required, nonEmptyTrimmed]),
        description: this.fb.nonNullable.control(''),
    });

    protected initials = computed(() => {
        const name = this.form.get('name')?.value || '';
        return name.split(/\s+/).map((w: string) => w[0] ?? '').slice(0, 2).join('').toUpperCase() || '?';
    });

    ngOnInit(): void {
        const mode = this.data.mode ?? (this.data.projectId ? 'edit' : 'create');
        this.isEdit.set(mode === 'edit');

        if (mode === 'edit' && this.data.projectId) {
            const project = this.projects.allProjects().find(p => p.id === this.data.projectId);
            if (project) {
                this.form.patchValue({
                    name: project.name,
                    description: project.description || '',
                });
                this.color.set(project.color || '#6366f1');
            } else {
                // Fallback: try fetching it
                this.projects.getProject(this.data.projectId).subscribe({
                    next: (p) => {
                        this.form.patchValue({ name: p.name, description: p.description || '' });
                        this.color.set(p.color || '#6366f1');
                    },
                    error: () => this.toast.error('Project not found.'),
                });
            }
        }
    }

    protected pickColor(c: string): void {
        this.color.set(c);
    }

    protected close(): void {
        this.overlay.closeModal();
    }

    protected submit(): void {
        if (this.form.invalid || this.submitting()) return;
        const { name, description } = this.form.getRawValue();
        const payload = {
            name: name.trim(),
            description: description.trim(),
            color: this.color(),
        };
        this.submitting.set(true);

        if (this.isEdit() && this.data.projectId) {
            this.projects.updateProject(this.data.projectId, payload).subscribe({
                next: () => {
                    this.toast.success('Project updated.');
                    this.projects.loadProjects();
                    this.submitting.set(false);
                    this.close();
                },
                error: (err: { error?: { detail?: string }; message?: string }) => {
                    this.toast.error('Failed to update project: ' + (err?.error?.detail || err?.message));
                    this.submitting.set(false);
                },
            });
        } else {
            this.projects.createProject(payload.name, payload.description, payload.color).subscribe({
                next: () => {
                    this.toast.success('Project created.');
                    this.projects.loadProjects();
                    this.submitting.set(false);
                    this.close();
                },
                error: (err: { error?: { detail?: string }; message?: string }) => {
                    this.toast.error('Failed to create project: ' + (err?.error?.detail || err?.message));
                    this.submitting.set(false);
                },
            });
        }
    }

    protected deleteProject(): void {
        if (!this.isEdit() || !this.data.projectId) return;
        const id = this.data.projectId;
        const name = this.form.get('name')?.value || 'this project';
        // TODO(frontend): replace with overlay.openModal('confirm', ...) when Phase 8 lands.
        if (!confirm(`Delete project "${name}"? Datasets and images are kept; project-specific settings are removed.`)) return;
        this.submitting.set(true);
        this.projects.deleteProject(id).subscribe({
            next: () => {
                this.toast.success(`Deleted project "${name}".`);
                // Scope fallback: if the deleted project is the active one, drop to Global.
                if (this.scope.projectId() === id) {
                    console.log('[scope] active project deleted; falling back to Global');
                    this.scope.setGlobal();
                    void this.router.navigate(['/projects']);
                }
                this.projects.loadProjects();
                this.submitting.set(false);
                this.close();
            },
            error: (err: { error?: { detail?: string }; message?: string }) => {
                this.toast.error('Failed to delete project: ' + (err?.error?.detail || err?.message));
                this.submitting.set(false);
            },
        });
    }
}
