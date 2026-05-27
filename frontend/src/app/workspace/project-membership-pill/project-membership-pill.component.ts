import {
    ChangeDetectionStrategy,
    Component,
    computed,
    effect,
    inject,
    input,
    signal,
} from '@angular/core';
import { ProjectService } from '../../services/project.service';
import { ScopeStore } from '../../state/scope.store';
import { ToastService } from '../../services/toast';
import { IcoComponent } from '../../icons/ico.component';

type MembershipState = 'hidden' | 'add' | 'member';

/** Minimal dataset shape accepted by this pill (id may be absent for legacy entries). */
export interface PillDataset {
    id?: string;
    name: string;
}

/**
 * Topbar pill that adds/removes the open dataset to/from the project the
 * global {@link ScopeStore} scope currently points at. Renders nothing in
 * Global scope (or with no dataset). Retrofit of the dead legacy
 * `ViewerToolbarComponent` project-context affordance into the live
 * workspace topbar — but driven by the shared `ScopeStore`, not a local
 * dropdown.
 */
@Component({
    selector: 'app-project-membership-pill',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @switch (state()) {
            @case ('add') {
                <button type="button" class="btn primary sm"
                        (click)="add()"
                        [title]="'Add ' + (dataset()?.name ?? 'dataset') + ' to ' + activeProjectName()">
                    <app-ico name="Plus" [size]="13"/> Add to project
                </button>
            }
            @case ('member') {
                <span class="chip success ws-membership-in">
                    <app-ico name="Check" [size]="11"/>
                    In {{ activeProjectName() }}
                    <button type="button" class="ws-membership-remove"
                            (click)="remove()"
                            [title]="'Remove ' + (dataset()?.name ?? 'dataset') + ' from ' + activeProjectName()"
                            aria-label="Remove from project">
                        <app-ico name="X" [size]="11"/>
                    </button>
                </span>
            }
        }
    `,
    styles: [`
        .ws-membership-in { display: inline-flex; align-items: center; gap: 6px; }
        .ws-membership-remove {
            display: inline-flex; align-items: center; justify-content: center;
            padding: 0; margin-left: 2px;
            background: transparent; border: 0; cursor: pointer;
            color: inherit; opacity: 0.7;
        }
        .ws-membership-remove:hover { opacity: 1; color: var(--color-danger); }
    `],
})
export class ProjectMembershipPillComponent {
    private scope = inject(ScopeStore);
    private projects = inject(ProjectService);
    private toast = inject(ToastService);

    /** The dataset currently open in the workspace. */
    dataset = input<PillDataset | null>(null);

    /** Dataset ids in the active-scope project. Refreshed by the effect below. */
    private projectDatasetIds = signal<Set<string>>(new Set());

    /** Display name of the active-scope project (fallback "project"). */
    activeProjectName = computed<string>(() => {
        const id = this.scope.projectId();
        if (!id) return 'project';
        return this.projects.allProjects().find(p => p.id === id)?.name ?? 'project';
    });

    /** hidden (Global / no dataset) · add (not a member) · member (is a member). */
    state = computed<MembershipState>(() => {
        const pid = this.scope.projectId();
        const ds = this.dataset();
        if (!pid || !ds) return 'hidden';
        return this.projectDatasetIds().has(this.keyOf(ds)) ? 'member' : 'add';
    });

    constructor() {
        // Refetch membership whenever the scoped project changes. Mirrors the
        // pattern in DatasetsScreen.refreshProjectMembership.
        effect(() => {
            const pid = this.scope.projectId();
            if (!pid) {
                this.projectDatasetIds.set(new Set());
                return;
            }
            this.projects.getProjectDatasets(pid).subscribe({
                next: rows => this.projectDatasetIds.set(new Set(rows.map(r => r.id ?? r.name))),
                error: () => this.projectDatasetIds.set(new Set()),
            });
        });
    }

    add(): void {
        const pid = this.scope.projectId();
        const ds = this.dataset();
        if (!pid || !ds) return;
        const key = this.keyOf(ds);
        this.projects.addProjectDataset(pid, key).subscribe({
            next: () => {
                this.projectDatasetIds.update(s => new Set(s).add(key));
                this.toast.success(`Added "${ds.name}" to ${this.activeProjectName()}.`);
                this.projects.loadProjects();
            },
            error: (err: { error?: { detail?: string }; message?: string }) =>
                this.toast.error('Failed to add to project: ' + (err?.error?.detail || err?.message)),
        });
    }

    remove(): void {
        const pid = this.scope.projectId();
        const ds = this.dataset();
        if (!pid || !ds) return;
        const key = this.keyOf(ds);
        this.projects.removeProjectDataset(pid, key).subscribe({
            next: () => {
                this.projectDatasetIds.update(s => { const n = new Set(s); n.delete(key); return n; });
                this.toast.success(`Removed "${ds.name}" from ${this.activeProjectName()}.`);
                this.projects.loadProjects();
            },
            error: (err: { error?: { detail?: string }; message?: string }) =>
                this.toast.error('Failed to remove from project: ' + (err?.error?.detail || err?.message)),
        });
    }

    private keyOf(ds: PillDataset): string {
        return ds.id ?? ds.name;
    }
}
