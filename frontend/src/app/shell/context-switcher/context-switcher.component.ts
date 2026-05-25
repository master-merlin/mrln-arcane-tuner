import {
    ChangeDetectionStrategy,
    Component,
    ElementRef,
    HostListener,
    computed,
    inject,
    signal,
} from '@angular/core';
import { ProjectService } from '../../services/project.service';
import { ScopeStore } from '../../state/scope.store';
import { IcoComponent } from '../../icons/ico.component';

/**
 * Scope picker shown in the topbar (and later in the workspace toolbar).
 * Ported from `.agent/workdir/design/mrln/project/hifi/app-ctx.jsx:59-129`.
 */
@Component({
    selector: 'app-context-switcher',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './context-switcher.component.html',
})
export class ContextSwitcherComponent {
    protected scope = inject(ScopeStore);
    protected projects = inject(ProjectService);
    protected open = signal(false);

    /** Active project (resolved from the scope id), or null for Global. */
    protected activeProject = computed(() => {
        const id = this.scope.projectId();
        if (!id) return null;
        return this.projects.allProjects().find(p => p.id === id) ?? null;
    });

    private host = inject(ElementRef<HTMLElement>);

    @HostListener('document:mousedown', ['$event'])
    protected outsideClick(e: MouseEvent) {
        if (this.open() && !this.host.nativeElement.contains(e.target as Node)) {
            this.open.set(false);
        }
    }

    protected toggle() { this.open.update(o => !o); }
    protected pickGlobal() { this.scope.setGlobal(); this.open.set(false); }
    protected pickProject(id: string) { this.scope.setProject(id); this.open.set(false); }
}
