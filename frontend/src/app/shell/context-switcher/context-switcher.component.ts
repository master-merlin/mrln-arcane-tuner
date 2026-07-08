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
import { OverlayStore } from '../../state/overlay.store';
import { IcoComponent } from '../../icons/ico.component';

/**
 * Scope picker shown in the topbar (and later in the workspace toolbar).
 * Ported from `.agent/workdir/design/mrln/project/hifi/app-ctx.jsx:59-129`.
 *
 * The popover options are real `role="menuitem"` buttons with a roving
 * tabindex (T18): only the active option is in the tab order, Arrow keys move
 * the active option (and focus), Home/End jump to the ends, Escape closes and
 * restores focus to the trigger, and focus moves into the panel on open.
 */
@Component({
    selector: 'app-context-switcher',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './context-switcher.component.html',
    styleUrl: './context-switcher.component.css',
})
export class ContextSwitcherComponent {
    protected scope = inject(ScopeStore);
    protected projects = inject(ProjectService);
    private overlay = inject(OverlayStore);
    protected open = signal(false);

    /** Roving-tabindex active option index within the open menu. */
    protected activeIndex = signal(0);

    /** Active project (resolved from the scope id), or null for Global. */
    protected activeProject = computed(() => {
        const id = this.scope.projectId();
        if (!id) return null;
        return this.projects.allProjects().find(p => p.id === id) ?? null;
    });

    /** Total menu options: Global + each project + "New project…". */
    protected menuCount = computed(() => this.projects.allProjects().length + 2);

    /** Flat index of the "New project…" option (always last). */
    protected newProjectIndex = computed(() => this.projects.allProjects().length + 1);

    private host = inject(ElementRef<HTMLElement>);

    @HostListener('document:mousedown', ['$event'])
    protected outsideClick(e: MouseEvent) {
        if (this.open() && !this.host.nativeElement.contains(e.target as Node)) {
            this.open.set(false);
        }
    }

    protected toggle() {
        const willOpen = !this.open();
        this.open.set(willOpen);
        if (willOpen) {
            // Start on the currently-active option, then move focus into the
            // panel once the menu items have rendered (microtask mirrors the
            // modal-layer focus pattern).
            this.activeIndex.set(this.initialActiveIndex());
            queueMicrotask(() => this.focusActive());
        }
    }

    /** Index of the option matching the current scope (Global or active project). */
    private initialActiveIndex(): number {
        if (this.scope.scope().kind === 'global') return 0;
        const pid = this.scope.projectId();
        const idx = this.projects.allProjects().findIndex(p => p.id === pid);
        return idx >= 0 ? idx + 1 : 0;
    }

    protected pickGlobal() {
        this.scope.setGlobal();
        this.close();
    }

    protected pickProject(id: string) {
        this.scope.setProject(id);
        this.close();
    }

    protected newProject() {
        this.overlay.openModal('project-dialog', { mode: 'create' });
        this.close();
    }

    /** Close the panel and restore focus to the trigger pill. */
    private close() {
        this.open.set(false);
        queueMicrotask(() => this.triggerEl()?.focus());
    }

    /** Keyboard navigation within the open menu (roving tabindex + Escape). */
    protected onMenuKeydown(e: KeyboardEvent) {
        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                this.move(1);
                break;
            case 'ArrowUp':
                e.preventDefault();
                this.move(-1);
                break;
            case 'Home':
                e.preventDefault();
                this.activeIndex.set(0);
                this.focusActive();
                break;
            case 'End':
                e.preventDefault();
                this.activeIndex.set(this.menuCount() - 1);
                this.focusActive();
                break;
            case 'Escape':
                e.preventDefault();
                this.close();
                break;
        }
    }

    private move(delta: number) {
        const count = this.menuCount();
        this.activeIndex.set((this.activeIndex() + delta + count) % count);
        this.focusActive();
    }

    private menuItemEls(): HTMLElement[] {
        const host = this.host.nativeElement as HTMLElement;
        return Array.from(host.querySelectorAll<HTMLElement>('[role="menuitem"]'));
    }

    private focusActive() {
        this.menuItemEls()[this.activeIndex()]?.focus();
    }

    private triggerEl(): HTMLElement | null {
        const host = this.host.nativeElement as HTMLElement;
        return host.querySelector<HTMLElement>('.ctx-pill');
    }
}
