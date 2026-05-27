import {
    ChangeDetectionStrategy,
    Component,
    ElementRef,
    HostListener,
    computed,
    inject,
    signal,
} from '@angular/core';
import { NavigationEnd, Router } from '@angular/router';
import { toSignal } from '@angular/core/rxjs-interop';
import { filter } from 'rxjs';
import { ContextSwitcherComponent } from '../context-switcher/context-switcher.component';
import { IcoComponent } from '../../icons/ico.component';
import { DownloadIndicatorComponent } from './download-indicator.component';
import { ScopeStore } from '../../state/scope.store';
import { ProjectService } from '../../services/project.service';
import {
    ALL_DATASET_SEARCH_FIELDS,
    SearchStore,
    type DatasetSearchField,
} from '../../state/search.store';

interface Crumb {
    label: string;
    muted?: boolean;
    last?: boolean;
}

interface FieldOption {
    key: DatasetSearchField;
    label: string;
}

/**
 * Topbar — crumbs, scope switcher (on scope-aware routes), functional
 * search input (wired to {@link SearchStore}), notifications + settings
 * icons.
 *
 * The field-selector popover only renders on `/datasets`; on other
 * routes the input is still present but unused (placeholder reflects
 * the aspirational cross-route search).
 */
@Component({
    selector: 'app-topbar',
    standalone: true,
    imports: [ContextSwitcherComponent, IcoComponent, DownloadIndicatorComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './topbar.component.html',
})
export class TopbarComponent {
    private router = inject(Router);
    protected scope = inject(ScopeStore);
    protected projects = inject(ProjectService);
    protected search = inject(SearchStore);
    private host = inject(ElementRef<HTMLElement>);

    protected fieldMenuOpen = signal(false);

    protected readonly allFields: readonly FieldOption[] = [
        { key: 'name', label: 'Name' },
        { key: 'classifier', label: 'Classifier' },
        { key: 'tags', label: 'Tags' },
        { key: 'description', label: 'Description' },
        { key: 'trigger_word', label: 'Trigger word' },
        { key: 'notes', label: 'Notes' },
    ];

    protected readonly totalFields = ALL_DATASET_SEARCH_FIELDS.length;

    private url = toSignal(
        this.router.events.pipe(filter(e => e instanceof NavigationEnd)),
        { initialValue: null },
    );

    protected scopeLabel = computed(() => {
        void this.url();
        const id = this.scope.projectId();
        if (!id) return 'Global';
        return this.projects.allProjects().find(p => p.id === id)?.name ?? 'Global';
    });

    protected crumbs = computed<Crumb[]>(() => {
        void this.url();
        const path = this.router.url.split('?')[0];
        const scopeLabel = this.scopeLabel();
        if (path.startsWith('/datasets')) return [{ label: scopeLabel, muted: true }, { label: 'Datasets', last: true }];
        if (path.startsWith('/projects')) return [{ label: scopeLabel, muted: true }, { label: 'Projects', last: true }];
        if (path.startsWith('/training')) return [{ label: scopeLabel, muted: true }, { label: 'Training', last: true }];
        if (path.startsWith('/jobs'))     return [{ label: scopeLabel, muted: true }, { label: 'Jobs', last: true }];
        if (path.startsWith('/tools'))    return [{ label: 'Utilities', muted: true }, { label: 'LoRA Tools', last: true }];
        if (path.startsWith('/server'))   return [{ label: 'System', muted: true }, { label: 'Server', last: true }];
        return [];
    });

    protected showScope = computed(() => {
        void this.url();
        const path = this.router.url.split('?')[0];
        return ['/datasets', '/projects', '/training', '/jobs'].some(p => path.startsWith(p));
    });

    protected onDatasetsRoute = computed(() => {
        void this.url();
        return this.router.url.split('?')[0].startsWith('/datasets');
    });

    protected searchPlaceholder = computed(() =>
        this.onDatasetsRoute()
            ? 'Search datasets…'
            : 'Search datasets, captions, jobs…',
    );

    protected fieldButtonTitle = computed(() => {
        const labels = this.allFields
            .filter(f => this.search.fields().has(f.key))
            .map(f => f.label);
        return labels.length
            ? `Searching in: ${labels.join(', ')}`
            : 'No fields selected (falls back to Name)';
    });

    protected onSearchInput(event: Event): void {
        const value = (event.target as HTMLInputElement).value;
        this.search.query.set(value);
    }

    protected clearSearch(): void {
        this.search.query.set('');
    }

    protected onFieldToggle(key: DatasetSearchField, event: Event): void {
        const checked = (event.target as HTMLInputElement).checked;
        this.search.setField(key, checked);
    }

    protected toggleFieldMenu(): void {
        this.fieldMenuOpen.update(v => !v);
    }

    @HostListener('document:mousedown', ['$event'])
    protected onOutsidePointer(event: MouseEvent): void {
        if (!this.fieldMenuOpen()) return;
        if (!this.host.nativeElement.contains(event.target as Node)) {
            this.fieldMenuOpen.set(false);
        }
    }

    @HostListener('document:keydown.escape')
    protected onEsc(): void {
        if (this.fieldMenuOpen()) this.fieldMenuOpen.set(false);
    }
}
