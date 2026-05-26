import { Injectable, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { NavigationEnd, Router } from '@angular/router';
import { filter } from 'rxjs/operators';

export type DatasetSearchField =
    | 'name'
    | 'classifier'
    | 'tags'
    | 'description'
    | 'trigger_word'
    | 'notes';

export const ALL_DATASET_SEARCH_FIELDS: readonly DatasetSearchField[] = [
    'name',
    'classifier',
    'tags',
    'description',
    'trigger_word',
    'notes',
] as const;

/**
 * Shared search state for the topbar input + the datasets screen filter.
 *
 * `query` is the live substring text. `fields` is the set of dataset
 * fields the query is matched against. Both are signals so consumers
 * react automatically.
 *
 * The query is cleared whenever the router navigates AWAY from
 * `/datasets`, so stale text doesn't silently keep filtering when the
 * user comes back later. Field selection is preserved across route
 * changes because it's a session-level user preference.
 */
@Injectable({ providedIn: 'root' })
export class SearchStore {
    readonly query = signal('');
    readonly fields = signal<Set<DatasetSearchField>>(
        new Set(ALL_DATASET_SEARCH_FIELDS),
    );

    constructor() {
        const router = inject(Router);
        router.events
            .pipe(
                filter((e): e is NavigationEnd => e instanceof NavigationEnd),
                takeUntilDestroyed(),
            )
            .subscribe(e => {
                if (!e.urlAfterRedirects.startsWith('/datasets')) {
                    this.query.set('');
                }
            });
    }

    setField(field: DatasetSearchField, enabled: boolean): void {
        const next = new Set(this.fields());
        if (enabled) next.add(field);
        else next.delete(field);
        this.fields.set(next);
    }

    resetFields(): void {
        this.fields.set(new Set(ALL_DATASET_SEARCH_FIELDS));
    }
}
