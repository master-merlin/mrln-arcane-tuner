import { TestBed } from '@angular/core/testing';
import { signal, WritableSignal } from '@angular/core';
import { of } from 'rxjs';
import { settle } from '../../../../testing/async';
import { DatasetsScreen } from '../datasets-screen';
import { DatasetStore } from '../../../state/dataset.store';
import { DatasetService } from '../../../services/dataset';
import { ProjectService } from '../../../services/project.service';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { ToastService } from '../../../services/toast';
import { ScopeStore } from '../../../state/scope.store';
import { OverlayStore } from '../../../state/overlay.store';
import { SearchStore } from '../../../state/search.store';

/**
 * Loading / empty tri-state (plan items D2 + D3).
 *
 * The library grid must render one of three states via a single `gridState()`
 * computed so the template can `@switch` on it:
 *   - `loading`      — first fetch in flight → skeleton cards (no false-empty flash)
 *   - `grid`         — datasets present → cards
 *   - `empty`        — load resolved with zero rows, no search / filters → CTA
 *   - `search-empty` — zero rows because the search query excluded them all
 *   - `filter-empty` — zero rows because active filter chips excluded them all
 *
 * The component is instantiated directly (no template render), so the gating
 * signal is asserted rather than the rendered DOM (matching the harness style
 * of the sibling specs in this folder).
 */
describe('DatasetsScreen — loading / empty tri-state', () => {
    let loading: WritableSignal<boolean>;
    let entities: WritableSignal<{ id: string; name: string }[]>;
    let query: WritableSignal<string>;

    function gridState(comp: DatasetsScreen): string {
        return (comp as unknown as { gridState: () => string }).gridState();
    }

    function make(): DatasetsScreen {
        return TestBed.runInInjectionContext(() => new DatasetsScreen());
    }

    beforeEach(() => {
        loading = signal(true);
        entities = signal<{ id: string; name: string }[]>([]);
        query = signal('');

        TestBed.configureTestingModule({
            providers: [
                {
                    provide: DatasetStore,
                    useValue: { loadAll: () => Promise.resolve(), entities, loading },
                },
                {
                    provide: DatasetService,
                    useValue: { getCacheStats: () => of(null), getLegacyThumbnailSurvey: () => of({ datasets: [], dataset_count: 0, total_files: 0, total_bytes: 0 }), getMpxDistribution: () => of(null) },
                },
                { provide: ProjectService, useValue: { getProjectDatasets: () => of([]) } },
                { provide: RuntimeConfigService, useValue: { mediaBaseUrl: '' } },
                { provide: ToastService, useValue: { success: () => {}, error: () => {} } },
                { provide: ScopeStore, useValue: { projectId: signal<string | null>(null) } },
                { provide: OverlayStore, useValue: { workspace: signal(null), openModal: () => {} } },
                { provide: SearchStore, useValue: { query, fields: signal(new Set<string>()) } },
            ],
        });
    });

    it('reports the loading state while the first fetch is in flight (even with 0 rows)', async () => {
        loading.set(true);
        const comp = make();
        TestBed.tick();
        await settle();
        expect(gridState(comp)).toBe('loading');
    });

    it('stays in the loading state even if some rows are already cached', async () => {
        loading.set(true);
        entities.set([{ id: 'd1', name: 'ds-1' }]);
        const comp = make();
        TestBed.tick();
        await settle();
        // Loading wins: we do not flash a partial grid mid-refresh.
        expect(gridState(comp)).toBe('loading');
    });

    it('shows the empty CTA only after loading resolves with zero datasets', async () => {
        loading.set(false);
        entities.set([]);
        const comp = make();
        TestBed.tick();
        await settle();
        expect(gridState(comp)).toBe('empty');
    });

    it('shows the grid when datasets are present and loading has resolved', async () => {
        loading.set(false);
        entities.set([{ id: 'd1', name: 'ds-1' }]);
        const comp = make();
        TestBed.tick();
        await settle();
        expect(gridState(comp)).toBe('grid');
    });

    it('distinguishes a search-empty result from the true-empty CTA', async () => {
        loading.set(false);
        entities.set([{ id: 'd1', name: 'ds-1' }]);
        query.set('no-such-dataset');
        const comp = make();
        TestBed.tick();
        await settle();
        expect(gridState(comp)).toBe('search-empty');
    });

    it('exposes a handful of skeleton slots to render placeholder cards', () => {
        const comp = make();
        const slots = (comp as unknown as { skeletonSlots: unknown[] }).skeletonSlots;
        expect(Array.isArray(slots)).toBe(true);
        expect(slots.length).toBeGreaterThan(0);
    });
});
