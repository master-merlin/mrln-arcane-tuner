import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal, WritableSignal } from '@angular/core';
import { settle } from '../../../../testing/async';
import { of } from 'rxjs';
import { DatasetsScreen } from '../datasets-screen';
import { DatasetStore } from '../../../state/dataset.store';
import { DatasetService } from '../../../services/dataset';
import { ProjectService } from '../../../services/project.service';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { ToastService } from '../../../services/toast';
import { ScopeStore } from '../../../state/scope.store';
import { OverlayStore, WorkspaceState } from '../../../state/overlay.store';
import { SearchStore } from '../../../state/search.store';

/**
 * Regression guard for the stale project-membership filter: when a dataset is
 * added to the scoped project from the workspace "Add to project" pill, the
 * scope does not change — so the library grid (which stays mounted beneath the
 * workspace overlay) must re-sync its membership filter when the workspace
 * closes, otherwise the newly-added dataset stays hidden.
 *
 * The component class is instantiated directly (no template render) so the
 * effect / computed logic can be exercised in isolation; `TestBed.tick()`
 * flushes the effect and `await settle()` resolves the membership fetch.
 */
describe('DatasetsScreen — project membership refresh', () => {
    let projectId: WritableSignal<string | null>;
    let workspace: WritableSignal<WorkspaceState | null>;
    let getProjectDatasets: Mock;

    function ids(comp: DatasetsScreen): string[] {
        return (comp as unknown as {
            visibleDatasets: () => {
                id: string;
            }[];
        })
            .visibleDatasets()
            .map(d => d.id);
    }

    beforeEach(() => {
        projectId = signal<string | null>(null);
        workspace = signal<WorkspaceState | null>(null);
        getProjectDatasets = vi.fn().mockReturnValue(of([]));

        TestBed.configureTestingModule({
            providers: [
                {
                    provide: DatasetStore,
                    useValue: {
                        loadAll: () => Promise.resolve(),
                        entities: signal([{ id: 'd1', name: 'ds-1' }]),
                    },
                },
                {
                    provide: DatasetService,
                    useValue: {
                        getCacheStats: () => of(null),
                        getMpxDistribution: () => of(null),
                    },
                },
                { provide: ProjectService, useValue: { getProjectDatasets } },
                { provide: RuntimeConfigService, useValue: { mediaBaseUrl: '' } },
                { provide: ToastService, useValue: { success: () => { }, error: () => { } } },
                { provide: ScopeStore, useValue: { projectId } },
                { provide: OverlayStore, useValue: { workspace } },
                {
                    provide: SearchStore,
                    useValue: { query: signal(''), fields: signal(new Set<string>()) },
                },
            ],
        });
    });

    // Baseline: proves the harness flushes the membership effect at all.
    it('shows datasets that belong to the scoped project', async () => {
        projectId.set('p1');
        getProjectDatasets.mockReturnValue(of([{ id: 'd1', name: 'ds-1' }]));
        const comp = TestBed.runInInjectionContext(() => new DatasetsScreen());
        TestBed.tick();
        await settle();
        expect(ids(comp)).toEqual(['d1']);
    });

    // The bug: a dataset added to the already-scoped project from the workspace
    // pill must appear in the library after the workspace closes.
    it('re-syncs membership when the workspace closes (newly-added dataset appears)', async () => {
        projectId.set('p1');
        getProjectDatasets.mockReturnValue(of([])); // p1 has no datasets yet
        const comp = TestBed.runInInjectionContext(() => new DatasetsScreen());
        TestBed.tick();
        await settle();
        expect(ids(comp)).toEqual([]); // d1 not in p1 → filtered out

        // Pill adds d1 to p1 from inside the workspace, then the user returns.
        getProjectDatasets.mockReturnValue(of([{ id: 'd1', name: 'ds-1' }]));
        workspace.set({ datasetId: 'd1', mode: 'browse', imageIndex: 0 }); // open
        TestBed.tick();
        await settle();
        workspace.set(null); // close → must trigger a membership re-fetch
        TestBed.tick();
        await settle();

        expect(ids(comp)).toEqual(['d1']);
        expect(getProjectDatasets).toHaveBeenCalledWith('p1');
    });
});
