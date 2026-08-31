import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal, WritableSignal } from '@angular/core';
import { of } from 'rxjs';
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
 * deleteDataset() migrated off the chained native confirm() double-prompt to a
 * single themed Confirm modal. Global-scope delete now carries an "also delete
 * files on disk" CHECKBOX whose state is passed to the API; project-scope is a
 * plain destructive confirm. The action fires only from the modal's onConfirm.
 */
describe('DatasetsScreen — deleteDataset via themed confirm', () => {
    let projectId: WritableSignal<string | null>;
    let openModal: Mock;
    let deleteDataset: Mock;
    let removeProjectDataset: Mock;

    const DS = { id: 'd1', name: 'ds-1' } as unknown as Parameters<DatasetsScreen['deleteDataset']>[0];
    const ev = { stopPropagation: () => {} } as unknown as Event;

    function make(): DatasetsScreen {
        return TestBed.runInInjectionContext(() => new DatasetsScreen());
    }
    function lastModal(): { checkboxLabel?: string; destructive?: boolean; onConfirm?: (c?: boolean) => void } {
        return openModal.mock.calls.at(-1)![1];
    }

    beforeEach(() => {
        projectId = signal<string | null>(null);
        openModal = vi.fn();
        deleteDataset = vi.fn().mockReturnValue(of({}));
        removeProjectDataset = vi.fn().mockReturnValue(of({}));

        TestBed.configureTestingModule({
            providers: [
                {
                    provide: DatasetStore,
                    useValue: { loadAll: () => Promise.resolve(), entities: signal([{ id: 'd1', name: 'ds-1' }]) },
                },
                {
                    provide: DatasetService,
                    useValue: { getCacheStats: () => of(null), getLegacyThumbnailSurvey: () => of({ datasets: [], dataset_count: 0, total_files: 0, total_bytes: 0 }), getMpxDistribution: () => of(null), deleteDataset },
                },
                {
                    provide: ProjectService,
                    useValue: {
                        getProjectDatasets: () => of([]),
                        removeProjectDataset,
                        bumpDatasetStat: () => {},
                        loadProjects: () => {},
                    },
                },
                { provide: RuntimeConfigService, useValue: { mediaBaseUrl: '' } },
                { provide: ToastService, useValue: { success: () => {}, error: () => {} } },
                { provide: ScopeStore, useValue: { projectId } },
                { provide: OverlayStore, useValue: { workspace: signal(null), openModal } },
                { provide: SearchStore, useValue: { query: signal(''), fields: signal(new Set<string>()) } },
            ],
        });
    });

    it('global scope: opens a destructive confirm with a "delete files" checkbox and does not delete yet', () => {
        const comp = make();
        (comp as unknown as { deleteDataset(d: unknown, e: Event): void }).deleteDataset(DS, ev);

        expect(openModal).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
        expect(lastModal().checkboxLabel).toBeTruthy();
        expect(deleteDataset).not.toHaveBeenCalled();
    });

    it('global scope: ticking the checkbox deletes files on disk (onConfirm(true))', () => {
        const comp = make();
        (comp as unknown as { deleteDataset(d: unknown, e: Event): void }).deleteDataset(DS, ev);
        lastModal().onConfirm!(true);
        expect(deleteDataset).toHaveBeenCalledWith('ds-1', true);
    });

    it('global scope: leaving the checkbox unticked removes from library only (onConfirm(false))', () => {
        const comp = make();
        (comp as unknown as { deleteDataset(d: unknown, e: Event): void }).deleteDataset(DS, ev);
        lastModal().onConfirm!(false);
        expect(deleteDataset).toHaveBeenCalledWith('ds-1', false);
    });

    it('project scope: plain destructive confirm (no checkbox); removes only on confirm', () => {
        projectId.set('p1');
        const comp = make();
        (comp as unknown as { deleteDataset(d: unknown, e: Event): void }).deleteDataset(DS, ev);

        expect(openModal).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
        expect(lastModal().checkboxLabel).toBeFalsy();
        expect(removeProjectDataset).not.toHaveBeenCalled();
        lastModal().onConfirm!();
        expect(removeProjectDataset).toHaveBeenCalledWith('p1', 'd1');
    });
});
