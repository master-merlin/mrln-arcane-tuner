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
 * D5 — multi-select + contextual bulk action bar.
 *
 * The library grid gains a per-card selection model (`selected` set keyed by
 * dataset id) and a bulk bar that runs the existing per-dataset operations
 * across the whole selection. Bulk Delete MUST route through the themed
 * `confirm` modal and act only on `onConfirm`. Selecting a card must not open
 * it (stopPropagation on the checkbox).
 *
 * Component is instantiated directly (no render) mirroring the sibling specs.
 */
describe('DatasetsScreen — selection model + bulk actions', () => {
    let projectId: WritableSignal<string | null>;
    let openModal: Mock;
    let deleteDataset: Mock;
    let rescanDataset: Mock;
    let removeProjectDataset: Mock;
    let addProjectDataset: Mock;
    let loadAll: Mock;

    const DS1 = { id: 'd1', name: 'ds-1' };
    const DS2 = { id: 'd2', name: 'ds-2' };
    const DS3 = { id: 'd3', name: 'ds-3' };
    const ev = () => ({ stopPropagation: vi.fn() }) as unknown as Event;

    interface Sel {
        selected: WritableSignal<Set<string>>;
        isSelected(d: unknown): boolean;
        toggleSelection(d: unknown, e?: Event): void;
        clearSelection(): void;
        toggleSelectAll(): void;
        selectionCount(): number;
        allVisibleSelected(): boolean;
        selectedDatasets(): { id: string }[];
        bulkDelete(): void;
        bulkRescan(): void;
        bulkAddToProject(projectId: string, projectName: string): void;
    }

    function make(): DatasetsScreen & Sel {
        return TestBed.runInInjectionContext(() => new DatasetsScreen()) as DatasetsScreen & Sel;
    }
    function lastModal(): { message?: string; destructive?: boolean; onConfirm?: (c?: boolean) => void } {
        return openModal.mock.calls.at(-1)![1];
    }

    beforeEach(() => {
        projectId = signal<string | null>(null);
        openModal = vi.fn();
        deleteDataset = vi.fn().mockReturnValue(of({}));
        rescanDataset = vi.fn().mockReturnValue(of({ task_id: 't1' }));
        removeProjectDataset = vi.fn().mockReturnValue(of({}));
        addProjectDataset = vi.fn().mockReturnValue(of({ status: 'ok' }));
        loadAll = vi.fn().mockResolvedValue(undefined);

        TestBed.configureTestingModule({
            providers: [
                {
                    provide: DatasetStore,
                    useValue: {
                        loadAll,
                        loading: signal(false),
                        entities: signal([DS1, DS2, DS3]),
                    },
                },
                {
                    provide: DatasetService,
                    useValue: {
                        getCacheStats: () => of(null), getLegacyThumbnailSurvey: () => of({ datasets: [], dataset_count: 0, total_files: 0, total_bytes: 0 }),
                        getMpxDistribution: () => of(null),
                        deleteDataset,
                        rescanDataset,
                    },
                },
                {
                    provide: ProjectService,
                    useValue: {
                        getProjectDatasets: () => of([]),
                        allProjects: signal([{ id: 'p1', name: 'Proj One' }]),
                        addProjectDataset,
                        removeProjectDataset,
                        bumpDatasetStat: () => {},
                        loadProjects: () => {},
                    },
                },
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api', mediaBaseUrl: '/media' } },
                { provide: ToastService, useValue: { success: () => {}, error: () => {} } },
                { provide: ScopeStore, useValue: { projectId } },
                { provide: OverlayStore, useValue: { workspace: signal(null), openModal, openWorkspace: () => {} } },
                { provide: SearchStore, useValue: { query: signal(''), fields: signal(new Set<string>()) } },
            ],
        });
    });

    it('adds and removes a dataset from the selection (keyed by id)', () => {
        const c = make();
        expect(c.isSelected(DS1)).toBe(false);
        c.toggleSelection(DS1);
        expect(c.isSelected(DS1)).toBe(true);
        expect(c.selectionCount()).toBe(1);
        c.toggleSelection(DS1);
        expect(c.isSelected(DS1)).toBe(false);
        expect(c.selectionCount()).toBe(0);
    });

    it('stops propagation so selecting a card does not open it', () => {
        const c = make();
        const e = ev();
        c.toggleSelection(DS1, e);
        expect((e.stopPropagation as Mock)).toHaveBeenCalled();
    });

    it('clearSelection empties the set', () => {
        const c = make();
        c.toggleSelection(DS1);
        c.toggleSelection(DS2);
        expect(c.selectionCount()).toBe(2);
        c.clearSelection();
        expect(c.selectionCount()).toBe(0);
    });

    it('toggleSelectAll selects every visible dataset, then clears when all selected', () => {
        const c = make();
        c.toggleSelectAll();
        expect(c.selectionCount()).toBe(3);
        expect(c.allVisibleSelected()).toBe(true);
        c.toggleSelectAll();
        expect(c.selectionCount()).toBe(0);
        expect(c.allVisibleSelected()).toBe(false);
    });

    it('selectedDatasets returns only the selected visible rows', () => {
        const c = make();
        c.toggleSelection(DS1);
        c.toggleSelection(DS3);
        expect(c.selectedDatasets().map(d => d.id).sort()).toEqual(['d1', 'd3']);
    });

    it('bulkDelete (global scope) opens a destructive confirm and deletes ONLY on onConfirm', () => {
        const c = make();
        c.toggleSelection(DS1);
        c.toggleSelection(DS2);
        c.bulkDelete();

        expect(openModal).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
        expect(lastModal().message).toMatch(/2/);
        expect(deleteDataset).not.toHaveBeenCalled();

        lastModal().onConfirm!();
        expect(deleteDataset).toHaveBeenCalledTimes(2);
        expect(deleteDataset).toHaveBeenCalledWith('ds-1', false);
        expect(deleteDataset).toHaveBeenCalledWith('ds-2', false);
    });

    it('bulkDelete clears the selection after confirming', () => {
        const c = make();
        c.toggleSelection(DS1);
        c.bulkDelete();
        lastModal().onConfirm!();
        expect(c.selectionCount()).toBe(0);
    });

    it('bulkDelete (project scope) removes each selected dataset from the project on confirm', () => {
        projectId.set('p1');
        const c = make();
        c.toggleSelection(DS1);
        c.toggleSelection(DS2);
        c.bulkDelete();

        expect(openModal).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
        expect(removeProjectDataset).not.toHaveBeenCalled();
        lastModal().onConfirm!();
        expect(removeProjectDataset).toHaveBeenCalledTimes(2);
        expect(removeProjectDataset).toHaveBeenCalledWith('p1', 'd1');
        expect(removeProjectDataset).toHaveBeenCalledWith('p1', 'd2');
    });

    it('bulkRescan launches an incremental rescan for each selected dataset', () => {
        const c = make();
        c.toggleSelection(DS1);
        c.toggleSelection(DS3);
        c.bulkRescan();
        expect(rescanDataset).toHaveBeenCalledTimes(2);
        expect(rescanDataset).toHaveBeenCalledWith('ds-1', 'safe');
        expect(rescanDataset).toHaveBeenCalledWith('ds-3', 'safe');
    });

    it('bulkAddToProject adds every selected dataset to the chosen project', () => {
        const c = make();
        c.toggleSelection(DS1);
        c.toggleSelection(DS2);
        c.bulkAddToProject('p1', 'Proj One');
        expect(addProjectDataset).toHaveBeenCalledTimes(2);
        expect(addProjectDataset).toHaveBeenCalledWith('p1', 'd1');
        expect(addProjectDataset).toHaveBeenCalledWith('p1', 'd2');
    });
});
