import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { DatasetsScreen } from '../datasets-screen';
import { DatasetStore } from '../../../state/dataset.store';
import { DatasetService } from '../../../services/dataset';
import { DatasetUploadService } from '../../../services/dataset-upload.service';
import { ProjectService } from '../../../services/project.service';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { ToastService } from '../../../services/toast';
import { ScopeStore } from '../../../state/scope.store';
import { OverlayStore } from '../../../state/overlay.store';
import { SearchStore } from '../../../state/search.store';

/**
 * Card drop / picker routing.
 *
 * The screen no longer owns the upload mechanics (classified optimistic counts,
 * preview seeding, backgrounded rescan) — those moved to DatasetUploadService
 * and are covered by its own spec. The screen's job now is ROUTING:
 *   - standard dataset → delegate every file to `uploadTargets` (all targets).
 *   - edit (paired) dataset → a dropped image is ambiguous (target vs control),
 *     so open the pair-role-chooser instead of uploading blind.
 *
 * The component class is instantiated directly (no template render).
 */
describe('DatasetsScreen — card drop routing', () => {
    let uploadTargets: Mock;
    let openModal: Mock;

    function fileList(...names: string[]): FileList {
        return names.map(n => new File([''], n)) as unknown as FileList;
    }

    function makeScreen(datasets: Array<{ id: string; name: string; kind?: string }>): DatasetsScreen {
        uploadTargets = vi.fn();
        openModal = vi.fn();

        TestBed.configureTestingModule({
            providers: [
                {
                    provide: DatasetStore,
                    useValue: {
                        loadAll: () => Promise.resolve(),
                        entities: signal(datasets),
                        applyOptimisticUpload: vi.fn(),
                    },
                },
                {
                    provide: DatasetService,
                    useValue: {
                        getCacheStats: () => of(null),
                        getMpxDistribution: () => of(null),
                    },
                },
                { provide: DatasetUploadService, useValue: { uploadTargets } },
                { provide: ProjectService, useValue: { getProjectDatasets: vi.fn().mockReturnValue(of([])) } },
                { provide: RuntimeConfigService, useValue: { mediaBaseUrl: '' } },
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn() } },
                { provide: ScopeStore, useValue: { projectId: signal<string | null>(null) } },
                { provide: OverlayStore, useValue: { workspace: signal(null), openModal } },
                { provide: SearchStore, useValue: { query: signal(''), fields: signal(new Set<string>()) } },
            ],
        });
        return TestBed.runInInjectionContext(() => new DatasetsScreen());
    }

    it('delegates a standard-dataset drop straight to uploadTargets', () => {
        const comp = makeScreen([{ id: 'd1', name: 'ds-1' }]) as any;
        const files = fileList('cat.jpg', 'dog.png');
        comp.onUploadFiles('ds-1', files);
        expect(uploadTargets).toHaveBeenCalledWith('ds-1', files);
        expect(openModal).not.toHaveBeenCalled();
    });

    it('opens the pair-role-chooser for an edit dataset instead of uploading', () => {
        const comp = makeScreen([{ id: 'd1', name: 'edit-ds', kind: 'edit' }]) as any;
        comp.onUploadFiles('edit-ds', fileList('before.jpg'));
        expect(openModal).toHaveBeenCalledWith(
            'pair-role-chooser',
            expect.objectContaining({ datasetName: 'edit-ds' }),
        );
        expect(uploadTargets).not.toHaveBeenCalled();
    });

    it('hands the dropped files to the chooser', () => {
        const comp = makeScreen([{ id: 'd1', name: 'edit-ds', kind: 'edit' }]) as any;
        comp.onUploadFiles('edit-ds', fileList('a.jpg', 'b.jpg'));
        const data = openModal.mock.calls[0][1];
        expect(data.files.map((f: File) => f.name)).toEqual(['a.jpg', 'b.jpg']);
    });

    it('does nothing for an empty drop', () => {
        const comp = makeScreen([{ id: 'd1', name: 'ds-1' }]) as any;
        comp.onUploadFiles('ds-1', fileList());
        expect(uploadTargets).not.toHaveBeenCalled();
        expect(openModal).not.toHaveBeenCalled();
    });
});
