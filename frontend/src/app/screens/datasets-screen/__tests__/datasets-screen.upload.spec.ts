import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
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
 * Card drop-upload contract.
 *
 * Regression guard for the UAT bug where dropping N images + N caption files
 * showed "2N images / 0 captioned / blank preview" until a manual navigation:
 *   1. Caption `.txt` files must NOT inflate the image count — they go to
 *      `caption_count` via the classified optimistic update.
 *   2. The first uploaded image seeds an optimistic card preview.
 *   3. The follow-up scan is the BACKGROUNDED rescan (Task Center progress),
 *      not the old blocking synchronous `/scan` — completion refreshes the
 *      card via DatasetStore's `dataset.invalidated` handler.
 *
 * The component class is instantiated directly (no template render); uploads
 * resolve synchronously via `of(...)`, so the optimistic apply + rescan launch
 * have already run by the time we assert.
 */
describe('DatasetsScreen — card drop-upload', () => {
    let applyOptimisticUpload: Mock;
    let uploadFile: Mock;
    let rescanDataset: Mock;
    let scanDataset: Mock;

    function fileList(...names: string[]): FileList {
        // The handler only uses `length` + `Array.from(files)`, so an array of
        // File objects stands in for a real FileList.
        return names.map(n => new File([''], n)) as unknown as FileList;
    }

    function makeScreen(): DatasetsScreen {
        applyOptimisticUpload = vi.fn();
        uploadFile = vi.fn((_name: string, file: File) => of({ filename: file.name, status: 'uploaded' }));
        rescanDataset = vi.fn().mockReturnValue(of({ task_id: 't1' }));
        scanDataset = vi.fn().mockReturnValue(of({}));

        TestBed.configureTestingModule({
            providers: [
                {
                    provide: DatasetStore,
                    useValue: {
                        loadAll: () => Promise.resolve(),
                        entities: signal([{ id: 'd1', name: 'ds-1' }]),
                        applyOptimisticUpload,
                    },
                },
                {
                    provide: DatasetService,
                    useValue: {
                        getCacheStats: () => of(null),
                        getMpxDistribution: () => of(null),
                        uploadFile,
                        rescanDataset,
                        scanDataset,
                    },
                },
                { provide: ProjectService, useValue: { getProjectDatasets: vi.fn().mockReturnValue(of([])) } },
                { provide: RuntimeConfigService, useValue: { mediaBaseUrl: '' } },
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn() } },
                { provide: ScopeStore, useValue: { projectId: signal<string | null>(null) } },
                { provide: OverlayStore, useValue: { workspace: signal(null) } },
                { provide: SearchStore, useValue: { query: signal(''), fields: signal(new Set<string>()) } },
            ],
        });
        return TestBed.runInInjectionContext(() => new DatasetsScreen());
    }

    it('classifies captions vs images and seeds a preview from the first image', () => {
        const comp = makeScreen() as any;
        comp.onUploadFiles('ds-1', fileList('cat.jpg', 'cat.txt', 'dog.png', 'dog.caption'));
        expect(uploadFile).toHaveBeenCalledTimes(4);
        expect(applyOptimisticUpload).toHaveBeenCalledTimes(1);
        // 2 images (cat.jpg, dog.png), 2 captions (.txt, .caption); preview = first image.
        expect(applyOptimisticUpload).toHaveBeenCalledWith('ds-1', { media: 2, caption: 2 }, 'cat.jpg');
    });

    it('launches a backgrounded rescan, not the blocking synchronous scan', () => {
        const comp = makeScreen() as any;
        comp.onUploadFiles('ds-1', fileList('a.jpg'));
        expect(rescanDataset).toHaveBeenCalledWith('ds-1', 'safe');
        expect(scanDataset).not.toHaveBeenCalled();
    });

    it('does nothing for an empty drop', () => {
        const comp = makeScreen() as any;
        comp.onUploadFiles('ds-1', fileList());
        expect(uploadFile).not.toHaveBeenCalled();
        expect(applyOptimisticUpload).not.toHaveBeenCalled();
        expect(rescanDataset).not.toHaveBeenCalled();
    });
});
