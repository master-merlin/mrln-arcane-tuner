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
 * Library-card preview URL. Video-only datasets pick a clip as their
 * `preview_image`; an `<img>` can't paint mp4/webm/mkv/avi, so the card showed
 * a blank/ImageOff fallback. The clip must now route through the thumbnail
 * (poster) endpoint while stills keep their direct /media URL.
 */
describe('DatasetsScreen — preview URL routing', () => {
    function makeScreen(): any {
        TestBed.configureTestingModule({
            providers: [
                {
                    provide: DatasetStore,
                    useValue: { loadAll: () => Promise.resolve(), entities: signal([]) },
                },
                {
                    provide: DatasetService,
                    useValue: { getCacheStats: () => of(null), getMpxDistribution: () => of(null) },
                },
                { provide: DatasetUploadService, useValue: { uploadTargets: vi.fn() } },
                { provide: ProjectService, useValue: { getProjectDatasets: vi.fn().mockReturnValue(of([])) } },
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api', mediaBaseUrl: '/media' } },
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn() } },
                { provide: ScopeStore, useValue: { projectId: signal<string | null>(null) } },
                { provide: OverlayStore, useValue: { workspace: signal(null), openModal: vi.fn() } },
                { provide: SearchStore, useValue: { query: signal(''), fields: signal(new Set<string>()) } },
            ],
        });
        return TestBed.runInInjectionContext(() => new DatasetsScreen());
    }

    it('routes a video-only dataset cover through the thumbnail endpoint', () => {
        const comp = makeScreen();
        const url = comp.previewUrl({ name: 'Clips Set', preview_image: 'clip001.mp4' });
        expect(url).toBe('/api/datasets/Clips%20Set/thumbnail?image_rel_path=clip001.mp4');
    });

    it('serves an image dataset cover directly from /media (unchanged)', () => {
        const comp = makeScreen();
        const url = comp.previewUrl({ name: 'Photos', preview_image: 'cat.jpg' });
        expect(url).toBe('/media/Photos/cat.jpg');
    });

    it('returns null for a missing dataset or one with no preview', () => {
        const comp = makeScreen();
        expect(comp.previewUrl({ name: 'x', preview_image: 'clip.mp4', missing: true })).toBeNull();
        expect(comp.previewUrl({ name: 'x', preview_image: '' })).toBeNull();
    });
});
