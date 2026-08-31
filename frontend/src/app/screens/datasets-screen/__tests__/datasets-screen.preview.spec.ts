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
import { PREVIEW_MAX_EDGE } from '../../../shared/media-preview';

/**
 * Library-card preview URL and its failure path.
 *
 * Every cover resolves to a bounded thumbnail rendition: video because an
 * `<img>` cannot paint mp4/webm/mkv/avi at all, stills because a full-size
 * training source decoded into a 260px card is what made the library
 * unscrollable. The `<img>` must still recover if a thumbnail is unavailable.
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
                    useValue: { getCacheStats: () => of(null), getLegacyThumbnailSurvey: () => of({ datasets: [], dataset_count: 0, total_files: 0, total_bytes: 0 }), getMpxDistribution: () => of(null) },
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
        expect(url).toBe(
            `/api/datasets/Clips%20Set/thumbnail?image_rel_path=clip001.mp4&max_edge=${PREVIEW_MAX_EDGE}`,
        );
    });

    it('routes a still cover through the thumbnail endpoint, never the original', () => {
        const comp = makeScreen();
        const url = comp.previewUrl({ name: 'Photos', preview_image: 'cat.jpg' });
        expect(url).toBe(
            `/api/datasets/Photos/thumbnail?image_rel_path=cat.jpg&max_edge=${PREVIEW_MAX_EDGE}`,
        );
        expect(url).not.toBe('/media/Photos/cat.jpg');
    });

    it('returns null for a missing dataset or one with no preview', () => {
        const comp = makeScreen();
        expect(comp.previewUrl({ name: 'x', preview_image: 'clip.mp4', missing: true })).toBeNull();
        expect(comp.previewUrl({ name: 'x', preview_image: '' })).toBeNull();
    });

    describe('cover failure path', () => {
        function failedImg(): HTMLImageElement {
            const img = document.createElement('img');
            img.src = '/api/datasets/Photos/thumbnail?image_rel_path=cat.jpg';
            return img;
        }

        it('retries once against /media when the thumbnail is unavailable', () => {
            // Thumbnail generation is Pillow-based; a format the browser can
            // paint but Pillow cannot decode (AVIF without its plugin) would
            // otherwise leave a hole where a perfectly good cover belongs.
            const comp = makeScreen();
            const img = failedImg();

            comp.onPreviewError({ target: img } as unknown as Event, {
                name: 'Photos', preview_image: 'cat.jpg',
            });

            expect(img.src).toContain('/media/Photos/cat.jpg');
            expect(img.style.display).not.toBe('none');
        });

        it('gives up after the retry rather than looping', () => {
            const comp = makeScreen();
            const img = failedImg();
            const d = { name: 'Photos', preview_image: 'cat.jpg' };

            comp.onPreviewError({ target: img } as unknown as Event, d);
            comp.onPreviewError({ target: img } as unknown as Event, d);

            expect(img.style.display).toBe('none');
        });

        it('hides immediately when there is no dataset to fall back to', () => {
            const comp = makeScreen();
            const img = failedImg();

            comp.onPreviewError({ target: img } as unknown as Event, undefined);

            expect(img.style.display).toBe('none');
        });
    });
});
