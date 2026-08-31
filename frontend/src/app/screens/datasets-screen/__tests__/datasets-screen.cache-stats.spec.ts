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
 * Cold cache-stats handling (LANE-52).
 *
 * The backend no longer holds `GET /datasets/cache/stats` open while it walks
 * the library — one such request was measured at 479.81 s on the live server,
 * with the Datasets screen waiting behind it. It answers immediately with
 * `ready: false` and placeholder ZEROS, which puts the burden here: rendering
 * those zeros would replace an honest "calculating…" with a confident "0 B" for
 * a library holding 85 GB.
 */
describe('DatasetsScreen — cold cache stats', () => {
    let entities: WritableSignal<Array<Record<string, unknown>>>;
    let responses: Array<Record<string, unknown> | null>;
    let calls: number;

    const warm = (bytes: number) => ({
        total_bytes: bytes, latent_bytes: 0, embedding_bytes: 0,
        cached_datasets: 1, dataset_root_bytes: bytes, ready: true,
    });
    const cold = () => ({
        total_bytes: 0, latent_bytes: 0, embedding_bytes: 0,
        cached_datasets: 0, dataset_root_bytes: 0, ready: false,
    });

    function stats(comp: DatasetsScreen): Record<string, unknown> | null {
        return (comp as unknown as { cacheStats: () => Record<string, unknown> | null }).cacheStats();
    }

    function make(): DatasetsScreen {
        return TestBed.runInInjectionContext(() => new DatasetsScreen());
    }

    beforeEach(() => {
        vi.useFakeTimers();
        entities = signal<Array<Record<string, unknown>>>([]);
        responses = [];
        calls = 0;

        TestBed.configureTestingModule({
            providers: [
                {
                    provide: DatasetStore,
                    useValue: { loadAll: () => Promise.resolve(), entities, loading: signal(false) },
                },
                {
                    provide: DatasetService,
                    useValue: {
                        getCacheStats: () => {
                            const next = responses[Math.min(calls, responses.length - 1)];
                            calls += 1;
                            return of(next);
                        },
                        getLegacyThumbnailSurvey: () => of({ datasets: [], dataset_count: 0, total_files: 0, total_bytes: 0 }),
                        getMpxDistribution: () => of(null),
                    },
                },
                { provide: ProjectService, useValue: { getProjectDatasets: () => of([]) } },
                { provide: RuntimeConfigService, useValue: { mediaBaseUrl: '' } },
                { provide: ToastService, useValue: { success: () => {}, error: () => {} } },
                { provide: ScopeStore, useValue: { projectId: signal<string | null>(null) } },
                { provide: OverlayStore, useValue: { workspace: signal(null), openModal: () => {} } },
                { provide: SearchStore, useValue: { query: signal(''), fields: signal(new Set<string>()) } },
            ],
        });
    });

    afterEach(() => { vi.useRealTimers(); });

    it('never stores a not-ready payload as if it were a measurement', () => {
        responses = [cold()];
        const comp = make();
        expect(stats(comp)).toBeNull();
    });

    it('picks the figures up once the sweep finishes', () => {
        responses = [cold(), warm(4096)];
        const comp = make();
        expect(stats(comp)).toBeNull();

        vi.advanceTimersByTime(5000);

        expect(stats(comp)).not.toBeNull();
        expect(stats(comp)!['dataset_root_bytes']).toBe(4096);
    });

    it('stops re-polling instead of asking forever', () => {
        responses = [cold()];
        make();
        vi.advanceTimersByTime(10 * 60 * 1000);
        // 12 attempts total (ARCHITECTURE D10 — every wait bounded); ten minutes
        // of virtual time must not produce a 121st request.
        expect(calls).toBe(12);
    });

    it('asks only once when the first answer is already ready', () => {
        responses = [warm(10)];
        const comp = make();
        vi.advanceTimersByTime(60_000);
        expect(calls).toBe(1);
        expect(stats(comp)!['dataset_root_bytes']).toBe(10);
    });
});
