import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { DatasetWorkspaceComponent } from '../dataset-workspace.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetStore } from '../../state/dataset.store';
import { MediaItemStore } from '../../state/media-item.store';
import { CaptionCacheStore } from '../../state/caption-cache.store';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { ScopeStore } from '../../state/scope.store';
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';
import { RuntimeConfigService } from '../../services/runtime-config.service';

/**
 * LANE-58 — the workspace's "first `/pairs` in flight" state.
 *
 * The browse body gates on `pairsPending()`: skeleton while the FIRST fetch
 * of a dataset is outstanding and the store holds nothing, the real grid
 * otherwise. Three facts are pinned here, each on the signal the template
 * reads (the same stub bed the main workspace spec uses — the component is
 * injected headlessly, so the observable is the gate, not the DOM):
 *
 *   1. pending while the fetch is outstanding and the store is empty;
 *   2. released the moment the fetch settles — on failure too;
 *   3. NEVER pending for a dataset that already has rows, however many
 *      refreshes are in flight (every mutation funnels through
 *      `refreshDataset`; blanking a populated grid would be the worse bug).
 */

class StubOverlay {
    workspace = signal({ datasetId: 'd1', imageIndex: 0, mode: 'browse' as const });
    modalStack = signal<any[]>([]);
    topModal = signal<any>(null);
    closeWorkspace = vi.fn();
    setWorkspaceMode = vi.fn();
    setWorkspaceImage = vi.fn();
    openModal = vi.fn();
}
class StubDatasetStore {
    entities = signal<any[]>([{ id: 'd1', name: 'alpha', version: '1.0.0', multimedia_count: 33 }]);
    byId = (_: string) => signal<any>(this.entities()[0]);
    loadAll = vi.fn().mockReturnValue(Promise.resolve());
    upsertLocal = vi.fn();
}
class StubMediaItems {
    rows = signal<any[]>([]);
    byDataset = (_: string) => this.rows;
    mediaRev = signal(0);
    bumpMedia = vi.fn();
    saveCaption = vi.fn().mockReturnValue(Promise.resolve({ ok: true, value: {} }));
}
class StubCaptionCache {
    byDataset = signal<Record<string, Map<string, any>>>({});
    get = (_: string) => new Map<string, any>();
    seed = vi.fn();
    setCaption = vi.fn();
    setRow = vi.fn();
    remove = vi.fn();
}
class StubScope { projectId = signal<string | null>(null); }
class StubDatasetService {
    getDatasetPairs = vi.fn().mockReturnValue(of([]));
    getDataset = vi.fn().mockReturnValue(of(null));
    thumbnailUrl = vi.fn().mockReturnValue('/thumb');
    bumpVersion = vi.fn().mockReturnValue(of({ version: '2.0.0' }));
    saveCaption = vi.fn();
    saveCaptionVariant = vi.fn().mockReturnValue(of({ ok: true }));
    getCaptionVariantMap = vi.fn().mockReturnValue(of({ variants: {} }));
}
class StubToast { success = vi.fn(); error = vi.fn(); }
class StubRtc { apiUrl = '/api'; mediaBaseUrl = '/media'; }

/** A refreshDataset the test settles by hand. */
function deferredSync() {
    let resolve!: () => void;
    let reject!: (e: unknown) => void;
    const promise = new Promise<void>((res, rej) => { resolve = res; reject = rej; });
    return { refreshDataset: vi.fn().mockReturnValue(promise), resolve, reject };
}

function bed(sync: { refreshDataset: unknown }, media = new StubMediaItems()) {
    TestBed.configureTestingModule({
        providers: [
            DatasetWorkspaceComponent,
            { provide: OverlayStore, useClass: StubOverlay },
            { provide: DatasetStore, useClass: StubDatasetStore },
            { provide: MediaItemStore, useValue: media },
            { provide: CaptionCacheStore, useClass: StubCaptionCache },
            { provide: DatasetSyncService, useValue: sync },
            { provide: ScopeStore, useClass: StubScope },
            { provide: DatasetService, useClass: StubDatasetService },
            { provide: ToastService, useClass: StubToast },
            { provide: RuntimeConfigService, useClass: StubRtc },
        ],
    });
    const cmp = TestBed.inject(DatasetWorkspaceComponent) as any;
    TestBed.tick();   // run the constructor effects → ensurePairsLoaded('alpha')
    return cmp;
}

const flush = () => new Promise<void>(r => setTimeout(r, 0));

describe('DatasetWorkspaceComponent — first /pairs in flight (LANE-58)', () => {
    afterEach(() => TestBed.resetTestingModule());

    it('is pending while the first fetch is outstanding and the store is empty', async () => {
        const sync = deferredSync();
        const cmp = bed(sync);
        expect(sync.refreshDataset).toHaveBeenCalledWith('alpha');
        expect(cmp.pairsPending()).toBe(true);
        expect(cmp.skeletonSlots()).toBe(Math.min(33, cmp.density() * 4));

        sync.resolve();
        await flush();
        expect(cmp.pairsPending()).toBe(false);
    });

    it('releases the pending state when the fetch fails, not only when it succeeds', async () => {
        const sync = deferredSync();
        const cmp = bed(sync);
        expect(cmp.pairsPending()).toBe(true);

        sync.reject(new Error('offline'));
        await flush().catch(() => undefined);
        await flush();
        expect(cmp.pairsPending()).toBe(false);
    });

    it('is never pending for a dataset that already has rows, even mid-refresh', () => {
        const sync = deferredSync();
        const media = new StubMediaItems();
        media.rows.set([{ media_file: 'a.png', dataset_name: 'alpha', media_type: 'image', metadata: null }]);
        const cmp = bed(sync, media);
        expect(sync.refreshDataset).toHaveBeenCalledWith('alpha');   // fetch IS in flight
        expect(cmp.pairs().length).toBe(1);
        expect(cmp.pairsPending()).toBe(false);                      // …but no skeleton
    });

    it('falls back to one row of placeholders when the row has no media count', () => {
        const sync = deferredSync();
        const cmp = bed(sync);
        const store = TestBed.inject(DatasetStore) as unknown as StubDatasetStore;
        store.entities.set([{ id: 'd1', name: 'alpha', version: '1.0.0' }]);
        expect(cmp.skeletonSlots()).toBe(cmp.density());
    });
});
