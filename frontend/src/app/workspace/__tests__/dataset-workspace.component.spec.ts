import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of, throwError } from 'rxjs';
import { DatasetWorkspaceComponent } from '../dataset-workspace.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetStore } from '../../state/dataset.store';
import { MediaItemStore } from '../../state/media-item.store';
import { CaptionCacheStore } from '../../state/caption-cache.store';
import { ScopeStore } from '../../state/scope.store';
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';
import { RuntimeConfigService } from '../../services/runtime-config.service';

// ---------------------------------------------------------------------------
// Stub dependencies
// ---------------------------------------------------------------------------

class StubOverlay {
    workspace = signal({ datasetId: 'd1', imageIndex: 0, mode: 'browse' as const });
    modalStack = signal<any[]>([]);
    topModal = signal<any>(null);
    closeWorkspace = jasmine.createSpy('closeWorkspace');
    setWorkspaceMode = jasmine.createSpy('setWorkspaceMode');
    setWorkspaceImage = jasmine.createSpy('setWorkspaceImage');
    openModal = jasmine.createSpy('openModal');
}

/**
 * `dataset()` in the component resolves from `datasets.entities()` first.
 * Seed `entities` with a matching row so `dataset()` returns non-null and
 * `pairs` computed doesn't short-circuit to `[]`.
 */
class StubDatasetStore {
    byId = (_: string) => signal<any>({ id: 'd1', name: 'alpha', version: '1.0.0' });
    entities = signal<any[]>([{ id: 'd1', name: 'alpha', version: '1.0.0' }]);
    loadAll = jasmine.createSpy('loadAll').and.returnValue(Promise.resolve());
}

/**
 * `byDataset` returns a fresh `signal([])` per call by default.
 * Tests replace the function to return a signal with known items so the
 * `pairs` computed sees them through the real reactive chain — no spyOn
 * on the computed itself.
 */
class StubMediaItems {
    byDataset = (_: string) => signal<any[]>([]);
    loadForDataset = jasmine.createSpy('loadForDataset').and.returnValue(Promise.resolve());
    mediaRev = signal(0);
    bumpMedia = jasmine.createSpy('bumpMedia');
}

/**
 * `CaptionCacheStore.byDataset` is a plain `Signal<Record<string, Map<string, CaptionRow>>>`.
 * The component reads it as `this.captions.byDataset()[datasetName]`.
 * Default: no captions for any dataset.
 */
class StubCaptionCache {
    byDataset = signal<Record<string, Map<string, any>>>({});
    get = (_: string) => new Map<string, any>();
    seed = jasmine.createSpy('seed');
}

class StubScope { projectId = signal<string | null>(null); }
class StubDatasetService {
    getDatasetPairs = jasmine.createSpy().and.returnValue({ subscribe: () => {} });
    getDataset = jasmine.createSpy('getDataset').and.returnValue({ subscribe: () => {} });
    thumbnailUrl = jasmine.createSpy('thumbnailUrl').and.returnValue('/thumb');
    bumpVersion = jasmine.createSpy('bumpVersion').and.returnValue({
        subscribe: (cb: any) => cb({ version: '2.0.0' }),
    });
    saveCaption = jasmine.createSpy('saveCaption');
}
class StubToast { success = jasmine.createSpy(); error = jasmine.createSpy(); }
class StubRtc { apiUrl = '/api'; mediaBaseUrl = '/media'; }

// ---------------------------------------------------------------------------
// Test bed factory — called once per `it` block (TestBed is reset between
// describes automatically by Jasmine's Angular integration).
// ---------------------------------------------------------------------------

function bed(): DatasetWorkspaceComponent {
    TestBed.configureTestingModule({
        providers: [
            DatasetWorkspaceComponent,
            { provide: OverlayStore, useClass: StubOverlay },
            { provide: DatasetStore, useClass: StubDatasetStore },
            { provide: MediaItemStore, useClass: StubMediaItems },
            { provide: CaptionCacheStore, useClass: StubCaptionCache },
            { provide: ScopeStore, useClass: StubScope },
            { provide: DatasetService, useClass: StubDatasetService },
            { provide: ToastService, useClass: StubToast },
            { provide: RuntimeConfigService, useClass: StubRtc },
        ],
    });
    return TestBed.inject(DatasetWorkspaceComponent);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Build a minimal MediaItem-shaped object for seeding `byDataset`.
 * `projectPair` destructures `{ id, dataset_name, media_file, stem,
 * media_type, caption_file, ...metadata }`, so `enabled`, `has_mask`,
 * and `quality_score` end up in `pair.metadata` automatically.
 */
function mediaItem(over: Record<string, any> = {}): any {
    return { media_file: 'i.jpg', ...over };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('DatasetWorkspaceComponent.filterCounts (Enabled / Excluded)', () => {

    it('exposes counts for the enabled/excluded filter tabs', () => {
        const cmp = bed();
        // Drive the underlying signal that `pairs` computed reads from.
        // Replace `byDataset` so every call returns a backing signal with
        // our known items — the `pairs` computed then executes its real
        // reactive chain through `projectPair`.
        const items = [
            mediaItem({ media_file: 'a.jpg', enabled: true,  has_mask: false, quality_score: 0.5 }),
            mediaItem({ media_file: 'b.jpg', enabled: true,  has_mask: false, quality_score: 0.5 }),
            mediaItem({ media_file: 'c.jpg', enabled: false, has_mask: false, quality_score: 0.5 }),
        ];
        (TestBed.inject(MediaItemStore) as any).byDataset = (_: string) => signal(items);

        const c = (cmp as any).filterCounts();
        expect(c.enabled).toBe(2);
        expect(c.excluded).toBe(1);
    });

    it('exposes existing counts (all / captioned / masked / lowHps) unchanged', () => {
        const cmp = bed();
        const items = [
            mediaItem({ media_file: 'a.jpg', enabled: true,  has_mask: true,  quality_score: 0.5  }),
            mediaItem({ media_file: 'b.jpg', enabled: true,  has_mask: false, quality_score: 0.20 }),
            mediaItem({ media_file: 'c.jpg', enabled: false, has_mask: false, quality_score: 0.5  }),
        ];
        (TestBed.inject(MediaItemStore) as any).byDataset = (_: string) => signal(items);

        // Caption content lives in CaptionCacheStore. `byDataset()` returns
        // `Record<datasetName, Map<mediaFile, CaptionRow>>`. Seed one
        // captioned entry for dataset "alpha" (the name used by StubDatasetStore).
        const captionMap = new Map([['a.jpg', { caption_content: 'hi' }]]);
        (TestBed.inject(CaptionCacheStore) as any).byDataset = signal({ alpha: captionMap });

        const c = (cmp as any).filterCounts();
        expect(c.all).toBe(3);
        expect(c.captioned).toBe(1);
        expect(c.masked).toBe(1);
        expect(c.lowHps).toBe(1);
    });

    it('visiblePairs filters to excluded when filter is "excluded"', () => {
        const cmp = bed();
        (TestBed.inject(MediaItemStore) as any).byDataset = (_: string) => signal([
            mediaItem({ media_file: 'a.jpg', enabled: true  }),
            mediaItem({ media_file: 'b.jpg', enabled: false }),
        ]);

        // `filter` is a writable signal on the component — set it directly.
        (cmp as any).filter.set('excluded');

        const visible = (cmp as any).visiblePairs();
        expect(visible.length).toBe(1);
        expect(visible[0].metadata.enabled).toBe(false);
    });

    it('pair with no enabled field counts as enabled (default-included)', () => {
        const cmp = bed();
        // No `enabled` key at all — `projectPair` spreads the MediaItem's
        // remaining fields into `metadata`, so `pair.metadata.enabled` is
        // `undefined`. The predicate `enabled !== false` is truthy for
        // `undefined`, so it counts as enabled and not as excluded.
        (TestBed.inject(MediaItemStore) as any).byDataset = (_: string) => signal([
            mediaItem({ media_file: 'a.jpg' }),
        ]);

        const c = (cmp as any).filterCounts();
        expect(c.enabled).toBe(1);
        expect(c.excluded).toBe(0);
    });

});

describe('DatasetWorkspaceComponent.bumpMajor', () => {
    it('confirms, calls bumpVersion(major), stamps version, toasts success', async () => {
        spyOn(window, 'confirm').and.returnValue(true);
        const cmp = bed();
        const api = TestBed.inject(DatasetService) as any;
        const datasets = TestBed.inject(DatasetStore) as any;
        const toast = TestBed.inject(ToastService) as any;

        api.bumpVersion = jasmine.createSpy('bumpVersion').and.returnValue(of({ version: '2.0.0' }));
        datasets.upsertLocal = jasmine.createSpy('upsertLocal');

        await (cmp as any).bumpMajor();

        expect(window.confirm).toHaveBeenCalled();
        expect(api.bumpVersion).toHaveBeenCalledWith('alpha', 'major');
        expect(datasets.upsertLocal).toHaveBeenCalledWith(
            jasmine.objectContaining({ name: 'alpha', version: '2.0.0' }),
        );
        expect(toast.success).toHaveBeenCalled();
    });

    it('does NOT call bumpVersion when confirm returns false', async () => {
        spyOn(window, 'confirm').and.returnValue(false);
        const cmp = bed();
        const api = TestBed.inject(DatasetService) as any;

        api.bumpVersion = jasmine.createSpy('bumpVersion').and.returnValue(of({ version: '2.0.0' }));

        await (cmp as any).bumpMajor();

        expect(api.bumpVersion).not.toHaveBeenCalled();
    });
});

describe('DatasetWorkspaceComponent.editVersion', () => {
    it('opens version-edit modal with current dataset name + version', () => {
        const cmp = bed();
        const overlay = TestBed.inject(OverlayStore) as any;

        (cmp as any).editVersion();

        expect(overlay.openModal).toHaveBeenCalledWith(
            'version-edit',
            jasmine.objectContaining({
                datasetName: 'alpha',
                currentVersion: '1.0.0',
                onSaved: jasmine.any(Function),
            }),
        );
    });

    it('onSaved callback upserts the dataset with the new version', () => {
        const cmp = bed();
        const overlay = TestBed.inject(OverlayStore) as any;
        const datasets = TestBed.inject(DatasetStore) as any;
        datasets.upsertLocal = jasmine.createSpy('upsertLocal');

        (cmp as any).editVersion();

        // Pull onSaved from the captured openModal payload (third tuple slot in args).
        const payload = (overlay.openModal as jasmine.Spy).calls.mostRecent().args[1];
        payload.onSaved('9.9.9');

        expect(datasets.upsertLocal).toHaveBeenCalledWith(
            jasmine.objectContaining({ name: 'alpha', version: '9.9.9' }),
        );
    });

    it('no-op when dataset signal is null', () => {
        const cmp = bed();
        const overlay = TestBed.inject(OverlayStore) as any;
        const datasets = TestBed.inject(DatasetStore) as any;
        // Empty entity store → `dataset()` resolves to null.
        datasets.entities = signal<any[]>([]);

        (cmp as any).editVersion();

        expect(overlay.openModal).not.toHaveBeenCalled();
    });
});

describe('DatasetWorkspaceComponent.ensurePatchBump (auto patch bump)', () => {
    it('first call fires bumpVersion(patch); subsequent calls no-op', () => {
        const cmp = bed();
        const api = TestBed.inject(DatasetService) as any;
        const datasets = TestBed.inject(DatasetStore) as any;
        api.bumpVersion = jasmine.createSpy('bumpVersion')
            .and.returnValue(of({ version: '1.0.1' }));
        datasets.upsertLocal = jasmine.createSpy('upsertLocal');

        // First call — should bump.
        (cmp as any).ensurePatchBump();
        // Second call — should be a no-op.
        (cmp as any).ensurePatchBump();

        const patchCalls = api.bumpVersion.calls.allArgs()
            .filter((args: any[]) => args[1] === 'patch');
        expect(patchCalls.length).toBe(1);
        expect(patchCalls[0]).toEqual(['alpha', 'patch']);
        expect(datasets.upsertLocal).toHaveBeenCalledTimes(1);
        expect(datasets.upsertLocal).toHaveBeenCalledWith(
            jasmine.objectContaining({ name: 'alpha', version: '1.0.1' }),
        );
    });

    it('on bump HTTP failure, flag clears so the next call retries', () => {
        const cmp = bed();
        const api = TestBed.inject(DatasetService) as any;
        // First call fails:
        let callCount = 0;
        api.bumpVersion = jasmine.createSpy('bumpVersion')
            .and.callFake(() => {
                callCount++;
                if (callCount === 1) return throwError(() => new Error('boom'));
                return of({ version: '1.0.1' });
            });
        spyOn(console, 'warn');

        // First call — HTTP fails synchronously through the observable.
        (cmp as any).ensurePatchBump();
        // Second call — should retry since the flag was reset.
        (cmp as any).ensurePatchBump();

        expect(api.bumpVersion).toHaveBeenCalledTimes(2);
    });

    it('mediaRev increment triggers ensurePatchBump via constructor effect', () => {
        const cmp = bed();
        const api = TestBed.inject(DatasetService) as any;
        const mediaItems = TestBed.inject(MediaItemStore) as any;
        api.bumpVersion = jasmine.createSpy('bumpVersion')
            .and.returnValue(of({ version: '1.0.1' }));

        // Baseline was captured at construction time (rev = 0).
        // Bumping to 1 should cross the baseline guard and fire ensurePatchBump.
        mediaItems.mediaRev.set(1);
        TestBed.tick();

        const patchCalls = api.bumpVersion.calls.allArgs()
            .filter((args: any[]) => args[1] === 'patch');
        expect(patchCalls.length).toBe(1);
        expect(patchCalls[0]).toEqual(['alpha', 'patch']);
    });

    it('second mediaRev increment within session does not re-bump', () => {
        const cmp = bed();
        const api = TestBed.inject(DatasetService) as any;
        const mediaItems = TestBed.inject(MediaItemStore) as any;
        api.bumpVersion = jasmine.createSpy('bumpVersion')
            .and.returnValue(of({ version: '1.0.1' }));

        // First increment — fires the bump and sets hasBumpedPatchInSession.
        mediaItems.mediaRev.set(1);
        TestBed.tick();
        // Second increment — effect fires again but ensurePatchBump no-ops.
        mediaItems.mediaRev.set(2);
        TestBed.tick();

        const patchCalls = api.bumpVersion.calls.allArgs()
            .filter((args: any[]) => args[1] === 'patch');
        expect(patchCalls.length).toBe(1);
    });
});
