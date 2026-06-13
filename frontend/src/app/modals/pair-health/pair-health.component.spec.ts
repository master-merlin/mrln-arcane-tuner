/**
 * Pair-health modal — behavioural specs.
 *
 * Focus: health load on open, the unpaired-stems projection (slot-1
 * missing list), the disable-unpaired bulk action (store rows resolved
 * by lowercased stem), and orphan deletion (API + refresh funnel).
 */
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { signal } from '@angular/core';
import { PairHealthModalComponent } from './pair-health.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService, PairHealth } from '../../services/dataset';
import { MediaItemStore } from '../../state/media-item.store';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { ToastService } from '../../services/toast';

const HEALTH: PairHealth = {
    kind: 'edit',
    target_count: 3,
    paired_count: 1,
    fully_paired: false,
    active_slots: ['control'],
    missing_by_slot: { control: ['img2', 'img3'] },
    orphans: [{ slot: 'control', rel_path: 'control/ghost.jpg' }],
    warnings: [{ stem: 'img1', type: 'dim_mismatch' }],
};

class StubOverlay {
    private _modal = signal<{ kind: string; data: unknown } | null>({
        kind: 'pair-health',
        data: { datasetName: 'editds' },
    });
    topModal = this._modal;
    closeModal = vi.fn();
}

class StubDatasetService {
    getPairHealth = vi.fn().mockReturnValue(of(HEALTH));
    deleteControlOrphans = vi.fn().mockReturnValue(of({ deleted: 1 }));
}

class StubMediaItems {
    toggleEnabled = vi.fn().mockResolvedValue({ ok: true });
    byDataset = vi.fn().mockReturnValue(() => [
        { media_file: 'img1.png', enabled: true },
        { media_file: 'IMG2.png', enabled: true },     // stem matching is case-insensitive
        { media_file: 'img3.png', enabled: false },    // already disabled — skipped
    ]);
}

class StubSync { refreshDataset = vi.fn().mockResolvedValue(undefined); }
class StubToast { success = vi.fn(); error = vi.fn(); info = vi.fn(); }

function bed() {
    TestBed.configureTestingModule({
        providers: [
            { provide: OverlayStore, useClass: StubOverlay },
            { provide: DatasetService, useClass: StubDatasetService },
            { provide: MediaItemStore, useClass: StubMediaItems },
            { provide: DatasetSyncService, useClass: StubSync },
            { provide: ToastService, useClass: StubToast },
        ],
    });
    const cmp = TestBed.runInInjectionContext(() => new PairHealthModalComponent());
    return {
        cmp: cmp as any,
        api: TestBed.inject(DatasetService) as unknown as StubDatasetService,
        items: TestBed.inject(MediaItemStore) as unknown as StubMediaItems,
        sync: TestBed.inject(DatasetSyncService) as unknown as StubSync,
    };
}

describe('PairHealthModalComponent', () => {
    it('loads health on construction and projects unpaired stems', async () => {
        const { cmp, api } = bed();
        await Promise.resolve();   // settle the constructor reload()
        expect(api.getPairHealth).toHaveBeenCalledWith('editds');
        expect(cmp.health()).toEqual(HEALTH);
        expect(cmp.unpairedStems()).toEqual(['img2', 'img3']);
    });

    it('disableUnpaired toggles only enabled rows whose stem is unpaired', async () => {
        const { cmp, items } = bed();
        await Promise.resolve();

        await cmp.disableUnpaired();

        // IMG2.png matches stem 'img2' case-insensitively; img3 is already
        // disabled; img1 is paired — exactly one toggle.
        expect(items.toggleEnabled).toHaveBeenCalledTimes(1);
        expect(items.toggleEnabled).toHaveBeenCalledWith('editds', 'IMG2.png', false);
    });

    it('deleteOrphans calls the API, refreshes the dataset, and reloads health', async () => {
        const { cmp, api, sync } = bed();
        await Promise.resolve();
        api.getPairHealth.mockClear();

        await cmp.deleteOrphans();

        expect(api.deleteControlOrphans).toHaveBeenCalledWith('editds');
        expect(sync.refreshDataset).toHaveBeenCalledWith('editds');
        expect(api.getPairHealth).toHaveBeenCalledTimes(1);   // the reload
    });
});
