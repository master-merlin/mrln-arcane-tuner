/**
 * Pair-health "Manage" tab — control upload + re-match behaviours.
 *
 * The modal is now a tabbed hub (Health | Manage). Manage covers three
 * actions, all funnelled so the grid stays in sync afterward:
 *   - Upload controls → DatasetUploadService.uploadControls(slot); matched
 *     files auto-pair, the rest drop into the browser-held "pending" tray.
 *   - Assign a pending (browser) file → uploadControlFile with the picked stem.
 *   - Re-match an on-disk orphan → DatasetService.assignControl, keeping it in
 *     its own slot, then reload the health report.
 * It also accepts a hand-off of pending files + the chosen slot from the
 * drop-time chooser via the modal data.
 */
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { signal } from '@angular/core';
import { PairHealthModalComponent } from './pair-health.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService, PairHealth } from '../../services/dataset';
import { DatasetUploadService } from '../../services/dataset-upload.service';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { MediaItemStore } from '../../state/media-item.store';
import { ToastService } from '../../services/toast';

function file(name: string): File {
    return new File([''], name);
}

function health(overrides: Partial<PairHealth> = {}): PairHealth {
    return {
        kind: 'edit',
        target_count: 2,
        paired_count: 1,
        fully_paired: false,
        active_slots: ['control'],
        missing_by_slot: { control: ['cat02'] },
        orphans: [],
        warnings: [],
        ...overrides,
    };
}

class StubOverlay {
    private _modal = signal<{ kind: string; data: unknown } | null>(null);
    topModal = this._modal;
    closeModal = vi.fn();
    openModal = vi.fn();
    seed(data: unknown): void {
        this._modal.set({ kind: 'pair-health', data });
    }
}

function bed(opts: {
    data?: Record<string, unknown>;
    health?: PairHealth;
    targets?: string[];
    uploadControlsResult?: { matched: string[]; unmatched: File[] };
}) {
    const getPairHealth = vi.fn().mockReturnValue(of(opts.health ?? health()));
    const uploadControlFile = vi.fn(() => of({ filename: 'x', status: 'uploaded' }));
    const assignControl = vi.fn(() => of({ rel_path: 'control/cat02.jpg', target_stem: 'cat02' }));
    const uploadControls = vi.fn().mockResolvedValue(
        opts.uploadControlsResult ?? { matched: [], unmatched: [] },
    );
    const refreshDataset = vi.fn().mockResolvedValue(undefined);

    TestBed.configureTestingModule({
        providers: [
            { provide: OverlayStore, useClass: StubOverlay },
            {
                provide: DatasetService,
                useValue: { getPairHealth, uploadControlFile, assignControl, deleteControlOrphans: vi.fn() },
            },
            { provide: DatasetUploadService, useValue: { uploadControls } },
            { provide: DatasetSyncService, useValue: { refreshDataset } },
            {
                provide: MediaItemStore,
                useValue: {
                    byDataset: (_n: string) =>
                        signal((opts.targets ?? ['cat01', 'cat02']).map(s => ({ media_file: `${s}.png` }))),
                    toggleEnabled: vi.fn(),
                },
            },
            { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn() } },
        ],
    });
    const overlay = TestBed.inject(OverlayStore) as unknown as StubOverlay;
    overlay.seed({ datasetName: 'editds', ...(opts.data ?? {}) });
    const cmp = TestBed.runInInjectionContext(() => new PairHealthModalComponent());
    return { cmp: cmp as any, overlay, getPairHealth, uploadControlFile, assignControl, uploadControls, refreshDataset };
}

describe('PairHealthModalComponent — Manage tab', () => {
    it('defaults to the Health tab', () => {
        const t = bed({});
        expect(t.cmp.tab()).toBe('health');
    });

    it('opens straight to Manage when the data requests it', () => {
        const t = bed({ data: { tab: 'manage' } });
        expect(t.cmp.tab()).toBe('manage');
    });

    it('seeds the pending tray + slot from a chooser hand-off', () => {
        const stray = file('mystery.jpg');
        const t = bed({ data: { tab: 'manage', slot: 2, pendingControls: [stray] } });
        expect(t.cmp.slot()).toBe(2);
        expect(t.cmp.pending()).toEqual([stray]);
    });

    it('uploads picked controls and queues the unmatched ones', async () => {
        const stray = file('mystery.jpg');
        const t = bed({
            uploadControlsResult: { matched: ['cat01'], unmatched: [stray] },
        });
        await t.cmp.onPickControls([file('cat01.jpg'), stray]);
        expect(t.uploadControls).toHaveBeenCalledWith('editds', [expect.any(File), stray], 1);
        expect(t.cmp.pending()).toEqual([stray]);
    });

    it('assigns a pending file with the picked target stem and clears it', async () => {
        const stray = file('mystery.jpg');
        const t = bed({ data: { tab: 'manage', pendingControls: [stray] } });
        await t.cmp.assignPending(stray, 'cat02');
        expect(t.uploadControlFile).toHaveBeenCalledWith('editds', stray, 1, 'cat02');
        expect(t.cmp.pending()).toEqual([]);
        expect(t.refreshDataset).toHaveBeenCalledWith('editds');
    });

    it('re-matches an on-disk orphan in its own slot via assignControl', async () => {
        const t = bed({
            data: { tab: 'manage' },
            health: health({ orphans: [{ slot: 'control_2', rel_path: 'control_2/ghost.jpg' }] }),
        });
        await t.cmp.assignOrphan({ slot: 'control_2', rel_path: 'control_2/ghost.jpg' }, 'cat02');
        expect(t.assignControl).toHaveBeenCalledWith('editds', 'control_2/ghost.jpg', 2, 'cat02');
        expect(t.refreshDataset).toHaveBeenCalledWith('editds');
        // Health reloaded after the re-match (constructor call + this one).
        expect(t.getPairHealth.mock.calls.length).toBeGreaterThanOrEqual(2);
    });

    it('exposes orphans from the health report for the re-match tray', async () => {
        const t = bed({
            health: health({ orphans: [{ slot: 'control', rel_path: 'control/ghost.jpg' }] }),
        });
        await Promise.resolve();   // settle the constructor reload()
        expect(t.cmp.orphans()).toEqual([{ slot: 'control', rel_path: 'control/ghost.jpg' }]);
    });

    it('exposes sorted target stems for the assignment dropdowns', () => {
        const t = bed({ targets: ['cat02', 'cat01'] });
        expect(t.cmp.targetStems()).toEqual(['cat01', 'cat02']);
    });
});
