/**
 * Pair-role chooser — behavioural specs.
 *
 * Fires when images are dropped onto an EDIT dataset (the role is ambiguous:
 * target "after" vs control "before"). Covers:
 *   - Target choice → delegate the whole batch to uploadTargets, then close.
 *   - Control choice → uploadControls(slot); all matched ⇒ close; some
 *     unmatched ⇒ switch into the manual-assignment tray (do NOT close).
 *   - Manual assign → uploadControlFile with the picked stem; clears the tray
 *     entry; closes once the tray empties.
 *   - The slot selector flows through to both control paths.
 */
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { signal } from '@angular/core';
import { PairRoleChooserModalComponent } from './pair-role-chooser.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService } from '../../services/dataset';
import { DatasetUploadService } from '../../services/dataset-upload.service';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { MediaItemStore } from '../../state/media-item.store';
import { ToastService } from '../../services/toast';

function file(name: string): File {
    return new File([''], name);
}

class StubOverlay {
    private _modal = signal<{ kind: string; data: unknown } | null>(null);
    topModal = this._modal;
    closeModal = vi.fn();
    seed(datasetName: string, files: File[]): void {
        this._modal.set({ kind: 'pair-role-chooser', data: { datasetName, files } });
    }
}

function bed(files: File[], targets: string[] = [], uploadControlsResult?: { matched: string[]; unmatched: File[] }) {
    const uploadTargets = vi.fn();
    const uploadControls = vi.fn().mockResolvedValue(
        uploadControlsResult ?? { matched: [], unmatched: [] },
    );
    const uploadControlFile = vi.fn(() => of({ filename: 'x', status: 'uploaded' }));
    const refreshDataset = vi.fn().mockResolvedValue(undefined);

    TestBed.configureTestingModule({
        providers: [
            { provide: OverlayStore, useClass: StubOverlay },
            { provide: DatasetUploadService, useValue: { uploadTargets, uploadControls } },
            { provide: DatasetService, useValue: { uploadControlFile } },
            { provide: DatasetSyncService, useValue: { refreshDataset } },
            {
                provide: MediaItemStore,
                useValue: {
                    byDataset: (_n: string) =>
                        signal(targets.map(s => ({ media_file: `${s}.png` }))),
                },
            },
            { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn() } },
        ],
    });
    const overlay = TestBed.inject(OverlayStore) as unknown as StubOverlay;
    overlay.seed('ds', files);
    const cmp = TestBed.runInInjectionContext(() => new PairRoleChooserModalComponent());
    return {
        cmp: cmp as any,
        overlay,
        uploadTargets,
        uploadControls,
        uploadControlFile,
        refreshDataset,
    };
}

describe('PairRoleChooserModalComponent', () => {
    it('chooseTarget delegates the batch to uploadTargets and closes', () => {
        const files = [file('a.jpg'), file('b.png')];
        const t = bed(files);
        t.cmp.chooseTarget();
        expect(t.uploadTargets).toHaveBeenCalledWith('ds', files);
        expect(t.overlay.closeModal).toHaveBeenCalled();
    });

    it('chooseControl uploads to the selected slot and closes when all match', async () => {
        const files = [file('cat01.jpg')];
        const t = bed(files, ['cat01'], { matched: ['cat01'], unmatched: [] });
        t.cmp.slot.set(2);
        await t.cmp.chooseControl();
        expect(t.uploadControls).toHaveBeenCalledWith('ds', files, 2);
        expect(t.overlay.closeModal).toHaveBeenCalled();
    });

    it('chooseControl opens the manual tray (no close) when some files are unmatched', async () => {
        const stray = file('mystery.jpg');
        const t = bed([file('cat01.jpg'), stray], ['cat01'], {
            matched: ['cat01'], unmatched: [stray],
        });
        await t.cmp.chooseControl();
        expect(t.overlay.closeModal).not.toHaveBeenCalled();
        expect(t.cmp.unmatched()).toEqual([stray]);
        expect(t.cmp.mode()).toBe('assign');
    });

    it('assign uploads the tray file with the picked target stem and clears it', async () => {
        const stray = file('mystery.jpg');
        const t = bed([stray], ['cat01', 'cat02'], { matched: [], unmatched: [stray] });
        await t.cmp.chooseControl();
        await t.cmp.assign(stray, 'cat02');
        expect(t.uploadControlFile).toHaveBeenCalledWith('ds', stray, 1, 'cat02');
        expect(t.cmp.unmatched()).toEqual([]);
    });

    it('closes after the last tray file is assigned', async () => {
        const stray = file('mystery.jpg');
        const t = bed([stray], ['cat01'], { matched: [], unmatched: [stray] });
        await t.cmp.chooseControl();
        await t.cmp.assign(stray, 'cat01');
        expect(t.overlay.closeModal).toHaveBeenCalled();
    });

    it('exposes target stems for the assignment dropdown', () => {
        const t = bed([file('x.jpg')], ['cat02', 'cat01']);
        expect(t.cmp.targetStems()).toEqual(['cat01', 'cat02']);
    });
});
