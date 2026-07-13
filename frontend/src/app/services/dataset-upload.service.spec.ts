import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { DatasetUploadService } from './dataset-upload.service';
import { DatasetService } from './dataset';
import { DatasetStore } from '../state/dataset.store';
import { DatasetSyncService } from '../state/dataset-sync.service';
import { MediaItemStore } from '../state/media-item.store';
import { ToastService } from './toast';

/**
 * The single upload authority used by the datasets-screen cards, the
 * in-workspace grid drop zone, and the pair-role-chooser / Pairs manager.
 *
 * - `uploadTargets` reproduces the card-drop contract: caption files don't
 *   inflate the image count, the first image seeds an optimistic preview, and
 *   the follow-up is the BACKGROUNDED safe rescan (not a blocking scan).
 * - `uploadControls` auto-matches each dropped file's stem against existing
 *   target stems (case-insensitive, but uploads the REAL target stem so the
 *   backend's case-sensitive pairing holds), and returns the leftovers for the
 *   caller's manual-assignment tray.
 */
describe('DatasetUploadService', () => {
    let uploadFile: Mock;
    let uploadControlFile: Mock;
    let rescanDataset: Mock;
    let applyOptimisticUpload: Mock;
    let refreshDataset: Mock;

    function file(name: string): File {
        return new File([''], name);
    }

    function setup(targets: string[] = []): DatasetUploadService {
        uploadFile = vi.fn((_n: string, f: File) => of({ filename: f.name, status: 'uploaded' }));
        uploadControlFile = vi.fn((_n: string, f: File) => of({ filename: f.name, status: 'uploaded' }));
        rescanDataset = vi.fn().mockReturnValue(of({ task_id: 't1' }));
        applyOptimisticUpload = vi.fn();
        refreshDataset = vi.fn().mockResolvedValue(undefined);

        TestBed.configureTestingModule({
            providers: [
                DatasetUploadService,
                {
                    provide: DatasetService,
                    useValue: { uploadFile, uploadControlFile, rescanDataset },
                },
                { provide: DatasetStore, useValue: { applyOptimisticUpload } },
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
        return TestBed.inject(DatasetUploadService);
    }

    describe('uploadTargets', () => {
        it('classifies captions vs images and seeds a preview from the first image', () => {
            const svc = setup();
            svc.uploadTargets('ds-1', [file('cat.jpg'), file('cat.txt'), file('dog.png')]);
            expect(uploadFile).toHaveBeenCalledTimes(3);
            expect(applyOptimisticUpload).toHaveBeenCalledWith(
                'ds-1', { media: 2, caption: 1 }, 'cat.jpg',
            );
        });

        it('launches a backgrounded safe rescan', () => {
            const svc = setup();
            svc.uploadTargets('ds-1', [file('a.jpg')]);
            expect(rescanDataset).toHaveBeenCalledWith('ds-1', 'safe');
        });

        it('does nothing for an empty list', () => {
            const svc = setup();
            svc.uploadTargets('ds-1', []);
            expect(uploadFile).not.toHaveBeenCalled();
            expect(applyOptimisticUpload).not.toHaveBeenCalled();
        });

        it('accepts audio extensions as media (no client-side extension gate, C0)', () => {
            const svc = setup();
            svc.uploadTargets('ds-1', [
                file('song.wav'), file('song.mp3'), file('song.flac'), file('song.ogg'), file('song.opus'),
            ]);
            expect(uploadFile).toHaveBeenCalledTimes(5);
            // Audio isn't IMAGE_EXTS, so none of them seed the optimistic
            // preview — media count still includes all five.
            expect(applyOptimisticUpload).toHaveBeenCalledWith(
                'ds-1', { media: 5, caption: 0 }, undefined,
            );
        });

        it('does not misclassify a lyrics sidecar as a caption (media count)', () => {
            const svc = setup();
            svc.uploadTargets('ds-1', [file('song.wav'), file('song.lyrics.txt')]);
            // song.lyrics.txt still ends in .txt, so the client-side classifier
            // (which mirrors the backend's naive splitext-based CAPTION_EXTS
            // check) counts it as a caption too — this documents that shared
            // behavior rather than asserting a stricter client-side split the
            // backend scan itself doesn't perform pre-rescan.
            expect(uploadFile).toHaveBeenCalledTimes(2);
            expect(applyOptimisticUpload).toHaveBeenCalledWith(
                'ds-1', { media: 1, caption: 1 }, undefined,
            );
        });
    });

    describe('uploadControls', () => {
        it('auto-matches by filename stem and uploads matched files into the slot', async () => {
            const svc = setup(['cat01', 'cat02']);
            const res = await svc.uploadControls(
                'ds-1', [file('cat01.jpg'), file('mystery.png')], 1,
            );
            expect(uploadControlFile).toHaveBeenCalledTimes(1);
            expect(uploadControlFile).toHaveBeenCalledWith(
                'ds-1', expect.any(File), 1, 'cat01',
            );
            expect(res.matched).toEqual(['cat01']);
            expect(res.unmatched.map((f: File) => f.name)).toEqual(['mystery.png']);
        });

        it('matches case-insensitively but uploads the real target stem', async () => {
            const svc = setup(['Cat01']);
            const res = await svc.uploadControls('ds-1', [file('CAT01.jpg')], 1);
            expect(uploadControlFile).toHaveBeenCalledWith(
                'ds-1', expect.any(File), 1, 'Cat01',
            );
            expect(res.matched).toEqual(['Cat01']);
        });

        it('refreshes the dataset after matched uploads', async () => {
            const svc = setup(['cat01']);
            await svc.uploadControls('ds-1', [file('cat01.jpg')], 2);
            expect(refreshDataset).toHaveBeenCalledWith('ds-1');
        });

        it('skips upload + refresh and returns leftovers when nothing matches', async () => {
            const svc = setup(['cat01']);
            const res = await svc.uploadControls('ds-1', [file('nope.jpg')], 1);
            expect(uploadControlFile).not.toHaveBeenCalled();
            expect(refreshDataset).not.toHaveBeenCalled();
            expect(res.matched).toEqual([]);
            expect(res.unmatched.length).toBe(1);
        });
    });
});
