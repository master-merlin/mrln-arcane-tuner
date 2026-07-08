/**
 * CacheModalComponent — purge-confirm contract spec.
 *
 * purge() wipes cached latents/embeddings from disk and cannot be undone, so it
 * must route through the themed confirm modal: purgeCache only fires from the
 * modal's onConfirm callback, never synchronously.
 */
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { CacheModalComponent } from './cache.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';

describe('CacheModalComponent — purge confirm contract', () => {
    let api: { listCache: ReturnType<typeof vi.fn>; purgeCache: ReturnType<typeof vi.fn> };

    beforeEach(() => {
        api = {
            listCache: vi.fn().mockReturnValue(of({ cache: {} })),
            purgeCache: vi.fn().mockReturnValue(of({ deleted: 3, freed_bytes: 1024 })),
        };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore,
                { provide: DatasetService, useValue: api },
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn() } },
            ],
        });
        TestBed.inject(OverlayStore).openModal('cache', { datasetName: 'ds1' });
    });

    it('purgeAll opens a destructive confirm (label "Purge") and purges only on confirm', () => {
        const comp = TestBed.createComponent(CacheModalComponent).componentInstance as any;
        const overlay = TestBed.inject(OverlayStore);
        const openSpy = vi.spyOn(overlay, 'openModal');

        comp.purgeAll();

        // A themed destructive confirm opens; nothing is purged yet.
        expect(openSpy).toHaveBeenCalledWith(
            'confirm',
            expect.objectContaining({ destructive: true, confirmLabel: 'Purge' }),
        );
        expect(api.purgeCache).not.toHaveBeenCalled();

        // The purge only fires from the modal's confirm callback.
        const data = openSpy.mock.calls.at(-1)![1] as { onConfirm: () => void };
        data.onConfirm();
        expect(api.purgeCache).toHaveBeenCalledWith('ds1', {});
    });
});
