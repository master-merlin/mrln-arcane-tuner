/**
 * SimilarImagesModalComponent — delete-confirm contract spec.
 *
 * deleteOne() is destructive (removes the image, caption and masks from disk),
 * so it must route through the themed confirm modal: the delete only fires from
 * the modal's onConfirm callback, never synchronously.
 */
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { SimilarImagesModalComponent } from './similar-images.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';
import { RuntimeConfigService } from '../../services/runtime-config.service';

describe('SimilarImagesModalComponent — delete confirm contract', () => {
    let api: { deletePair: ReturnType<typeof vi.fn> };

    beforeEach(() => {
        api = { deletePair: vi.fn().mockReturnValue(of({})) };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore,
                { provide: DatasetService, useValue: api },
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn() } },
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api', mediaBaseUrl: '/media' } },
            ],
        });
        TestBed.inject(OverlayStore).openModal('similar-images', { datasetName: 'ds1', items: [] });
    });

    it('deleteOne opens a destructive confirm and deletes only on confirm', () => {
        const comp = TestBed.createComponent(SimilarImagesModalComponent).componentInstance as any;
        const overlay = TestBed.inject(OverlayStore);
        const openSpy = vi.spyOn(overlay, 'openModal');

        comp.deleteOne({ path: 'dup.png' });

        // A themed destructive confirm opens; nothing is deleted yet.
        expect(openSpy).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
        expect(api.deletePair).not.toHaveBeenCalled();

        // The delete only fires from the modal's confirm callback.
        const data = openSpy.mock.calls.at(-1)![1] as { onConfirm: () => void };
        data.onConfirm();
        expect(api.deletePair).toHaveBeenCalledWith('ds1', 'dup.png');
    });
});
