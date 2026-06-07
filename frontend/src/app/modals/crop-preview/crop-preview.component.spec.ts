/**
 * Crop-preview modal — minimal behavioural specs.
 *
 * Focus: the post-success wiring that the destructive crop bug exposed.
 * After ``DatasetService.cropImage`` returns successfully the modal must
 *   (a) close,
 *   (b) toast,
 *   (c) bump ``MediaItemStore.mediaRev`` so every <img> elsewhere in the
 *       workspace refreshes — without this the URL is unchanged, the
 *       browser doesn't fetch, and the user sees pre-crop bytes while
 *       the file on disk has already been overwritten (and the next
 *       crop destroys more of the original).
 *
 * We deliberately don't drive the full template; the canvas + drag
 * interaction lives in its own helpers and is exercised by the visual
 * QA pass for this PR.
 */
import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { signal } from '@angular/core';
import { CropPreviewModalComponent } from './crop-preview.component';
import { OverlayStore } from '../../state/overlay.store';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';
import { MediaItemStore } from '../../state/media-item.store';

class StubOverlay {
    private _modal = signal<{
        kind: string;
        data: any;
    } | null>({
        kind: 'crop-preview',
        data: { datasetName: 'alpha', path: 'img.jpg', width: 1024, height: 1024 },
    });
    topModal = this._modal;
    closeModal = vi.fn();
}

class StubRtc {
    apiUrl = '/api';
    mediaBaseUrl = '/media';
}

class StubDatasetService {
    cropImage = vi.fn();
    calcCropTargets = vi.fn().mockReturnValue(of({}));
}

class StubToast {
    success = vi.fn();
    error = vi.fn();
    info = vi.fn();
}

class StubMediaItems {
    bumpMedia = vi.fn();
}

describe('CropPreviewModalComponent.applyCrop', () => {
    let cmp: CropPreviewModalComponent;
    let dsApi: StubDatasetService;
    let overlay: StubOverlay;
    let toast: StubToast;
    let mediaItems: StubMediaItems;

    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [
                CropPreviewModalComponent,
                { provide: OverlayStore, useClass: StubOverlay },
                { provide: RuntimeConfigService, useClass: StubRtc },
                { provide: DatasetService, useClass: StubDatasetService },
                { provide: ToastService, useClass: StubToast },
                { provide: MediaItemStore, useClass: StubMediaItems },
            ],
        });
        cmp = TestBed.inject(CropPreviewModalComponent);
        dsApi = TestBed.inject(DatasetService) as any;
        overlay = TestBed.inject(OverlayStore) as any;
        toast = TestBed.inject(ToastService) as any;
        mediaItems = TestBed.inject(MediaItemStore) as any;
    });

    it('on success: bumps mediaRev, toasts, closes the modal', () => {
        dsApi.cropImage.mockReturnValue(of({ status: 'cropped', file: 'img.jpg' }));
        (cmp as any).applyCrop();
        expect(mediaItems.bumpMedia).toHaveBeenCalledTimes(1);
        expect(toast.success).toHaveBeenCalled();
        expect(overlay.closeModal).toHaveBeenCalled();
    });

    it('on error: does NOT bump mediaRev — pre-crop bytes are still on disk', () => {
        dsApi.cropImage.mockReturnValue(throwError(() => ({ error: { detail: 'boom' } })));
        (cmp as any).applyCrop();
        expect(mediaItems.bumpMedia).not.toHaveBeenCalled();
        expect(toast.error).toHaveBeenCalled();
        expect(overlay.closeModal).not.toHaveBeenCalled();
    });

    it('aborts when datasetName or path is missing — no HTTP, no bump', () => {
        (cmp as any).data = { datasetName: 'alpha' /* no path */ };
        (cmp as any).applyCrop();
        expect(dsApi.cropImage).not.toHaveBeenCalled();
        expect(mediaItems.bumpMedia).not.toHaveBeenCalled();
    });
});
