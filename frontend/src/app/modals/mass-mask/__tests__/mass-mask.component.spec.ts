/**
 * Mass-mask modal — onCompleted callback fires from all three tabs.
 * Generate / Apply / Caption each have their own completion path.
 */
import { TestBed, fakeAsync, flushMicrotasks, tick } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { MassMaskModalComponent } from '../mass-mask.component';
import { OverlayStore } from '../../../state/overlay.store';
import { MediaItemStore } from '../../../state/media-item.store';
import { CaptionCacheStore } from '../../../state/caption-cache.store';
import { DatasetSyncService } from '../../../state/dataset-sync.service';
import { DatasetService } from '../../../services/dataset';
import { WebSocketService } from '../../../services/websocket.service';
import { ToastService } from '../../../services/toast';

function makePair(mediaFile: string, hasMask = false) {
    return {
        media_file: mediaFile, caption_file: undefined, media_type: 'image',
        caption_content: '', masked_caption_content: undefined,
        metadata: { enabled: true, has_mask: hasMask, width: 512, height: 512 },
    };
}

describe('MassMaskModalComponent — onCompleted callback', () => {
    let api: any;
    let overlay: OverlayStore;
    let media: MediaItemStore;
    let onCompleted: jasmine.Spy;

    beforeEach(() => {
        api = {
            getDatasetPairs: jasmine.createSpy('getDatasetPairs').and.returnValue(of([makePair('a.png')])),
            generateMask: jasmine.createSpy('generateMask').and.returnValue(of({ mask_path: 'm', message: 'ok' })),
            generateCaption: jasmine.createSpy('generateCaption').and.returnValue(of({ caption: 'masked cap' })),
            massApplyMasks: jasmine.createSpy('massApplyMasks').and.returnValue(of({ applied: 1, skipped: 0, warnings: [] })),
        };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore, MediaItemStore, CaptionCacheStore,
                { provide: DatasetService, useValue: api },
                { provide: WebSocketService, useValue: { entityChanged: signal(null), reconnected: signal(0) } },
                { provide: DatasetSyncService, useValue: { refreshDataset: jasmine.createSpy('refreshDataset').and.returnValue(Promise.resolve()) } },
                { provide: ToastService, useValue: {
                    success: jasmine.createSpy(),
                    error: jasmine.createSpy(),
                    info: jasmine.createSpy(),
                    warning: jasmine.createSpy(),
                } },
            ],
        });
        overlay = TestBed.inject(OverlayStore);
        media = TestBed.inject(MediaItemStore);
        onCompleted = jasmine.createSpy('onCompleted');
        overlay.openModal('mass-mask', { datasetName: 'ds1', onCompleted });
    });

    it('Generate tab — onCompleted fires after processMaskQueue drains', fakeAsync(() => {
        media.upsertFromPair('ds1', makePair('a.png') as any);

        const fixture = TestBed.createComponent(MassMaskModalComponent);
        const comp = fixture.componentInstance as any;
        comp.maskingSettings = { modelId: 'sam', params: {} };
        comp.running.set(true);

        comp.processMaskQueue([makePair('a.png')], 0);
        tick(200);
        flushMicrotasks();

        expect(onCompleted).toHaveBeenCalledTimes(1);
    }));

    it('Apply tab — onCompleted fires after massApplyMasks success', () => {
        const fixture = TestBed.createComponent(MassMaskModalComponent);
        const comp = fixture.componentInstance as any;
        // Seed at least one masked pair so the early `maskCount === 0`
        // guard in startApply doesn't bail out before the HTTP call.
        comp.pairs.set([makePair('a.png', true)]);
        spyOn(window, 'confirm').and.returnValue(true);

        comp.startApply();

        expect(api.massApplyMasks).toHaveBeenCalled();
        expect(onCompleted).toHaveBeenCalledTimes(1);
    });

    it('Caption tab — onCompleted fires after processCaptionQueue drains', fakeAsync(() => {
        media.upsertFromPair('ds1', makePair('a.png', true) as any);

        const fixture = TestBed.createComponent(MassMaskModalComponent);
        const comp = fixture.componentInstance as any;
        comp.captionSettings = { resolvedModelId: 'm', params: {}, systemPrompt: '' };
        comp.running.set(true);

        comp.processCaptionQueue([makePair('a.png', true)], 0);
        tick(200);
        flushMicrotasks();

        expect(onCompleted).toHaveBeenCalledTimes(1);
    }));

    it('does NOT fire onCompleted on cancel (running=false) mid-run', fakeAsync(() => {
        const fixture = TestBed.createComponent(MassMaskModalComponent);
        const comp = fixture.componentInstance as any;
        comp.maskingSettings = { modelId: 'sam', params: {} };
        comp.running.set(false);   // already cancelled

        comp.processMaskQueue([makePair('a.png'), makePair('b.png')], 0);
        tick(200);
        flushMicrotasks();

        expect(onCompleted).not.toHaveBeenCalled();
    }));
});
