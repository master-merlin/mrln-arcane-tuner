import { TestBed, fakeAsync, tick } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { MassMaskModalComponent } from './mass-mask.component';
import { OverlayStore } from '../../state/overlay.store';
import { MediaItemStore, mediaKey } from '../../state/media-item.store';
import { CaptionCacheStore } from '../../state/caption-cache.store';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { DatasetService } from '../../services/dataset';
import { WebSocketService } from '../../services/websocket.service';
import { ToastService } from '../../services/toast';

function makePair(mediaFile: string) {
    return {
        media_file: mediaFile, caption_file: undefined, media_type: 'image',
        caption_content: '', masked_caption_content: undefined,
        metadata: { enabled: true, has_mask: false, width: 512, height: 512 },
    };
}

describe('MassMaskComponent live updates', () => {
    let api: any;
    let overlay: OverlayStore;
    let media: MediaItemStore;
    let captions: CaptionCacheStore;

    beforeEach(() => {
        api = {
            getDatasetPairs: jasmine.createSpy().and.returnValue(of([makePair('a.png')])),
            generateMask: jasmine.createSpy().and.returnValue(of({ mask_path: 'm', message: 'ok' })),
            generateCaption: jasmine.createSpy().and.returnValue(of({ caption: 'masked cap' })),
        };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore, MediaItemStore, CaptionCacheStore,
                { provide: DatasetService, useValue: api },
                { provide: WebSocketService, useValue: { entityChanged: signal(null), reconnected: signal(0) } },
                { provide: DatasetSyncService, useValue: { refreshDataset: jasmine.createSpy('refreshDataset').and.returnValue(Promise.resolve()) } },
                { provide: ToastService, useValue: { success: jasmine.createSpy(), error: jasmine.createSpy(), info: jasmine.createSpy() } },
            ],
        });
        overlay = TestBed.inject(OverlayStore);
        media = TestBed.inject(MediaItemStore);
        captions = TestBed.inject(CaptionCacheStore);
        overlay.openModal('mass-mask', { datasetName: 'ds1' });
    });

    it('flips has_mask on the media item live, before the completion reconcile', fakeAsync(() => {
        media.upsertFromPair('ds1', makePair('a.png'));   // seeded has_mask:false
        const fixture = TestBed.createComponent(MassMaskModalComponent);
        const comp = fixture.componentInstance as any;
        comp.maskingSettings = { modelId: 'sam', params: {} };
        comp.running.set(true);

        const sync = TestBed.inject(DatasetSyncService) as any;

        comp.processMaskQueue([makePair('a.png')], 0);
        // generateMask emits synchronously, so markMaskGenerated has already
        // flipped the flag here — the setTimeout reaching the terminal-guard
        // refreshDataset reconcile has NOT fired yet. Asserting now isolates the
        // LIVE flip (the reconcile would set has_mask regardless).
        expect(api.generateMask).toHaveBeenCalled();
        expect(media.byId(mediaKey('ds1', 'a.png'))()?.has_mask).toBe(true);

        tick(200);   // drain the queue setTimeout + fire-and-forget reconcile
        expect(sync.refreshDataset).toHaveBeenCalledWith('ds1');
    }));

    it('writes masked caption text to CaptionCacheStore as the caption queue runs', fakeAsync(() => {
        media.upsertFromPair('ds1', makePair('a.png'));
        const fixture = TestBed.createComponent(MassMaskModalComponent);
        const comp = fixture.componentInstance as any;
        comp.captionSettings = { resolvedModelId: 'm', params: {}, systemPrompt: '' };
        comp.running.set(true);

        comp.processCaptionQueue([makePair('a.png')], 0);
        tick(200);

        expect(api.generateCaption).toHaveBeenCalled();
        expect(captions.get('ds1').get('a.png')?.masked_caption_content).toBe('masked cap');
    }));
});
