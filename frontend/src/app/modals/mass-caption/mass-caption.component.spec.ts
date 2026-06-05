import { TestBed, fakeAsync, flushMicrotasks, tick } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { MassCaptionModalComponent } from './mass-caption.component';
import { OverlayStore } from '../../state/overlay.store';
import { MediaItemStore, mediaKey } from '../../state/media-item.store';
import { CaptionCacheStore } from '../../state/caption-cache.store';
import { DatasetService } from '../../services/dataset';
import { WebSocketService } from '../../services/websocket.service';
import { ToastService } from '../../services/toast';

function makePair(mediaFile: string) {
    return {
        media_file: mediaFile, caption_file: null, media_type: 'image',
        caption_content: '', masked_caption_content: null,
        metadata: { enabled: true, width: 512, height: 512 },
    };
}

describe('MassCaptionComponent live updates', () => {
    let api: any;
    let overlay: OverlayStore;
    let media: MediaItemStore;
    let captions: CaptionCacheStore;

    beforeEach(() => {
        api = {
            // The reconcile call from completion re-fetches with caption_file populated.
            getDatasetPairs: jasmine.createSpy('getDatasetPairs').and.returnValue(
                of([{ ...makePair('a.png'), caption_file: 'a.txt' }]),
            ),
            generateCaption: jasmine.createSpy('generateCaption').and.returnValue(of({ caption: 'a cat' })),
            saveCaption: jasmine.createSpy('saveCaption').and.returnValue(of({})),
            unloadModels: jasmine.createSpy('unloadModels').and.returnValue(of({})),
        };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore, MediaItemStore, CaptionCacheStore,
                { provide: DatasetService, useValue: api },
                { provide: WebSocketService, useValue: { entityChanged: signal(null), reconnected: signal(0) } },
                { provide: ToastService, useValue: { success: jasmine.createSpy(), error: jasmine.createSpy(), info: jasmine.createSpy() } },
            ],
        });
        overlay = TestBed.inject(OverlayStore);
        media = TestBed.inject(MediaItemStore);
        captions = TestBed.inject(CaptionCacheStore);
        overlay.openModal('mass-caption', { datasetName: 'ds1' });
    });

    it('writes generated caption text to CaptionCacheStore and stamps the media item', fakeAsync(() => {
        // Seed the MediaItemStore synchronously so stampCaption can find the row.
        // makePair produces caption_file: null so has_caption starts false —
        // only the live stampCaption call can flip it to true before the reconcile.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        media.upsertFromPair('ds1', makePair('a.png') as any);

        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, systemPrompt: '' };
        comp.target.set('original');
        comp.running.set(true);

        comp.processQueue([makePair('a.png')], 0);
        // generateCaption + saveCaption emit synchronously (both return of(...)),
        // so setCaption + stampCaption have already run here — BEFORE the
        // setTimeout that reaches the terminal-guard loadForDataset reconcile.
        // Asserting now isolates the LIVE writes (the reconcile would set
        // has_caption anyway, making the post-tick assertion a tautology).
        expect(api.generateCaption).toHaveBeenCalled();
        expect(captions.get('ds1').get('a.png')?.caption_content).toBe('a cat');
        expect(media.byId(mediaKey('ds1', 'a.png'))()?.has_caption).toBe(true);

        tick(200);   // drain the queue setTimeout + fire-and-forget reconcile
        flushMicrotasks(); // flush the completion loadForDataset Promise
        expect(api.getDatasetPairs).toHaveBeenCalled();
    }));

    it('writes masked caption text to CaptionCacheStore for the masked target', fakeAsync(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        media.upsertFromPair('ds1', makePair('a.png') as any);
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, systemPrompt: '' };
        comp.target.set('masked');
        comp.running.set(true);
        comp.processQueue([makePair('a.png')], 0);
        tick(200);
        expect(api.generateCaption).toHaveBeenCalled();
        expect(captions.get('ds1').get('a.png')?.masked_caption_content).toBe('a cat');
    }));

    it('unloads the caption model when the batch completes', fakeAsync(() => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, systemPrompt: '' };
        comp.target.set('original');
        comp.running.set(true);
        comp.processQueue([makePair('a.png')], 0);
        tick(200);
        flushMicrotasks();
        expect(api.unloadModels).toHaveBeenCalled();
    }));

    it('unloads the caption model when the run is stopped early', fakeAsync(() => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, systemPrompt: '' };
        comp.target.set('original');
        comp.running.set(true);
        comp.cancel();   // user hits Stop → running flips off
        // Re-entering the queue with running=false hits the terminal guard,
        // which must free VRAM even though the batch never finished.
        comp.processQueue([makePair('a.png'), makePair('b.png')], 0);
        tick(200);
        flushMicrotasks();
        expect(api.unloadModels).toHaveBeenCalled();
    }));
});
