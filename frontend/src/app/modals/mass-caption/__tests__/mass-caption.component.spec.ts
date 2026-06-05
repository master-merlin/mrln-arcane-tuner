/**
 * Mass-caption modal — completion-callback spec.
 *
 * Narrow: assert that `data.onCompleted` fires exactly once when the
 * queue drains successfully, and does NOT fire when the user cancels
 * mid-run (running.set(false)). The store-side wiring (captions.setCaption,
 * mediaItems.stampCaption) is covered in the sibling spec; this file
 * locks the new completion contract added in PR4 Task 1.
 */
import { TestBed, fakeAsync, flushMicrotasks, tick } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { MassCaptionModalComponent } from '../mass-caption.component';
import { OverlayStore } from '../../../state/overlay.store';
import { MediaItemStore } from '../../../state/media-item.store';
import { CaptionCacheStore } from '../../../state/caption-cache.store';
import { DatasetService } from '../../../services/dataset';
import { WebSocketService } from '../../../services/websocket.service';
import { ToastService } from '../../../services/toast';

function makePair(mediaFile: string) {
    return {
        media_file: mediaFile, caption_file: null, media_type: 'image',
        caption_content: '', masked_caption_content: null,
        metadata: { enabled: true, width: 512, height: 512 },
    };
}

describe('MassCaptionModalComponent — onCompleted callback', () => {
    let api: any;
    let overlay: OverlayStore;
    let media: MediaItemStore;
    let onCompleted: jasmine.Spy;

    beforeEach(() => {
        api = {
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
        onCompleted = jasmine.createSpy('onCompleted');
        overlay.openModal('mass-caption', { datasetName: 'ds1', onCompleted });
    });

    it('fires onCompleted exactly once when the queue drains', fakeAsync(() => {
        media.upsertFromPair('ds1', makePair('a.png') as any);

        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, systemPrompt: '' };
        comp.target.set('original');
        comp.running.set(true);

        comp.processQueue([makePair('a.png')], 0);
        tick(200);              // drain the recursive setTimeout
        flushMicrotasks();      // flush loadForDataset Promise chain

        expect(onCompleted).toHaveBeenCalledTimes(1);
    }));

    it('does NOT fire onCompleted when running flips to false mid-run (cancel)', fakeAsync(() => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, systemPrompt: '' };
        // Pre-cancelled: running starts false, so the very first processQueue
        // call hits the guard and returns without invoking onCompleted.
        comp.running.set(false);

        comp.processQueue([makePair('a.png'), makePair('b.png')], 0);
        tick(200);
        flushMicrotasks();

        expect(onCompleted).not.toHaveBeenCalled();
    }));
});
