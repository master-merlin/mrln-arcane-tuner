import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { MassCaptionModalComponent } from './mass-caption.component';
import { OverlayStore } from '../../state/overlay.store';
import { MediaItemStore } from '../../state/media-item.store';
import { CaptionCacheStore } from '../../state/caption-cache.store';
import { DatasetService } from '../../services/dataset';
import { WebSocketService } from '../../services/websocket.service';
import { ToastService } from '../../services/toast';
import { TaskStore } from '../../state/task.store';

function makePair(mediaFile: string) {
    return {
        media_file: mediaFile, caption_file: null, media_type: 'image',
        caption_content: '', masked_caption_content: null,
        metadata: { enabled: true, width: 512, height: 512 },
    };
}

describe('MassCaptionComponent launcher', () => {
    let api: any;
    let overlay: OverlayStore;

    beforeEach(() => {
        api = {
            getDatasetPairs: jasmine.createSpy('getDatasetPairs').and.returnValue(of([])),
            batchCaption: jasmine.createSpy('batchCaption').and.returnValue(of({ task_id: 't1' })),
        };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore, MediaItemStore, CaptionCacheStore,
                { provide: DatasetService, useValue: api },
                { provide: WebSocketService, useValue: { entityChanged: signal(null), reconnected: signal(0) } },
                { provide: ToastService, useValue: { success: jasmine.createSpy(), error: jasmine.createSpy(), info: jasmine.createSpy() } },
                { provide: TaskStore, useValue: { byId: () => signal(undefined), cancel: jasmine.createSpy() } },
            ],
        });
        overlay = TestBed.inject(OverlayStore);
        overlay.openModal('mass-caption', { datasetName: 'ds1' });
    });

    it('Execute posts a batch caption task and runs no client loop', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.pairs.set([{ media_file: 'a.png', caption_file: null, caption_content: '' }]);
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        expect(api.batchCaption).toHaveBeenCalled();
        expect(comp.taskId()).toBe('t1');
    });

    it('sets running=true immediately and stores task_id on success', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm2', params: { steps: 10 }, resolvedSystemPrompt: 'describe' };
        comp.target.set('original');
        comp.pairs.set([makePair('b.png')]);
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        expect(comp.running()).toBe(true);
        expect(comp.taskId()).toBe('t1');
    });

    it('passes all required batch caption fields to the API', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'flux-1', params: { max_new_tokens: 256 }, resolvedSystemPrompt: 'a photo of' };
        comp.target.set('masked');
        comp.pairs.set([
            { media_file: 'img1.png', caption_file: null, caption_content: '', metadata: { has_mask: true } },
        ]);
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        expect(api.batchCaption).toHaveBeenCalledWith(jasmine.objectContaining({
            dataset_name: 'ds1',
            image_rel_paths: ['img1.png'],
            model_id: 'flux-1',
            system_prompt: 'a photo of',
            target: 'masked',
        }));
    });

    it('does not call batchCaption when candidates list is empty', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        // All pairs have captions → strategy='keep' yields zero candidates
        comp.pairs.set([{ media_file: 'a.png', caption_file: 'a.txt', caption_content: 'already captioned' }]);
        comp.start();
        expect(api.batchCaption).not.toHaveBeenCalled();
    });

    it('does not call batchCaption when user cancels the confirm dialog', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.pairs.set([makePair('c.png')]);
        spyOn(window, 'confirm').and.returnValue(false);
        comp.start();
        expect(api.batchCaption).not.toHaveBeenCalled();
    });

    it('cancel() calls TaskStore.cancel with the task id and resets running', () => {
        const taskStore = TestBed.inject(TaskStore) as any;
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.pairs.set([makePair('d.png')]);
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        expect(comp.taskId()).toBe('t1');
        comp.cancel();
        expect(taskStore.cancel).toHaveBeenCalledWith('t1');
        expect(comp.running()).toBe(false);
    });
});
