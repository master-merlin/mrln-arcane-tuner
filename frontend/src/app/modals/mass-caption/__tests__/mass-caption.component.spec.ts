/**
 * Mass-caption modal — launcher behaviour spec.
 *
 * Verifies that start() dispatches a batchCaption request and that closing
 * the modal does NOT cancel the background task (the onDestroy stop was
 * removed in Task 9). The per-image processQueue store writes are gone —
 * the backend owns the loop now.
 */
import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { MassCaptionModalComponent } from '../mass-caption.component';
import { OverlayStore } from '../../../state/overlay.store';
import { MediaItemStore } from '../../../state/media-item.store';
import { CaptionCacheStore } from '../../../state/caption-cache.store';
import { DatasetSyncService } from '../../../state/dataset-sync.service';
import { DatasetService } from '../../../services/dataset';
import { WebSocketService } from '../../../services/websocket.service';
import { ToastService } from '../../../services/toast';
import { TaskStore } from '../../../state/task.store';

function makePair(mediaFile: string) {
    return {
        media_file: mediaFile, caption_file: null, media_type: 'image',
        caption_content: '', masked_caption_content: null,
        metadata: { enabled: true, width: 512, height: 512 },
    };
}

describe('MassCaptionModalComponent — launcher contract (Task 9)', () => {
    let api: any;
    let overlay: OverlayStore;
    let taskStoreSpy: {
        byId: Mock;
        active: ReturnType<typeof signal>;
        cancel: Mock;
    };

    beforeEach(() => {
        api = {
            getDatasetPairs: vi.fn().mockReturnValue(of([])),
            batchCaption: vi.fn().mockReturnValue(of({ task_id: 't1' })),
        };
        taskStoreSpy = { byId: vi.fn().mockReturnValue(signal(undefined)), active: signal([]), cancel: vi.fn() };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore, MediaItemStore, CaptionCacheStore,
                { provide: DatasetService, useValue: api },
                { provide: WebSocketService, useValue: { entityChanged: signal(null), reconnected: signal(0) } },
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn() } },
                { provide: TaskStore, useValue: taskStoreSpy },
                { provide: DatasetSyncService, useValue: { refreshDataset: vi.fn().mockReturnValue(Promise.resolve()) } },
            ],
        });
        overlay = TestBed.inject(OverlayStore);
        overlay.openModal('mass-caption', { datasetName: 'ds1' });
    });

    it('fires batchCaption and stores the task_id on execute', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.pairs.set([makePair('a.png')]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        expect(api.batchCaption).toHaveBeenCalled();
        expect(comp.taskId()).toBe('t1');
    });

    it('does NOT run a client-side processQueue loop (no generateCaption call)', () => {
        api.generateCaption = vi.fn().mockReturnValue(of({ caption: 'x' }));
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.pairs.set([makePair('a.png'), makePair('b.png')]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        expect(api.generateCaption).not.toHaveBeenCalled();
    });

    it('closing the modal does NOT cancel the task (no onDestroy running.set(false))', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.pairs.set([makePair('a.png')]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        expect(comp.running()).toBe(true);
        // Simulate the component being destroyed (modal close)
        fixture.destroy();
        // running should still be true — the background task is not aborted
        expect(comp.running()).toBe(true);
    });

    it('cancel() delegates to TaskStore.cancel and clears running flag', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.pairs.set([makePair('a.png')]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        comp.cancel();
        expect(taskStoreSpy.cancel).toHaveBeenCalledWith('t1');
        expect(comp.running()).toBe(false);
    });

    it('reattaches to an in-flight original caption task for the dataset on open (no duplicate launch)', () => {
        // A caption task for ds1 is already running when the modal opens.
        taskStoreSpy.active = signal([
            { id: 'live', type: 'caption_batch', dataset_name: 'ds1', target: 'original', status: 'running' },
        ]);
        const live = signal<any>({ current: 12, total: 36, ok: 12, failed: 0 });
        taskStoreSpy.byId.mockReturnValue(live);

        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        fixture.detectChanges();

        // Bound to the existing task, showing progress — not the launcher.
        expect(comp.running()).toBe(true);
        expect(comp.taskId()).toBe('live');
        expect(comp.pct()).toBe(33);
        // Did not re-fetch pairs to build a fresh candidate list.
        expect(api.getDatasetPairs).not.toHaveBeenCalled();
    });

    it('does NOT reattach to a masked caption task (that belongs to the mass-mask modal)', () => {
        // Same dataset, but target=masked → this original-caption modal must ignore it.
        taskStoreSpy.active = signal([
            { id: 'masked', type: 'caption_batch', dataset_name: 'ds1', target: 'masked', status: 'running' },
        ]);
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        fixture.detectChanges();
        expect(comp.running()).toBe(false);
        expect(comp.taskId()).toBe(null);
    });

    it('pct() reflects task progress from TaskStore', () => {
        const taskSignal = signal<any>({ current: 3, total: 10, current_item: 'img.png', title: 'Captioning' });
        taskStoreSpy.byId.mockReturnValue(taskSignal);

        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.pairs.set([makePair('a.png')]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        // After taskId is set, task() resolves via byId
        expect(comp.pct()).toBe(30);
    });
});
