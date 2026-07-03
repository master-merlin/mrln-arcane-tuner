/**
 * scene-detect-modal — enqueues scene_detect, polls getSceneProposals until
 * ready, renders the editable proposals, then splitVideo(mode='auto') on
 * confirm. DatasetService mocked; polling driven with fake timers.
 */
import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { settle } from '../../../testing/async';
import { SceneDetectModalComponent } from './scene-detect-modal';
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';
import { OverlayStore } from '../../state/overlay.store';
import type { DatasetPair } from '../../services/dataset';

function videoPair(media: string, fps = 24): DatasetPair {
    return {
        stem: media.replace(/\.[^.]+$/, ''),
        media_file: media,
        media_type: 'video',
        caption_file: null,
        caption_content: '',
        masked_caption_content: null,
        metadata: { fps, duration_s: 10 },
    };
}

describe('SceneDetectModalComponent', () => {
    let api: any;
    let toast: any;
    let fixture: ReturnType<typeof TestBed.createComponent<SceneDetectModalComponent>> | null = null;

    beforeEach(() => {
        fixture = null;
        api = {
            sceneDetect: vi.fn().mockReturnValue(of({ task_id: 'sd1' })),
            getSceneProposals: vi.fn().mockReturnValue(of({ segments: [], ready: false })),
            splitVideo: vi.fn().mockReturnValue(of({ task_id: 'vs1' })),
        };
        toast = { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() };
        TestBed.configureTestingModule({
            providers: [
                { provide: DatasetService, useValue: api },
                { provide: ToastService, useValue: toast },
            ],
        });
    });

    afterEach(() => {
        fixture?.destroy();
        fixture = null;
        vi.useRealTimers();
    });

    function make(videoPairs: DatasetPair[] = [videoPair('src.mp4')]) {
        const overlay = TestBed.inject(OverlayStore);
        overlay.openModal('scene-detect', { datasetName: 'ds1', videoPairs });
        fixture = TestBed.createComponent(SceneDetectModalComponent);
        const comp = fixture.componentInstance as any;
        fixture.detectChanges();
        return { fixture: fixture!, comp, overlay };
    }

    it('detect() enqueues scene_detect with threshold + min len and enters detecting', () => {
        const { comp } = make();
        comp.sourceRel.set('src.mp4');
        comp.threshold.set(30);
        comp.minSceneLen.set(2);
        comp.detect();
        expect(api.sceneDetect).toHaveBeenCalledWith('ds1', {
            source_rel_path: 'src.mp4',
            threshold: 30,
            min_scene_len_s: 2,
        });
        expect(comp.step()).toBe('detecting');
    });

    it('checkResults() advances to review once proposals are ready', async () => {
        api.getSceneProposals.mockReturnValue(of({
            segments: [{ start_s: 0, end_s: 2, label: null }, { start_s: 2, end_s: 4, label: null }],
            ready: true,
        }));
        const { comp } = make();
        comp.sourceRel.set('src.mp4');
        await comp.checkResults();
        expect(comp.step()).toBe('review');
        expect(comp.segments().length).toBe(2);
    });

    it('auto-poll fires getSceneProposals on the 2s interval and flips to review when ready', async () => {
        vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'setTimeout', 'Date'] });
        const { comp } = make();
        comp.sourceRel.set('src.mp4');
        comp.detect(); // starts polling
        // First tick: not ready.
        await vi.advanceTimersByTimeAsync(2000);
        expect(api.getSceneProposals).toHaveBeenCalledTimes(1);
        expect(comp.step()).toBe('detecting');
        // Next tick: ready → review, polling stops.
        api.getSceneProposals.mockReturnValue(of({ segments: [{ start_s: 0, end_s: 2, label: null }], ready: true }));
        await vi.advanceTimersByTimeAsync(2000);
        expect(comp.step()).toBe('review');
        const callsAtReview = api.getSceneProposals.mock.calls.length;
        // No further polling after ready.
        await vi.advanceTimersByTimeAsync(4000);
        expect(api.getSceneProposals.mock.calls.length).toBe(callsAtReview);
    });

    it('split() posts splitVideo(mode=auto) with the curated segments, toasts + closes', async () => {
        const { comp, overlay } = make();
        comp.sourceRel.set('src.mp4');
        comp.segments.set([{ start_s: 0, end_s: 2, label: null }]);
        comp.split();
        await settle();
        expect(api.splitVideo).toHaveBeenCalledWith('ds1', {
            source_rel_path: 'src.mp4',
            segments: [{ start_s: 0, end_s: 2, label: null }],
            mode: 'auto',
            output_prefix: null,
            archive_source: false,
        });
        expect(toast.success).toHaveBeenCalled();
        // Closing the modal pops it off the overlay stack.
        expect(overlay.modalStack().length).toBe(0);
    });

    it('detect() error surfaces a message and stays on config', () => {
        api.sceneDetect.mockReturnValue(throwError(() => ({ error: { detail: 'nope' } })));
        const { comp } = make();
        comp.sourceRel.set('src.mp4');
        comp.detect();
        expect(comp.step()).toBe('config');
        expect(comp.errorMsg()).toContain('nope');
    });

    it('editing proposals replaces the segment list', () => {
        const { comp } = make();
        comp.segments.set([{ start_s: 0, end_s: 2, label: null }, { start_s: 2, end_s: 4, label: null }]);
        // Simulate the editable preview table emitting a trimmed list.
        comp.segments.set([{ start_s: 0, end_s: 2, label: null }]);
        expect(comp.segments().length).toBe(1);
    });
});
