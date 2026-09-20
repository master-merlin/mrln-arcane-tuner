/**
 * video-trim-editor — emits trimChanged only on COMMIT (not per-drag) and
 * computes the 4n+1 / 8n+1 family verdicts from the effective window.
 */
import { TestBed } from '@angular/core/testing';
import { VideoTrimEditorComponent, type TrimChange } from './video-trim-editor';
import { ModelContextStore } from '../../../state/model-context.store';

describe('VideoTrimEditorComponent', () => {
    let fixture: ReturnType<typeof TestBed.createComponent<VideoTrimEditorComponent>> | null = null;

    beforeEach(() => {
        fixture = null;
        localStorage.clear();
        TestBed.configureTestingModule({});
    });

    afterEach(() => {
        fixture?.destroy();
        fixture = null;
    });

    function make(inputs: { duration: number; fps?: number; trimStartS?: number | null; trimEndS?: number | null; currentTime?: number | null }) {
        fixture = TestBed.createComponent(VideoTrimEditorComponent);
        const comp = fixture.componentInstance;
        fixture.componentRef.setInput('duration', inputs.duration);
        if (inputs.fps !== undefined) fixture.componentRef.setInput('fps', inputs.fps);
        fixture.componentRef.setInput('trimStartS', inputs.trimStartS ?? null);
        fixture.componentRef.setInput('trimEndS', inputs.trimEndS ?? null);
        fixture.componentRef.setInput('currentTime', inputs.currentTime ?? null);
        fixture.detectChanges();
        return { fixture: fixture!, comp };
    }

    it('seeds the window from the trim inputs (null = full clip)', () => {
        const { comp } = make({ duration: 10, trimStartS: 2, trimEndS: 8 });
        expect((comp as any).start()).toBe(2);
        expect((comp as any).end()).toBe(8);

        const full = make({ duration: 10 });
        expect((full.comp as any).start()).toBe(0);
        expect((full.comp as any).end()).toBe(10);
    });

    it('does NOT emit on a drag (input) — only on commit', () => {
        const { comp } = make({ duration: 10, fps: 24 });
        const spy = vi.fn();
        comp.trimChanged.subscribe(spy);
        (comp as any).onStartInput('3'); // simulates a drag
        (comp as any).onEndInput('7');
        expect(spy).not.toHaveBeenCalled();
        (comp as any).commit(); // pointerup
        expect(spy).toHaveBeenCalledTimes(1);
    });

    it('commit collapses clip-extent bounds to null (clears stored trim)', () => {
        const { comp } = make({ duration: 10 });
        const spy = vi.fn();
        comp.trimChanged.subscribe(spy);
        // Window equals the full clip [0,10] → both bounds null.
        (comp as any).commit();
        const change: TrimChange = spy.mock.lastCall![0];
        expect(change).toEqual({ start: null, end: null });
    });

    it('commit emits interior bounds as numbers', () => {
        const { comp } = make({ duration: 10 });
        const spy = vi.fn();
        comp.trimChanged.subscribe(spy);
        (comp as any).onStartInput('2');
        (comp as any).onEndInput('9');
        (comp as any).commit();
        const change: TrimChange = spy.mock.lastCall![0];
        expect(change).toEqual({ start: 2, end: 9 });
    });

    it('computes 4n+1 / 8n+1 family verdicts from the effective frame count', () => {
        // Want exactly 81 frames (passes both 4n+1 and 8n+1) at 27fps → 3.0s.
        const { comp } = make({ duration: 10, fps: 27, trimStartS: 0, trimEndS: 3 });
        expect((comp as any).effectiveFrames()).toBe(81);
        expect((comp as any).familyPass()).toEqual([true, true]); // [4n+1, 8n+1]
    });

    it('a window failing both rules reports both chips false', () => {
        // 24fps × (0..0.5) = 12 frames; 12 % 4 = 0 and 12 % 8 = 4 → both fail.
        const { comp } = make({ duration: 10, fps: 24, trimStartS: 0, trimEndS: 0.5 });
        expect((comp as any).effectiveFrames()).toBe(12);
        expect((comp as any).familyPass()).toEqual([false, false]);
    });

    it('start cannot cross end and vice versa', () => {
        const { comp } = make({ duration: 10, trimStartS: 2, trimEndS: 5 });
        (comp as any).onStartInput('9'); // beyond end (5) → clamps to end
        expect((comp as any).start()).toBe(5);
        (comp as any).onEndInput('1'); // below start (5) → clamps to start
        expect((comp as any).end()).toBe(5);
    });

    it('set-from-playhead commits the new bound', () => {
        const { comp } = make({ duration: 10, currentTime: 4 });
        const spy = vi.fn();
        comp.trimChanged.subscribe(spy);
        (comp as any).setStartFromPlayhead();
        expect((comp as any).start()).toBe(4);
        expect(spy).toHaveBeenCalledTimes(1);
        expect(spy.mock.lastCall![0].start).toBe(4);
    });

    function chips(root: HTMLElement): { label: string; pass: boolean }[] {
        return Array.from(root.querySelectorAll<HTMLElement>('[data-testid^="vte-chip-"]')).map(el => ({
            label: el.textContent!.trim(),
            pass: el.classList.contains('pass'),
        }));
    }

    function activate(frameRule: string | undefined): void {
        const store = TestBed.inject(ModelContextStore);
        store.setModelAware(true);
        // The rule is the FIXTURE's: 19n+5 belongs to no shipped family, so a
        // rule literal baked into the component cannot satisfy these tests.
        store.setDefinition({ id: 'probe-def', family: 'probe', name: 'Probe', frame_rule: frameRule });
    }

    it('trim_editor_shows_the_active_definition_rule', () => {
        activate('19n+5');
        // 24 frames = 19*1+5: on the 19n+5 ladder, NOT on 17n+5 / 4n+1 / 8n+1.
        const { fixture } = make({ duration: 10, fps: 24, trimStartS: 0, trimEndS: 1 });
        expect(chips(fixture.nativeElement)).toEqual([{ label: '19n+5', pass: true }]);
        const chip = fixture.nativeElement.querySelector('[data-testid="vte-chip-19"]') as HTMLElement;
        expect(chip.getAttribute('title')).toBe('19n+5 OK');
    });

    it('the active rule chip fails off-ladder and follows a definition switch', () => {
        activate('19n+5');
        // 22 frames: on the 17n+5 ladder, off the 19n+5 one.
        const { fixture } = make({ duration: 10, fps: 22, trimStartS: 0, trimEndS: 1 });
        expect(chips(fixture.nativeElement)).toEqual([{ label: '19n+5', pass: false }]);
        activate('17n+5');
        fixture.detectChanges();
        expect(chips(fixture.nativeElement)).toEqual([{ label: '17n+5', pass: true }]);
    });

    it('trim_editor_falls_back_without_a_model (and for a definition with no rule)', () => {
        const agnostic = make({ duration: 10, fps: 27, trimStartS: 0, trimEndS: 3 });
        expect(chips(agnostic.fixture.nativeElement)).toEqual([
            { label: '4n+1', pass: true },
            { label: '8n+1', pass: true },
        ]);
        agnostic.fixture.destroy();

        activate(undefined); // an image definition serves frame_rule: null
        const noRule = make({ duration: 10, fps: 27, trimStartS: 0, trimEndS: 3 });
        expect(chips(noRule.fixture.nativeElement).map(c => c.label)).toEqual(['4n+1', '8n+1']);
    });
});
