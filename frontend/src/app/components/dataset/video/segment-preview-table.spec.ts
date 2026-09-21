/**
 * segment-preview-table — renders rows + est-frame chips and emits the edited
 * list on delete / merge. Pure/OnPush; the parent owns the segment array.
 */
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { SegmentPreviewTableComponent } from './segment-preview-table';
import type { VideoSegment } from '../../../services/dataset';
import { ModelContextStore } from '../../../state/model-context.store';

function seg(start: number, end: number, label: string | null = null): VideoSegment {
    return { start_s: start, end_s: end, label };
}

describe('SegmentPreviewTableComponent', () => {
    let fixture: ReturnType<typeof TestBed.createComponent<SegmentPreviewTableComponent>> | null = null;

    beforeEach(() => {
        fixture = null;
        localStorage.clear();
        TestBed.configureTestingModule({});
    });

    afterEach(() => {
        fixture?.destroy();
        fixture = null;
    });

    function make(segments: VideoSegment[], opts: { fps?: number; editable?: boolean } = {}) {
        fixture = TestBed.createComponent(SegmentPreviewTableComponent);
        const comp = fixture.componentInstance;
        fixture.componentRef.setInput('segments', segments);
        if (opts.fps !== undefined) fixture.componentRef.setInput('fps', opts.fps);
        if (opts.editable !== undefined) fixture.componentRef.setInput('editable', opts.editable);
        fixture.detectChanges();
        return { fixture: fixture!, comp };
    }

    it('renders one row per segment', () => {
        const { fixture } = make([seg(0, 2), seg(2, 5), seg(5, 6)]);
        const rows = fixture.nativeElement.querySelectorAll('[data-testid="spt-row"]');
        expect(rows.length).toBe(3);
    });

    it('shows an empty state for no segments', () => {
        const { fixture } = make([]);
        expect(fixture.nativeElement.querySelector('[data-testid="spt-empty"]')).toBeTruthy();
        expect(fixture.nativeElement.querySelector('[data-testid="segment-preview-table"]')).toBeFalsy();
    });

    it('computes est frame count from fps (rounded duration × fps)', () => {
        // 2s window at 24fps = 48 frames; rows() exposes the derived view.
        const { comp } = make([seg(0, 2)], { fps: 24 });
        const r = (comp as any).rows();
        expect(r[0].frames).toBe(48);
    });

    it('renders the Frames column only when fps is provided', () => {
        const noFps = make([seg(0, 2)]);
        expect(noFps.fixture.nativeElement.querySelector('[data-testid="spt-frames"]')).toBeFalsy();
        noFps.fixture.destroy();

        const withFps = make([seg(0, 2)], { fps: 24 });
        expect(withFps.fixture.nativeElement.querySelector('[data-testid="spt-frames"]')).toBeTruthy();
    });

    it('emits the list minus the deleted row', () => {
        const { comp } = make([seg(0, 1), seg(1, 2), seg(2, 3)], { editable: true });
        const spy = vi.fn();
        comp.segmentsChange.subscribe(spy);
        (comp as any).remove(1);
        expect(spy).toHaveBeenCalledTimes(1);
        const next: VideoSegment[] = spy.mock.lastCall![0];
        expect(next.map(s => s.start_s)).toEqual([0, 2]);
    });

    it('merges a row into its predecessor (extends prev.end to this.end)', () => {
        const { comp } = make([seg(0, 1), seg(1, 4), seg(4, 5)], { editable: true });
        const spy = vi.fn();
        comp.segmentsChange.subscribe(spy);
        (comp as any).merge(1);
        const next: VideoSegment[] = spy.mock.lastCall![0];
        expect(next.length).toBe(2);
        expect(next[0]).toMatchObject({ start_s: 0, end_s: 4 });
        expect(next[1]).toMatchObject({ start_s: 4, end_s: 5 });
    });

    it('merge is a no-op for the first row', () => {
        const { comp } = make([seg(0, 1), seg(1, 2)], { editable: true });
        const spy = vi.fn();
        comp.segmentsChange.subscribe(spy);
        (comp as any).merge(0);
        expect(spy).not.toHaveBeenCalled();
    });

    it('hides edit controls when not editable', () => {
        const { fixture } = make([seg(0, 1)]);
        expect(fixture.nativeElement.querySelector('[data-testid="spt-delete"]')).toBeFalsy();
    });

    function chipRows(root: HTMLElement): { label: string; pass: boolean }[][] {
        return Array.from(root.querySelectorAll<HTMLElement>('[data-testid="spt-row"]')).map(row =>
            Array.from(row.querySelectorAll<HTMLElement>('[data-testid="spt-chip"]')).map(el => ({
                label: el.textContent!.trim(),
                pass: el.classList.contains('pass'),
            })),
        );
    }

    it('segment_table_falls_back_without_a_model', () => {
        // 1s @ 9fps = 9 frames: 4n+1 and 8n+1 both pass; 2s @ 9fps = 18: both fail.
        const { fixture } = make([seg(0, 1), seg(1, 3)], { fps: 9 });
        expect(chipRows(fixture.nativeElement)).toEqual([
            [{ label: '4n+1', pass: true }, { label: '8n+1', pass: true }],
            [{ label: '4n+1', pass: false }, { label: '8n+1', pass: false }],
        ]);
    });

    // ── LANE-92 VERIFY 3.01 (b): the rule judges TRAINING frames ──────────
    function activateServed(ingestFps: number | null): void {
        const store = TestBed.inject(ModelContextStore);
        store.setModelAware(true);
        // The shape `GET /api/caption-context/definitions` serves.
        store.setDefinition({ id: 'probe-def', family: 'probe', name: 'Probe', frame_rule: '17n+5', ingest_fps: ingestFps });
    }

    function cells(root: HTMLElement, testid: string): (string | null)[] {
        return Array.from(root.querySelectorAll<HTMLElement>('[data-testid="spt-row"]')).map(row => {
            const el = row.querySelector<HTMLElement>(`[data-testid="${testid}"]`);
            return el ? el.textContent!.replace(/\s+/g, ' ').trim() : null;
        });
    }

    it('segment_table_judges_training_frames: 30 fps under a 24 fps clock (5 -> 4 skipped, 90 -> 72 -> 56)', () => {
        activateServed(24.0);
        const { fixture } = make([seg(0, 5 / 30), seg(1, 4)], { fps: 30 });
        const root = fixture.nativeElement as HTMLElement;
        expect(cells(root, 'spt-frames')).toEqual(['4', '72']);
        expect(cells(root, 'spt-source-frames')).toEqual(['file: 5', 'file: 90']);
        expect(cells(root, 'spt-outcome')).toEqual(['skipped', '56 used']);
        expect(chipRows(root)).toEqual([
            [{ label: '17n+5', pass: false }],
            [{ label: '17n+5', pass: false }],
        ]);
        const head = root.querySelector<HTMLElement>('[data-testid="spt-frames-head"]')!;
        expect(head.textContent!.replace(/\s+/g, ' ').trim()).toBe('Frames at 24 fps (file: 30 fps)');
    });

    it('control: ingest_fps null keeps the clip\'s own clock and the plain header', () => {
        activateServed(null);
        const { fixture } = make([seg(1, 4)], { fps: 30 });
        const root = fixture.nativeElement as HTMLElement;
        expect(cells(root, 'spt-frames')).toEqual(['90']); // 17*5+5
        expect(cells(root, 'spt-source-frames')).toEqual([null]);
        expect(cells(root, 'spt-outcome')).toEqual([null]);
        expect(chipRows(root)).toEqual([[{ label: '17n+5', pass: true }]]);
        expect(root.querySelector('[data-testid="spt-frames-head"]')!.textContent!.trim()).toBe('Frames');
    });

    it('segment_table_shows_the_active_definition_rule', () => {
        const store = TestBed.inject(ModelContextStore);
        store.setModelAware(true);
        // 19n+5 belongs to no shipped family: only the definition can supply it.
        store.setDefinition({ id: 'probe-def', family: 'probe', name: 'Probe', frame_rule: '19n+5' });
        // 1s @ 24fps = 24 = 19+5 (pass); 1s..3s = 48 (fail).
        const { fixture } = make([seg(0, 1), seg(1, 3)], { fps: 24 });
        expect(chipRows(fixture.nativeElement)).toEqual([
            [{ label: '19n+5', pass: true }],
            [{ label: '19n+5', pass: false }],
        ]);
    });
});
