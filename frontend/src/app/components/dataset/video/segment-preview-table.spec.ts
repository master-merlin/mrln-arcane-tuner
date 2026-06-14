/**
 * segment-preview-table — renders rows + est-frame chips and emits the edited
 * list on delete / merge. Pure/OnPush; the parent owns the segment array.
 */
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { SegmentPreviewTableComponent } from './segment-preview-table';
import type { VideoSegment } from '../../../services/dataset';

function seg(start: number, end: number, label: string | null = null): VideoSegment {
    return { start_s: start, end_s: end, label };
}

describe('SegmentPreviewTableComponent', () => {
    let fixture: ReturnType<typeof TestBed.createComponent<SegmentPreviewTableComponent>> | null = null;

    beforeEach(() => {
        fixture = null;
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
});
