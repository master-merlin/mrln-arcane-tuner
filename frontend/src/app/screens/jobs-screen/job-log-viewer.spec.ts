import { describe, it, expect, vi, afterEach } from 'vitest';
import { TestBed, type ComponentFixture } from '@angular/core/testing';

import { JobLogViewerComponent } from './job-log-viewer';
import { ToastService } from '../../services/toast';
import type { LogLine } from '../../shared/job-metrics';

/**
 * T5 — job-log viewer. A lightweight, scrollable log region (much more history
 * than the old 14-line tail) with a text filter, Copy + Download actions and an
 * auto-Follow (stick-to-bottom) toggle that disengages when the user scrolls up.
 */

function lines(...texts: string[]): LogLine[] {
    return texts.map((t) => ({ tone: 'teal' as const, level: 'INFO', text: t }));
}

function setup(): { fixture: ComponentFixture<JobLogViewerComponent>; comp: JobLogViewerComponent } {
    TestBed.configureTestingModule({
        imports: [JobLogViewerComponent],
        providers: [{ provide: ToastService, useValue: { success: vi.fn(), error: vi.fn() } }],
    });
    const fixture = TestBed.createComponent(JobLogViewerComponent);
    return { fixture, comp: fixture.componentInstance };
}

type ViewerInternals = {
    query: { set: (v: string) => void };
    follow: () => boolean;
    filtered: () => LogLine[];
    toggleFollow: () => void;
    onScroll: (e: { target: Partial<HTMLElement> }) => void;
    copy: () => void;
    download: () => void;
};

afterEach(() => {
    delete (navigator as unknown as { clipboard?: unknown }).clipboard;
});

describe('JobLogViewerComponent filter', () => {
    it('narrows lines by the case-insensitive query', () => {
        const { fixture, comp } = setup();
        fixture.componentRef.setInput('lines', lines('loading model', 'step 5 loss', 'ERROR boom'));
        fixture.detectChanges();
        const c = comp as unknown as ViewerInternals;
        expect(c.filtered().length).toBe(3);
        c.query.set('LOSS');
        expect(c.filtered().length).toBe(1);
        expect(c.filtered()[0].text).toBe('step 5 loss');
    });

    it('renders substantially more than 14 rows when given a long log', () => {
        const many = Array.from({ length: 200 }, (_, i) => `line ${i}`);
        const { fixture } = setup();
        fixture.componentRef.setInput('lines', lines(...many));
        fixture.detectChanges();
        const rows = (fixture.nativeElement as HTMLElement).querySelectorAll('[data-testid="job-log-line"]');
        expect(rows.length).toBe(200);
    });
});

describe('JobLogViewerComponent follow toggle', () => {
    it('defaults to following and toggles', () => {
        const { comp } = setup();
        const c = comp as unknown as ViewerInternals;
        expect(c.follow()).toBe(true);
        c.toggleFollow();
        expect(c.follow()).toBe(false);
        c.toggleFollow();
        expect(c.follow()).toBe(true);
    });

    it('disengages follow when the user scrolls up and re-engages at the bottom', () => {
        const { comp } = setup();
        const c = comp as unknown as ViewerInternals;
        expect(c.follow()).toBe(true);
        // Scrolled up: a big gap remains below the viewport.
        c.onScroll({ target: { scrollHeight: 1000, scrollTop: 100, clientHeight: 300 } });
        expect(c.follow()).toBe(false);
        // Scrolled back to the bottom: follow re-engages.
        c.onScroll({ target: { scrollHeight: 1000, scrollTop: 700, clientHeight: 300 } });
        expect(c.follow()).toBe(true);
    });
});

describe('JobLogViewerComponent copy + download', () => {
    it('copies the (filtered) lines to the clipboard', () => {
        const writeText = vi.fn().mockResolvedValue(undefined);
        Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
        const { fixture, comp } = setup();
        fixture.componentRef.setInput('lines', lines('alpha', 'beta'));
        fixture.detectChanges();
        (comp as unknown as ViewerInternals).copy();
        expect(writeText).toHaveBeenCalledWith('alpha\nbeta');
    });

    it('download builds a blob and clicks an anchor', () => {
        const createObjectURL = vi.fn(() => 'blob:x');
        const revokeObjectURL = vi.fn();
        (URL as unknown as { createObjectURL: unknown }).createObjectURL = createObjectURL;
        (URL as unknown as { revokeObjectURL: unknown }).revokeObjectURL = revokeObjectURL;
        const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
        const { fixture, comp } = setup();
        fixture.componentRef.setInput('lines', lines('alpha', 'beta'));
        fixture.detectChanges();
        (comp as unknown as ViewerInternals).download();
        expect(createObjectURL).toHaveBeenCalled();
        expect(clickSpy).toHaveBeenCalled();
        clickSpy.mockRestore();
    });
});
