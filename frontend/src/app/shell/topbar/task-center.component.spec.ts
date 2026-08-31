import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import type { ComponentFixture } from '@angular/core/testing';
import { signal } from '@angular/core';
import { TaskCenterComponent } from './task-center.component';
import { TaskStore } from '../../state/task.store';
import { byTestId } from '../../../testing/by-test-id';

describe('TaskCenterComponent', () => {
    let cancel: Mock;
    beforeEach(() => {
        cancel = vi.fn();
        TestBed.configureTestingModule({
            imports: [TaskCenterComponent],
            providers: [
                {
                    provide: TaskStore,
                    useValue: {
                        active: signal([
                            {
                                id: 't1',
                                title: 'Captioning · ds',
                                status: 'running',
                                total: 4,
                                current: 2,
                                current_item: 'a.png',
                                ok: 2,
                                failed: 0,
                            },
                        ]),
                        activeCount: signal(1),
                        recent: signal([]),
                        cancel,
                    },
                },
            ],
        });
    });

    it('renders active task + cancels on click', () => {
        const f = TestBed.createComponent(TaskCenterComponent);
        const c = f.componentInstance as any;
        c.toggle();
        f.detectChanges();
        const txt = f.nativeElement.textContent;
        // Kind (tier 1) and subject (tier 2) render as separate tiers now.
        expect(txt).toContain('Captioning');
        expect(txt).toContain('ds');
        expect(txt).toContain('2 / 4');
        c.cancel('t1');
        expect(cancel).toHaveBeenCalledWith('t1');
    });
});

describe('TaskCenterComponent done-row summary', () => {
    function mount(recent: any[]) {
        TestBed.resetTestingModule();
        TestBed.configureTestingModule({
            imports: [TaskCenterComponent],
            providers: [
                {
                    provide: TaskStore,
                    useValue: {
                        active: signal([]),
                        activeCount: signal(0),
                        recent: signal(recent),
                        cancel: () => undefined,
                    },
                },
            ],
        });
        const f = TestBed.createComponent(TaskCenterComponent);
        (f.componentInstance as any).toggle();
        f.detectChanges();
        lastFixture = f;
        return f.nativeElement.textContent as string;
    }

    /** The fixture mount() last rendered — for assertions that need the DOM
     *  rather than the text (an absent block has no textContent to miss). */
    let lastFixture: ComponentFixture<TaskCenterComponent>;

    it('shows "done" but never "failed" for a clean run', () => {
        const txt = mount([
            {
                id: 'r1',
                title: 'Captioning · Mitsubishi 3000GT - 1990',
                status: 'completed',
                ok: 33,
                failed: 0,
            },
        ]);
        expect(txt).toContain('33 done');
        expect(txt).not.toContain('failed');
    });

    it('surfaces the failed count when something failed', () => {
        const txt = mount([
            { id: 'r2', title: 'Captioning · ds', status: 'completed', ok: 30, failed: 3 },
        ]);
        expect(txt).toContain('30 ok');
        expect(txt).toContain('3 failed');
    });

    it('surfaces a failed task\'s error text in the panel', () => {
        const txt = mount([
            {
                id: 'r4',
                title: 'Captioning · ds',
                status: 'failed',
                ok: 0,
                failed: 2,
                error: 'CUDA out of memory',
            },
        ]);
        expect(txt).toContain('CUDA out of memory');
    });

    it('surfaces the reason on a COMPLETED task that had failures (LANE-52)', () => {
        // A partially-failed batch finishes `completed` and carries the summary
        // on `error` (TaskManager.finish_batch). The row used to render the
        // error only when status === 'failed', so the reason was invisible on
        // exactly the tasks whose outcome was ambiguous.
        const txt = mount([
            {
                id: 'r5',
                title: 'Refine captions · ds',
                status: 'completed',
                ok: 2,
                failed: 1,
                error: '1 of 3 items failed (last: ReadTimeout)',
            },
        ]);
        expect(txt).toContain('1 of 3 items failed (last: ReadTimeout)');
    });

    it('renders no error block when a clean task has none', () => {
        // Positive control for the two tests above: keying on `t.error` must
        // not start printing an empty error row on every healthy task.
        const txt = mount([
            { id: 'r6', title: 'Captioning · ds', status: 'completed', ok: 4, failed: 0 },
        ]);
        expect(txt).not.toContain('items failed');
        expect(byTestId(lastFixture, 'task-center-error')).toBeNull();
    });

    it('maps type → kind label + accent and uses dataset_name as subject', () => {
        TestBed.resetTestingModule();
        TestBed.configureTestingModule({
            imports: [TaskCenterComponent],
            providers: [
                {
                    provide: TaskStore,
                    useValue: {
                        active: signal([]),
                        activeCount: signal(0),
                        recent: signal([
                            {
                                id: 'r3',
                                type: 'caption_batch',
                                title: 'Captioning · X',
                                dataset_name: 'Mitsubishi 3000GT - 1990',
                                status: 'completed',
                                ok: 5,
                                failed: 0,
                            },
                        ]),
                        cancel: () => undefined,
                    },
                },
            ],
        });
        const f = TestBed.createComponent(TaskCenterComponent);
        (f.componentInstance as any).toggle();
        f.detectChanges();
        expect(byTestId(f, 'task-center-kind')!.nativeElement.textContent).toContain('Captioning');
        expect(byTestId(f, 'task-center-subject')!.nativeElement.textContent).toContain(
            'Mitsubishi 3000GT - 1990',
        );
        // Accent token reused from the KPI rails, bound as a CSS custom prop.
        // Captioning maps to brand (matching its buttons elsewhere in the app).
        expect(byTestId(f, 'task-center-row')!.nativeElement.getAttribute('style')).toContain(
            '--color-brand',
        );
    });
});

describe('TaskCenterComponent persistent trigger + clear', () => {
    function mount(active: any[], recent: any[]) {
        const clearRecent = vi.fn();
        TestBed.resetTestingModule();
        TestBed.configureTestingModule({
            imports: [TaskCenterComponent],
            providers: [
                {
                    provide: TaskStore,
                    useValue: {
                        active: signal(active),
                        activeCount: signal(active.length),
                        recent: signal(recent),
                        cancel: () => undefined,
                        clearRecent,
                    },
                },
            ],
        });
        const f = TestBed.createComponent(TaskCenterComponent);
        return { f, clearRecent };
    }

    it('renders the Activity trigger even when idle (no active, no recent)', () => {
        const { f } = mount([], []);
        f.detectChanges();
        const trigger = byTestId(f, 'task-center-trigger');
        expect(trigger).not.toBeNull();
        // It is a real button with an accessible label.
        expect(trigger!.nativeElement.tagName).toBe('BUTTON');
        expect(trigger!.nativeElement.getAttribute('aria-label')).toBeTruthy();
    });

    it('shows an empty state when opened while idle', () => {
        const { f } = mount([], []);
        (f.componentInstance as any).toggle();
        f.detectChanges();
        expect(byTestId(f, 'task-center-empty')).not.toBeNull();
    });

    it('Clear empties the recent list without cancelling active tasks', () => {
        const { f, clearRecent } = mount(
            [{ id: 'a1', title: 'Captioning · ds', status: 'running', total: 4, current: 1, ok: 1, failed: 0 }],
            [{ id: 'r1', title: 'Captioning · ds', status: 'completed', ok: 3, failed: 0 }],
        );
        (f.componentInstance as any).toggle();
        f.detectChanges();
        const clear = byTestId(f, 'task-center-clear');
        expect(clear).not.toBeNull();
        clear!.nativeElement.click();
        expect(clearRecent).toHaveBeenCalledTimes(1);
    });
});
