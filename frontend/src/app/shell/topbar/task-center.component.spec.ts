import type { Mock } from "vitest";
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { TaskCenterComponent } from './task-center.component';
import { TaskStore } from '../../state/task.store';

describe('TaskCenterComponent', () => {
    let cancel: Mock;
    beforeEach(() => {
        cancel = vi.fn();
        TestBed.configureTestingModule({
            imports: [TaskCenterComponent],
            providers: [{ provide: TaskStore, useValue: {
                        active: signal([{ id: 't1', title: 'Captioning · ds', status: 'running',
                                total: 4, current: 2, current_item: 'a.png', ok: 2, failed: 0 }]),
                        activeCount: signal(1),
                        recent: signal([]),
                        cancel,
                    } }],
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
            providers: [{ provide: TaskStore, useValue: {
                        active: signal([]),
                        activeCount: signal(0),
                        recent: signal(recent),
                        cancel: () => undefined,
                    } }],
        });
        const f = TestBed.createComponent(TaskCenterComponent);
        (f.componentInstance as any).toggle();
        f.detectChanges();
        return f.nativeElement.textContent as string;
    }

    it('shows "done" but never "failed" for a clean run', () => {
        const txt = mount([{ id: 'r1', title: 'Captioning · Mitsubishi 3000GT - 1990',
                status: 'completed', ok: 33, failed: 0 }]);
        expect(txt).toContain('33 done');
        expect(txt).not.toContain('failed');
    });

    it('surfaces the failed count when something failed', () => {
        const txt = mount([{ id: 'r2', title: 'Captioning · ds',
                status: 'completed', ok: 30, failed: 3 }]);
        expect(txt).toContain('30 ok');
        expect(txt).toContain('3 failed');
    });

    it('maps type → kind label + accent and uses dataset_name as subject', () => {
        TestBed.resetTestingModule();
        TestBed.configureTestingModule({
            imports: [TaskCenterComponent],
            providers: [{ provide: TaskStore, useValue: {
                        active: signal([]), activeCount: signal(0),
                        recent: signal([{ id: 'r3', type: 'caption_batch', title: 'Captioning · X',
                                dataset_name: 'Mitsubishi 3000GT - 1990', status: 'completed', ok: 5, failed: 0 }]),
                        cancel: () => undefined,
                    } }],
        });
        const f = TestBed.createComponent(TaskCenterComponent);
        (f.componentInstance as any).toggle();
        f.detectChanges();
        const el = f.nativeElement as HTMLElement;
        expect(el.querySelector('.tc-kind')?.textContent).toContain('Captioning');
        expect(el.querySelector('.tc-subject')?.textContent).toContain('Mitsubishi 3000GT - 1990');
        // Accent token reused from the KPI rails, bound as a CSS custom prop.
        // Captioning maps to brand (matching its buttons elsewhere in the app).
        expect(el.querySelector('.tc-row')?.getAttribute('style')).toContain('--color-brand');
    });
});
