import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { TaskCenterComponent } from './task-center.component';
import { TaskStore } from '../../state/task.store';

describe('TaskCenterComponent', () => {
    let cancel: jasmine.Spy;
    beforeEach(() => {
        cancel = jasmine.createSpy('cancel');
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
        expect(txt).toContain('Captioning · ds');
        expect(txt).toContain('2 / 4');
        c.cancel('t1');
        expect(cancel).toHaveBeenCalledWith('t1');
    });
});
