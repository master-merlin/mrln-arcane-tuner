import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { TaskQueueHintComponent } from './task-queue-hint.component';
import { TaskStore } from '../../state/task.store';

function task(over: Partial<any>): any {
    return {
        id: 'x', type: 'caption_batch', title: 't', status: 'pending',
        dataset_name: 'ds', target: null, total: 10, current: 0,
        current_item: null, ok: 0, failed: 0, created_at: 0,
        started_at: null, finished_at: null, error: null,
        ...over,
    };
}

describe('TaskQueueHintComponent', () => {
    let active: ReturnType<typeof signal<any[]>>;

    function mount(input: any | null | undefined) {
        active = signal<any[]>([]);
        TestBed.configureTestingModule({
            providers: [{ provide: TaskStore, useValue: { active } }],
        });
        const fixture = TestBed.createComponent(TaskQueueHintComponent);
        fixture.componentRef.setInput('task', input);
        return fixture;
    }

    function text(fixture: any): string {
        return (fixture.nativeElement.textContent ?? '').replace(/\s+/g, ' ').trim();
    }

    it('renders nothing when there is no task', () => {
        const f = mount(undefined);
        f.detectChanges();
        expect(f.nativeElement.querySelector('[data-testid="task-queue-hint"]')).toBeNull();
    });

    it('renders nothing while the task is running', () => {
        const f = mount(task({ status: 'running' }));
        f.detectChanges();
        expect(f.nativeElement.querySelector('[data-testid="task-queue-hint"]')).toBeNull();
    });

    it('shows the queued banner with position #1 when nothing else is active', () => {
        const me = task({ id: 'me', status: 'pending', created_at: 200 });
        const f = mount(me);
        active.set([me]);
        f.detectChanges();
        expect(f.nativeElement.querySelector('[data-testid="task-queue-hint"]')).not.toBeNull();
        expect(text(f)).toContain('#1 in queue');
    });

    it('counts the running task plus earlier-queued tasks as ahead', () => {
        const me = task({ id: 'me', status: 'pending', created_at: 200 });
        const f = mount(me);
        active.set([
            task({ id: 'running', status: 'running', created_at: 100 }),  // ahead (running)
            me,                                                            // me
            task({ id: 'later', status: 'pending', created_at: 300 }),     // behind me — not ahead
        ]);
        f.detectChanges();
        // ahead = 1 (the running one) → position 2.
        expect(text(f)).toContain('#2 in queue');
    });
});
