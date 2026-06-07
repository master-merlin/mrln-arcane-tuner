import { TestBed } from '@angular/core/testing';

import { RunSummaryComponent } from './run-summary';
import { Job } from '../../../services/job';

/**
 * The Projects → Runs status chip must match the redesign (ProjectRuns in
 * screen-extras.jsx): running = success (+ dot), queued/pending = warning,
 * failed = danger, completed = a NEUTRAL chip with a green check (isDone) —
 * not the legacy solid-green `success` chip.
 */
function comp(status: string): RunSummaryComponent {
    TestBed.resetTestingModule();
    TestBed.configureTestingModule({ imports: [RunSummaryComponent] });
    const fixture = TestBed.createComponent(RunSummaryComponent);
    fixture.componentRef.setInput('job', { status, created_at: 0 } as unknown as Job);
    return fixture.componentInstance;
}

describe('RunSummaryComponent — status colours match the redesign', () => {
    it('completed → neutral chip + green check (isDone), no success tone', () => {
        const c = comp('completed');
        expect(c['isDone']()).toBe(true);
        expect(c['statusTone']()).toBe(''); // not "success" — the legacy look
    });

    it('pending → warning (queued), not the legacy teal', () => {
        const c = comp('pending');
        expect(c['isDone']()).toBe(false);
        expect(c['statusTone']()).toBe('warning');
    });

    it('running → success', () => {
        expect(comp('running')['statusTone']()).toBe('success');
    });

    it('failed → danger', () => {
        expect(comp('failed')['statusTone']()).toBe('danger');
    });

    it('stopped / paused → warning', () => {
        expect(comp('stopped')['statusTone']()).toBe('warning');
        expect(comp('paused')['statusTone']()).toBe('warning');
    });
});
