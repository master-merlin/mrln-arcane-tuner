import { TestBed } from '@angular/core/testing';
import { JobsViewState } from './jobs-view.state';
import { JobStatus, type Job } from '../services/job';

function makeJob(id: string, status: JobStatus, over: Partial<Job> = {}): Job {
    return { id, plugin_id: 'p', config: {}, status, created_at: 0, ...over };
}

describe('JobsViewState', () => {
    function make(): JobsViewState {
        TestBed.configureTestingModule({ providers: [JobsViewState] });
        return TestBed.inject(JobsViewState);
    }

    it('starts empty: no jobs, no selection, no selected job', () => {
        const state = make();
        expect(state.activeJobs()).toEqual([]);
        expect(state.archivedJobs()).toEqual([]);
        expect(state.selectedId()).toBeNull();
        expect(state.selectedJob()).toBeNull();
    });

    it('falls back to the running job when nothing is explicitly selected', () => {
        const state = make();
        const running = makeJob('job-run', JobStatus.RUNNING);
        state.activeJobs.set([makeJob('job-pending', JobStatus.PENDING), running]);
        expect(state.selectedJob()).toBe(running);
    });

    it('select() sets an explicit selection that wins over the running-job fallback', () => {
        const state = make();
        const running = makeJob('job-run', JobStatus.RUNNING);
        const pending = makeJob('job-pending', JobStatus.PENDING);
        state.activeJobs.set([pending, running]);
        state.select('job-pending');
        expect(state.selectedId()).toBe('job-pending');
        expect(state.selectedJob()).toBe(pending);
    });

    it('resolves an explicit selection from archivedJobs, not just activeJobs', () => {
        const state = make();
        const archived = makeJob('job-old', JobStatus.COMPLETED);
        state.archivedJobs.set([archived]);
        state.select('job-old');
        expect(state.selectedJob()).toBe(archived);
    });

    it('falls back to the running job when the explicit selection id no longer exists', () => {
        const state = make();
        const running = makeJob('job-run', JobStatus.RUNNING);
        state.activeJobs.set([running]);
        state.select('job-does-not-exist');
        // Selection recorded, but resolution can't find it -> falls back to running.
        expect(state.selectedId()).toBe('job-does-not-exist');
        expect(state.selectedJob()).toBe(running);
    });

    it('returns null when nothing is selected and no job is running', () => {
        const state = make();
        state.activeJobs.set([makeJob('job-pending', JobStatus.PENDING)]);
        expect(state.selectedJob()).toBeNull();
    });

    it('reacts to activeJobs/archivedJobs updates — selectedJob recomputes live', () => {
        const state = make();
        state.select('job-a');
        expect(state.selectedJob()).toBeNull();
        const a = makeJob('job-a', JobStatus.PAUSED);
        state.activeJobs.set([a]);
        expect(state.selectedJob()).toBe(a);
    });
});
