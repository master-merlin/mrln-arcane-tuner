import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Component } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { of } from 'rxjs';

import { JobsScreen } from './jobs-screen';
import { JobService, JobStatus, type Job } from '../../services/job';
import { ResumeJobService } from '../../services/resume-job.service';
import { JobStore } from '../../state/job.store';
import { JobsViewState } from '../../state/jobs-view.state';
import { OverlayStore } from '../../state/overlay.store';
import { TrainingHandoffService } from '../../state/training-handoff.service';
import { ScopeStore } from '../../state/scope.store';
import { TemplateService } from '../../services/template.service';
import { ToastService } from '../../services/toast';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { Router } from '@angular/router';
import { TrainingJobQueueComponent } from '../../components/training/training-job-queue/training-job-queue';
import { SystemMonitorComponent } from '../../components/system/system-monitor/system-monitor';

/**
 * Stub the two sibling-pane child components — they own their own WS/service
 * machinery (irrelevant to the lightbox under test) and would otherwise force
 * us to mock unrelated services. Selectors match the template so the screen's
 * own DOM still renders.
 */
@Component({ selector: 'app-training-job-queue', standalone: true, template: '' })
class StubJobQueue {}
@Component({ selector: 'app-system-monitor', standalone: true, template: '' })
class StubSystemMonitor {}

/**
 * B6b — video sample lightbox.
 *
 * Covers: `isVideoSample` extension detection, the `sampleMuted` autoplay
 * toggle, and the rendered lightbox swapping <video> in for video samples
 * (with a mute toggle) vs <img> for stills. Sample strip uses a placeholder
 * tile (play glyph) for videos since an <img> can't render an mp4/webm.
 */

const JOB_ID = 'job-1';

function makeJob(over: Partial<Job> = {}): Job {
    return {
        id: JOB_ID,
        plugin_id: 'wan',
        config: { definition_id: 'wan', lora_name: 'demo' },
        status: JobStatus.RUNNING,
        created_at: 0,
        logs: [],
        warnings: [],
        ...over,
    };
}

/**
 * Real JobsViewState (drives selectedJob via signals) + mocked heavy services
 * so the constructor's effects (sample/checkpoint/replay/log/sampling loads)
 * resolve synchronously and never hit the network.
 */
function setup(): { fixture: ComponentFixture<JobsScreen>; view: JobsViewState; comp: JobsScreen } {
    const jobService = {
        getJobSamples: vi.fn().mockReturnValue(of([])),
        getJobCheckpoints: vi.fn().mockReturnValue(of([])),
        getJobReplay: vi.fn().mockReturnValue(of({ loss: [], available: true })),
        getJobLogs: vi.fn().mockReturnValue(of([])),
        getSamplingStatus: vi.fn().mockReturnValue(of({ job_id: JOB_ID, sampling_paused: false })),
        getSamplingCadence: vi.fn().mockReturnValue(of({ job_id: JOB_ID, interval: 100, default_interval: 100 })),
        restartJob: vi.fn().mockReturnValue(of({ status: 'restarted', job_id: JOB_ID, fresh: false })),
        resumeFromCheckpoint: vi.fn().mockReturnValue(of(makeJob({ status: JobStatus.PENDING }))),
        stopJob: vi.fn().mockReturnValue(of({})),
        checkpointDownloadUrl: vi.fn().mockReturnValue('http://test/download'),
        checkpointZipDownloadUrl: vi.fn().mockReturnValue('http://test/zip'),
    };

    TestBed.configureTestingModule({
        imports: [JobsScreen],
        providers: [
            JobsViewState,
            { provide: JobService, useValue: jobService },
            { provide: JobStore, useValue: { loadAll: vi.fn().mockResolvedValue(undefined), loadHistory: vi.fn().mockResolvedValue(undefined) } },
            { provide: OverlayStore, useValue: { openModal: vi.fn() } },
            { provide: TrainingHandoffService, useValue: { set: vi.fn() } },
            { provide: ScopeStore, useValue: { setProject: vi.fn(), setGlobal: vi.fn() } },
            { provide: TemplateService, useValue: { createTrainingTemplate: vi.fn().mockReturnValue(of({})) } },
            { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn() } },
            { provide: RuntimeConfigService, useValue: { apiUrl: 'http://test' } },
            { provide: Router, useValue: { navigate: vi.fn() } },
            { provide: ResumeJobService, useValue: { open: vi.fn(), restart: vi.fn() } },
        ],
    });

    TestBed.overrideComponent(JobsScreen, {
        remove: { imports: [TrainingJobQueueComponent, SystemMonitorComponent] },
        add: { imports: [StubJobQueue, StubSystemMonitor] },
    });

    const fixture = TestBed.createComponent(JobsScreen);
    const view = TestBed.inject(JobsViewState);
    return { fixture, view, comp: fixture.componentInstance };
}

describe('JobsScreen.isVideoSample', () => {
    let comp: JobsScreen;
    beforeEach(() => {
        comp = setup().comp;
    });

    it('is true for .mp4 / .webm (case-insensitive)', () => {
        const c = comp as unknown as { isVideoSample: (f: string) => boolean };
        expect(c.isVideoSample('sample_step42.mp4')).toBe(true);
        expect(c.isVideoSample('clip.MP4')).toBe(true);
        expect(c.isVideoSample('out.webm')).toBe(true);
        expect(c.isVideoSample('out.WEBM')).toBe(true);
    });

    it('is false for stills and empty/missing', () => {
        const c = comp as unknown as { isVideoSample: (f: string | null | undefined) => boolean };
        expect(c.isVideoSample('sample.png')).toBe(false);
        expect(c.isVideoSample('sample.jpg')).toBe(false);
        expect(c.isVideoSample('sample.jpeg')).toBe(false);
        expect(c.isVideoSample('mp4-but-not-ext.png')).toBe(false);
        expect(c.isVideoSample('')).toBe(false);
        expect(c.isVideoSample(null)).toBe(false);
        expect(c.isVideoSample(undefined)).toBe(false);
    });
});

describe('JobsScreen.sampleMuted', () => {
    it('defaults muted and toggles', () => {
        const comp = setup().comp as unknown as {
            sampleMuted: () => boolean;
            toggleSampleMuted: () => void;
        };
        expect(comp.sampleMuted()).toBe(true);
        comp.toggleSampleMuted();
        expect(comp.sampleMuted()).toBe(false);
        comp.toggleSampleMuted();
        expect(comp.sampleMuted()).toBe(true);
    });

    it('openSample re-arms muted (autoplay must start muted)', () => {
        const comp = setup().comp as unknown as {
            sampleMuted: () => boolean;
            toggleSampleMuted: () => void;
            openSample: (s: { filename: string; step?: number }) => void;
        };
        comp.toggleSampleMuted();
        expect(comp.sampleMuted()).toBe(false);
        comp.openSample({ filename: 'clip.mp4', step: 10 });
        expect(comp.sampleMuted()).toBe(true);
    });
});

describe('JobsScreen lightbox rendering', () => {
    function open(filename: string): ComponentFixture<JobsScreen> {
        const { fixture, view, comp } = setup();
        const job = makeJob();
        view.activeJobs.set([job]);
        view.selectedId.set(JOB_ID);
        // Seed the sample list for the strip + nav, then open the lightbox.
        (comp as unknown as { samplesByJob: { set: (m: Map<string, unknown>) => void } })
            .samplesByJob.set(new Map([[JOB_ID, [{ filename, step: 7 }]]]));
        (comp as unknown as { openSample: (s: { filename: string; step?: number }) => void })
            .openSample({ filename, step: 7 });
        fixture.detectChanges();
        return fixture;
    }

    it('renders a <video> + mute toggle for an mp4 sample', () => {
        const el: HTMLElement = open('sample_step7.mp4').nativeElement;
        const video = el.querySelector('[data-testid="lightbox-video"]');
        expect(video).toBeTruthy();
        expect(video!.tagName.toLowerCase()).toBe('video');
        expect(el.querySelector('[data-testid="lightbox-img"]')).toBeNull();
        expect(el.querySelector('[data-testid="lightbox-mute"]')).toBeTruthy();
    });

    it('renders an <img> (no video, no mute toggle) for a png sample', () => {
        const el: HTMLElement = open('sample_step7.png').nativeElement;
        const img = el.querySelector('[data-testid="lightbox-img"]');
        expect(img).toBeTruthy();
        expect(img!.tagName.toLowerCase()).toBe('img');
        expect(el.querySelector('[data-testid="lightbox-video"]')).toBeNull();
        expect(el.querySelector('[data-testid="lightbox-mute"]')).toBeNull();
    });

    it('strip shows a video-thumb placeholder for video samples', () => {
        const { fixture, view, comp } = setup();
        view.activeJobs.set([makeJob()]);
        view.selectedId.set(JOB_ID);
        (comp as unknown as { samplesByJob: { set: (m: Map<string, unknown>) => void } })
            .samplesByJob.set(new Map([[JOB_ID, [{ filename: 'clip.mp4', step: 7 }]]]));
        fixture.detectChanges();
        const el: HTMLElement = fixture.nativeElement;
        expect(el.querySelector('[data-testid="sample-video-thumb"]')).toBeTruthy();
    });
});

/**
 * Tab re-focus refresh.
 *
 * Background tabs throttle the zoneless scheduler and the live WS log stream is
 * torn down at job completion, so samples/checkpoints written while the tab is
 * hidden never trigger the reactive refresh — they stay invisible until a full
 * page reload. Re-focusing the tab (visibilitychange → visible) must re-pull
 * the selected job's samples + checkpoints so freshly-written artifacts surface.
 */
describe('JobsScreen visibility refresh', () => {
    function setVisibility(state: 'visible' | 'hidden'): void {
        Object.defineProperty(document, 'visibilityState', { value: state, configurable: true });
    }
    afterEach(() => {
        // Drop the own-property shadow so the jsdom prototype getter is restored.
        delete (document as unknown as { visibilityState?: string }).visibilityState;
    });

    function svcOf(): { getJobSamples: ReturnType<typeof vi.fn>; getJobCheckpoints: ReturnType<typeof vi.fn> } {
        return TestBed.inject(JobService) as unknown as {
            getJobSamples: ReturnType<typeof vi.fn>;
            getJobCheckpoints: ReturnType<typeof vi.fn>;
        };
    }

    it('re-pulls samples + checkpoints for the selected job when the tab becomes visible', () => {
        const { fixture, view } = setup();
        view.activeJobs.set([makeJob()]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        // The select effect already fetched once on selection; reset counters so
        // we assert the refresh fires specifically on re-focus.
        const svc = svcOf();
        svc.getJobSamples.mockClear();
        svc.getJobCheckpoints.mockClear();

        setVisibility('visible');
        document.dispatchEvent(new Event('visibilitychange'));

        expect(svc.getJobSamples).toHaveBeenCalledWith(JOB_ID);
        expect(svc.getJobCheckpoints).toHaveBeenCalledWith(JOB_ID);
    });

    it('does nothing when the tab goes hidden', () => {
        const { fixture, view } = setup();
        view.activeJobs.set([makeJob()]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        const svc = svcOf();
        svc.getJobSamples.mockClear();
        svc.getJobCheckpoints.mockClear();

        setVisibility('hidden');
        document.dispatchEvent(new Event('visibilitychange'));

        expect(svc.getJobSamples).not.toHaveBeenCalled();
        expect(svc.getJobCheckpoints).not.toHaveBeenCalled();
    });

    it('does nothing when no job is selected', () => {
        const { fixture } = setup();
        fixture.detectChanges();
        const svc = svcOf();
        svc.getJobSamples.mockClear();
        svc.getJobCheckpoints.mockClear();

        setVisibility('visible');
        document.dispatchEvent(new Event('visibilitychange'));

        expect(svc.getJobSamples).not.toHaveBeenCalled();
        expect(svc.getJobCheckpoints).not.toHaveBeenCalled();
    });
});

describe('JobsScreen resume/restart buttons', () => {
    it('shows Resume (not Restart) for a stopped job with a resumable checkpoint', () => {
        const { fixture, view, comp } = setup();
        view.archivedJobs.set([makeJob({ status: JobStatus.STOPPED })]);
        view.selectedId.set(JOB_ID);
        // Seed a resumable checkpoint for the selected job.
        (comp as unknown as { checkpointsByJob: { set: (m: Map<string, unknown[]>) => void } })
            .checkpointsByJob.set(new Map([[JOB_ID, [{
                filename: 'l.safetensors', step: 500, is_final: false, size_bytes: 1,
                created_at: 0, resumable: true, checkpoint_dir: 'checkpoint-000500',
            }]]]));
        fixture.detectChanges();
        const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
        expect(text).toContain('Resume');
        expect(text).not.toContain('Restart fresh');
    });

    it('shows Restart for a stopped job with no resumable checkpoint', () => {
        const { fixture, view } = setup();
        view.archivedJobs.set([makeJob({ status: JobStatus.STOPPED })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
        expect(text).toContain('Restart');
    });

    it('openResumeDialog delegates to ResumeJobService with the job id and resumable checkpoints', () => {
        const { view, comp } = setup();
        const job = makeJob({ status: JobStatus.STOPPED });
        view.archivedJobs.set([job]);
        view.selectedId.set(JOB_ID);
        // Seed checkpoints: one resumable, one not.
        (comp as unknown as { checkpointsByJob: { set: (m: Map<string, unknown[]>) => void } })
            .checkpointsByJob.set(new Map([[JOB_ID, [
                { filename: 'a', step: 500, is_final: false, size_bytes: 1, created_at: 0, resumable: true, checkpoint_dir: 'checkpoint-000500' },
                { filename: 'b', step: 250, is_final: false, size_bytes: 1, created_at: 0, resumable: false, checkpoint_dir: null },
            ]]]));
        const svc = (comp as unknown as { resumeJobs: { open: ReturnType<typeof vi.fn> } }).resumeJobs;
        (comp as unknown as { openResumeDialog: () => void }).openResumeDialog();
        expect(svc.open).toHaveBeenCalledTimes(1);
        const [jobId, checkpoints] = svc.open.mock.calls[0];
        expect(jobId).toBe(JOB_ID);
        // Only the resumable checkpoint is passed.
        expect(checkpoints).toHaveLength(1);
        expect(checkpoints[0].checkpoint_dir).toBe('checkpoint-000500');
    });
});

/**
 * P4d — restart wrapper dedupe (F-ARCH-6) + prompt()/confirm() → typed modals.
 * Both restart entry points route through ResumeJobService.restart; the
 * destructive/save actions fire ONLY from the modal's confirm callback (the
 * old synchronous confirm()/prompt() guards are now async).
 */
describe('JobsScreen — restart delegation + typed dialogs (P4d)', () => {
    function selectJob(view: JobsViewState, fixture: ComponentFixture<JobsScreen>, over: Partial<Job> = {}) {
        view.archivedJobs.set([makeJob({ status: JobStatus.STOPPED, ...over })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
    }
    type Comp = Record<string, (...a: unknown[]) => void> & {
        resumeJobs: { restart: ReturnType<typeof vi.fn> };
    };

    it('restartJob() delegates to the single ResumeJobService.restart wrapper (wipe=false)', () => {
        const { fixture, view, comp } = setup();
        selectJob(view, fixture);
        const c = comp as unknown as Comp;
        c['restartJob']();
        expect(c.resumeJobs.restart).toHaveBeenCalledTimes(1);
        expect(c.resumeJobs.restart.mock.calls[0][0]).toBe(JOB_ID);
        expect(c.resumeJobs.restart.mock.calls[0][1]).toBe(false);
    });

    it('restartFresh() restarts (wipe=true) only after the confirm modal is confirmed', () => {
        const { fixture, view, comp } = setup();
        selectJob(view, fixture);
        const overlay = TestBed.inject(OverlayStore) as unknown as { openModal: ReturnType<typeof vi.fn> };
        const c = comp as unknown as Comp;
        c['restartFresh']();
        // Nothing happens until the user confirms.
        expect(c.resumeJobs.restart).not.toHaveBeenCalled();
        const [kind, data] = overlay.openModal.mock.calls.at(-1) as [string, { onConfirm: () => void }];
        expect(kind).toBe('confirm');
        data.onConfirm();
        expect(c.resumeJobs.restart).toHaveBeenCalledWith(JOB_ID, true, expect.any(Function));
    });

    it('restartFresh() captures the target job BEFORE opening the confirm — a selection change while the modal is open does not redirect the restart', () => {
        const OTHER_JOB_ID = 'job-2';
        const { fixture, view, comp } = setup();
        selectJob(view, fixture);
        view.archivedJobs.set([
            ...view.archivedJobs(),
            makeJob({ id: OTHER_JOB_ID, status: JobStatus.STOPPED }),
        ]);
        const overlay = TestBed.inject(OverlayStore) as unknown as { openModal: ReturnType<typeof vi.fn> };
        const c = comp as unknown as Comp;
        c['restartFresh']();
        // Simulate the user clicking a different job row while the confirm modal is still open.
        view.selectedId.set(OTHER_JOB_ID);
        fixture.detectChanges();

        const [, data] = overlay.openModal.mock.calls.at(-1) as [string, { onConfirm: () => void }];
        data.onConfirm();
        // The ORIGINALLY-selected job restarts, not whatever is selected now.
        expect(c.resumeJobs.restart).toHaveBeenCalledWith(JOB_ID, true, expect.any(Function));
    });

    it('stopJob() hard-stops only from the confirm modal callback', () => {
        const { fixture, view, comp } = setup();
        selectJob(view, fixture, { status: JobStatus.RUNNING });
        const overlay = TestBed.inject(OverlayStore) as unknown as { openModal: ReturnType<typeof vi.fn> };
        const jobService = TestBed.inject(JobService) as unknown as { stopJob: ReturnType<typeof vi.fn> };
        const c = comp as unknown as Comp;
        c['stopJob']();
        expect(jobService.stopJob).not.toHaveBeenCalled();
        const [kind, data] = overlay.openModal.mock.calls.at(-1) as [string, { onConfirm: () => void }];
        expect(kind).toBe('confirm');
        data.onConfirm();
        expect(jobService.stopJob).toHaveBeenCalledWith(JOB_ID);
    });

    it('saveAsTemplate() saves only from the input modal callback, with the entered name', () => {
        const { fixture, view, comp } = setup();
        selectJob(view, fixture);
        const overlay = TestBed.inject(OverlayStore) as unknown as { openModal: ReturnType<typeof vi.fn> };
        const templates = TestBed.inject(TemplateService) as unknown as { createTrainingTemplate: ReturnType<typeof vi.fn> };
        const c = comp as unknown as Comp;
        c['saveAsTemplate']();
        expect(templates.createTrainingTemplate).not.toHaveBeenCalled();
        const [kind, data] = overlay.openModal.mock.calls.at(-1) as [string, { onConfirm: (v: string) => void }];
        expect(kind).toBe('input');
        data.onConfirm('My Template');
        expect(templates.createTrainingTemplate).toHaveBeenCalledWith(
            expect.objectContaining({ name: 'My Template' }),
        );
    });
});
