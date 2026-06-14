import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Component } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { of } from 'rxjs';

import { JobsScreen } from './jobs-screen';
import { JobService, JobStatus, type Job } from '../../services/job';
import { JobStore } from '../../state/job.store';
import { JobsViewState } from '../../state/jobs-view.state';
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
    };

    TestBed.configureTestingModule({
        imports: [JobsScreen],
        providers: [
            JobsViewState,
            { provide: JobService, useValue: jobService },
            { provide: JobStore, useValue: { loadAll: vi.fn().mockResolvedValue(undefined), loadHistory: vi.fn().mockResolvedValue(undefined) } },
            { provide: TrainingHandoffService, useValue: { set: vi.fn() } },
            { provide: ScopeStore, useValue: { setProject: vi.fn(), setGlobal: vi.fn() } },
            { provide: TemplateService, useValue: { createTrainingTemplate: vi.fn().mockReturnValue(of({})) } },
            { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn() } },
            { provide: RuntimeConfigService, useValue: { apiUrl: 'http://test' } },
            { provide: Router, useValue: { navigate: vi.fn() } },
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
