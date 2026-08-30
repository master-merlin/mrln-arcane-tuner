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
function setup(adaptHistory: { events: unknown[] } = { events: [] }): {
    fixture: ComponentFixture<JobsScreen>;
    view: JobsViewState;
    comp: JobsScreen;
} {
    const jobService = {
        getJobAdaptiveHistory: vi
            .fn()
            .mockReturnValue(of({ modules: [], heat: {}, ...adaptHistory })),
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

describe('JobsScreen.isAudioSample', () => {
    let comp: JobsScreen;
    beforeEach(() => {
        comp = setup().comp;
    });

    it('is true for .wav/.flac/.ogg/.mp3/.opus (case-insensitive)', () => {
        const c = comp as unknown as { isAudioSample: (f: string) => boolean };
        expect(c.isAudioSample('sample_step42.wav')).toBe(true);
        expect(c.isAudioSample('clip.WAV')).toBe(true);
        expect(c.isAudioSample('out.flac')).toBe(true);
        expect(c.isAudioSample('out.OGG')).toBe(true);
        expect(c.isAudioSample('out.mp3')).toBe(true);
        expect(c.isAudioSample('out.opus')).toBe(true);
    });

    it('is false for stills, video, and empty/missing', () => {
        const c = comp as unknown as { isAudioSample: (f: string | null | undefined) => boolean };
        expect(c.isAudioSample('sample.png')).toBe(false);
        expect(c.isAudioSample('clip.mp4')).toBe(false);
        expect(c.isAudioSample('clip.webm')).toBe(false);
        expect(c.isAudioSample('wav-but-not-ext.png')).toBe(false);
        expect(c.isAudioSample('')).toBe(false);
        expect(c.isAudioSample(null)).toBe(false);
        expect(c.isAudioSample(undefined)).toBe(false);
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

    it('renders an <audio controls> (no mute toggle) for a wav sample', () => {
        const el: HTMLElement = open('sample_step7.wav').nativeElement;
        const audio = el.querySelector('[data-testid="lightbox-audio"]');
        expect(audio).toBeTruthy();
        expect(audio!.tagName.toLowerCase()).toBe('audio');
        expect(audio!.hasAttribute('controls')).toBe(true);
        expect(el.querySelector('[data-testid="lightbox-img"]')).toBeNull();
        expect(el.querySelector('[data-testid="lightbox-video"]')).toBeNull();
        expect(el.querySelector('[data-testid="lightbox-mute"]')).toBeNull();
    });

    it('strip shows an audio-thumb placeholder for audio samples', () => {
        const { fixture, view, comp } = setup();
        view.activeJobs.set([makeJob()]);
        view.selectedId.set(JOB_ID);
        (comp as unknown as { samplesByJob: { set: (m: Map<string, unknown>) => void } })
            .samplesByJob.set(new Map([[JOB_ID, [{ filename: 'sample.wav', step: 7 }]]]));
        fixture.detectChanges();
        const el: HTMLElement = fixture.nativeElement;
        expect(el.querySelector('[data-testid="sample-audio-thumb"]')).toBeTruthy();
        expect(el.querySelector('[data-testid="sample-video-thumb"]')).toBeNull();
    });
});

/**
 * Prompt attribution — multi-prompt runs (e.g. video + still preview) need each
 * sample marked with the prompt that generated it: a P<index> tag with the full
 * prompt as tooltip on the strip tile, and the prompt text in the lightbox.
 */
describe('JobsScreen sample prompt attribution', () => {
    const PROMPT = 'helicopter parked in a hangar';

    function seed(samples: unknown[]): { fixture: ComponentFixture<JobsScreen>; comp: JobsScreen } {
        const { fixture, view, comp } = setup();
        view.activeJobs.set([makeJob()]);
        view.selectedId.set(JOB_ID);
        (comp as unknown as { samplesByJob: { set: (m: Map<string, unknown>) => void } })
            .samplesByJob.set(new Map([[JOB_ID, samples]]));
        fixture.detectChanges();
        return { fixture, comp };
    }

    it('strip tile shows a P<index> tag carrying the prompt as tooltip', () => {
        const { fixture } = seed([{ filename: 'sample_01_step000050.mp4', step: 50, index: 1, prompt: PROMPT }]);
        const tag = (fixture.nativeElement as HTMLElement)
            .querySelector('[data-testid="sample-prompt-tag"]');
        expect(tag).toBeTruthy();
        expect(tag!.textContent).toContain('P1');
        expect(tag!.getAttribute('title')).toBe(PROMPT);
    });

    it('strip tile shows no tag when the sample has no prompt', () => {
        const { fixture } = seed([{ filename: 'sample_00_step000050.mp4', step: 50, index: 0 }]);
        expect((fixture.nativeElement as HTMLElement)
            .querySelector('[data-testid="sample-prompt-tag"]')).toBeNull();
    });

    it('lightbox shows the prompt text when present', () => {
        const sample = { filename: 'sample_01_step000050.mp4', step: 50, index: 1, prompt: PROMPT };
        const { fixture, comp } = seed([sample]);
        (comp as unknown as { openSample: (s: unknown) => void }).openSample(sample);
        fixture.detectChanges();
        const cap = (fixture.nativeElement as HTMLElement)
            .querySelector('[data-testid="lightbox-prompt"]');
        expect(cap).toBeTruthy();
        expect(cap!.textContent).toContain(PROMPT);
    });

    it('lightbox omits the prompt element when absent', () => {
        const sample = { filename: 'sample_00_step000050.mp4', step: 50, index: 0 };
        const { fixture, comp } = seed([sample]);
        (comp as unknown as { openSample: (s: unknown) => void }).openSample(sample);
        fixture.detectChanges();
        expect((fixture.nativeElement as HTMLElement)
            .querySelector('[data-testid="lightbox-prompt"]')).toBeNull();
    });

    it('prompt chip works identically for an audio (.wav) sample', () => {
        const { fixture } = seed([{ filename: 'sample_01_step000050.wav', step: 50, index: 1, prompt: PROMPT }]);
        const tag = (fixture.nativeElement as HTMLElement)
            .querySelector('[data-testid="sample-prompt-tag"]');
        expect(tag).toBeTruthy();
        expect(tag!.textContent).toContain('P1');
        expect(tag!.getAttribute('title')).toBe(PROMPT);
    });
});

/**
 * Lyrics — audio samples (ace_step15) additionally carry `lyrics` alongside
 * `prompt`. Shown in the lightbox under the prompt line when present.
 */
describe('JobsScreen sample lyrics attribution', () => {
    const PROMPT = 'upbeat synth pop';
    const LYRICS = 'verse one\nchorus';

    function seed(samples: unknown[]): { fixture: ComponentFixture<JobsScreen>; comp: JobsScreen } {
        const { fixture, view, comp } = setup();
        view.activeJobs.set([makeJob()]);
        view.selectedId.set(JOB_ID);
        (comp as unknown as { samplesByJob: { set: (m: Map<string, unknown>) => void } })
            .samplesByJob.set(new Map([[JOB_ID, samples]]));
        fixture.detectChanges();
        return { fixture, comp };
    }

    it('lightbox shows lyrics text under the prompt when present', () => {
        const sample = { filename: 'sample_00_step000050.wav', step: 50, index: 0, prompt: PROMPT, lyrics: LYRICS };
        const { fixture, comp } = seed([sample]);
        (comp as unknown as { openSample: (s: unknown) => void }).openSample(sample);
        fixture.detectChanges();
        const el: HTMLElement = fixture.nativeElement;
        const lyricsEl = el.querySelector('[data-testid="lightbox-lyrics"]');
        expect(lyricsEl).toBeTruthy();
        expect(lyricsEl!.textContent).toContain('verse one');
        // Prompt line still renders alongside lyrics.
        expect(el.querySelector('[data-testid="lightbox-prompt"]')!.textContent).toContain(PROMPT);
    });

    it('lightbox omits the lyrics element when absent (instrumental)', () => {
        const sample = { filename: 'sample_00_step000050.wav', step: 50, index: 0, prompt: 'instrumental lo-fi' };
        const { fixture, comp } = seed([sample]);
        (comp as unknown as { openSample: (s: unknown) => void }).openSample(sample);
        fixture.detectChanges();
        expect((fixture.nativeElement as HTMLElement)
            .querySelector('[data-testid="lightbox-lyrics"]')).toBeNull();
    });
});

/**
 * Sample grouping — with multiple prompts the strip can be viewed flat
 * ("by step": p0 s0, p1 s0, p0 s100, p1 s100 …) or grouped "by prompt"
 * (one row per prompt index, its steps in ascending order).
 */
describe('JobsScreen sample grouping', () => {
    const TWO_PROMPT_SAMPLES = [
        { filename: 'sample_00_step000100.mp4', step: 100, index: 0, prompt: 'video prompt' },
        { filename: 'sample_01_step000100.mp4', step: 100, index: 1, prompt: 'still prompt' },
        { filename: 'sample_00_step000000.mp4', step: 0, index: 0, prompt: 'video prompt' },
        { filename: 'sample_01_step000000.mp4', step: 0, index: 1, prompt: 'still prompt' },
    ];

    function seed(samples: unknown[]): { fixture: ComponentFixture<JobsScreen>; comp: JobsScreen } {
        const { fixture, view, comp } = setup();
        view.activeJobs.set([makeJob()]);
        view.selectedId.set(JOB_ID);
        (comp as unknown as { samplesByJob: { set: (m: Map<string, unknown>) => void } })
            .samplesByJob.set(new Map([[JOB_ID, samples]]));
        fixture.detectChanges();
        return { fixture, comp };
    }

    it('defaults to the flat by-step strip (no group rows) and offers the toggle', () => {
        const { fixture } = seed(TWO_PROMPT_SAMPLES);
        const el: HTMLElement = fixture.nativeElement;
        expect(el.querySelector('[data-testid="sample-prompt-group"]')).toBeNull();
        expect(el.querySelector('[data-testid="sample-group-by-prompt"]')).toBeTruthy();
        expect(el.querySelector('[data-testid="sample-group-by-step"]')).toBeTruthy();
    });

    it('hides the toggle when only one prompt index exists', () => {
        const { fixture } = seed([
            { filename: 'sample_00_step000100.mp4', step: 100, index: 0, prompt: 'video prompt' },
            { filename: 'sample_00_step000000.mp4', step: 0, index: 0, prompt: 'video prompt' },
        ]);
        expect((fixture.nativeElement as HTMLElement)
            .querySelector('[data-testid="sample-group-by-prompt"]')).toBeNull();
    });

    it('by-prompt renders one row per prompt with its steps ascending', () => {
        const { fixture } = seed(TWO_PROMPT_SAMPLES);
        const el: HTMLElement = fixture.nativeElement;
        (el.querySelector('[data-testid="sample-group-by-prompt"]') as HTMLButtonElement).click();
        fixture.detectChanges();

        const rows = el.querySelectorAll('[data-testid="sample-prompt-group"]');
        expect(rows.length).toBe(2);
        expect(rows[0].textContent).toContain('P0');
        expect(rows[1].textContent).toContain('P1');
        // Each row holds only its prompt's samples, steps ascending.
        const capsRow0 = [...rows[0].querySelectorAll('.sample-cap')].map((c) => c.textContent?.trim());
        expect(capsRow0.length).toBe(2);
        expect(capsRow0[0]).toContain('step 0');
        expect(capsRow0[1]).toContain('step 100');
    });

    it('switching back to by-step restores the flat strip', () => {
        const { fixture } = seed(TWO_PROMPT_SAMPLES);
        const el: HTMLElement = fixture.nativeElement;
        (el.querySelector('[data-testid="sample-group-by-prompt"]') as HTMLButtonElement).click();
        fixture.detectChanges();
        (el.querySelector('[data-testid="sample-group-by-step"]') as HTMLButtonElement).click();
        fixture.detectChanges();
        expect(el.querySelector('[data-testid="sample-prompt-group"]')).toBeNull();
    });

    it('by-prompt grouping works identically for audio (.wav) samples', () => {
        const { fixture } = seed([
            { filename: 'sample_00_step000100.wav', step: 100, index: 0, prompt: 'synth pop' },
            { filename: 'sample_01_step000100.wav', step: 100, index: 1, prompt: 'lo-fi beat' },
            { filename: 'sample_00_step000000.wav', step: 0, index: 0, prompt: 'synth pop' },
            { filename: 'sample_01_step000000.wav', step: 0, index: 1, prompt: 'lo-fi beat' },
        ]);
        const el: HTMLElement = fixture.nativeElement;
        (el.querySelector('[data-testid="sample-group-by-prompt"]') as HTMLButtonElement).click();
        fixture.detectChanges();

        const rows = el.querySelectorAll('[data-testid="sample-prompt-group"]');
        expect(rows.length).toBe(2);
        expect(rows[0].textContent).toContain('P0');
        expect(rows[1].textContent).toContain('P1');
        expect(el.querySelectorAll('[data-testid="sample-audio-thumb"]').length).toBe(4);
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

/**
 * T4 — auto-select the running (or most recently active) job.
 *
 * The center detail pane must be useful on load: when nothing is explicitly
 * selected it auto-selects the running job (else the most recently active job)
 * so it never shows the empty state while the queue actually has jobs. An
 * explicit user selection is never overridden; the true empty state survives
 * only for a genuinely empty queue.
 */
describe('JobsScreen auto-select (T4)', () => {
    it('auto-selects the running job when nothing is explicitly selected', () => {
        const { fixture, view } = setup();
        view.activeJobs.set([
            makeJob({ id: 'pending-1', status: JobStatus.PENDING }),
            makeJob({ id: 'running-1', status: JobStatus.RUNNING }),
        ]);
        fixture.detectChanges();
        expect(view.selectedId()).toBe('running-1');
        expect(view.selectedJob()?.id).toBe('running-1');
    });

    it('does not override an explicit user selection', () => {
        const { fixture, view } = setup();
        view.selectedId.set('pending-1'); // explicit choice made first
        view.activeJobs.set([
            makeJob({ id: 'pending-1', status: JobStatus.PENDING }),
            makeJob({ id: 'running-1', status: JobStatus.RUNNING }),
        ]);
        fixture.detectChanges();
        expect(view.selectedId()).toBe('pending-1');
    });

    it('falls back to the most recently active job when none is running', () => {
        const { fixture, view } = setup();
        view.archivedJobs.set([
            makeJob({ id: 'old', status: JobStatus.COMPLETED, created_at: 100, finished_at: 100 }),
            makeJob({ id: 'new', status: JobStatus.COMPLETED, created_at: 200, finished_at: 200 }),
        ]);
        fixture.detectChanges();
        expect(view.selectedId()).toBe('new');
    });

    it('keeps the empty state ONLY when there are zero jobs', () => {
        const { fixture, view } = setup();
        fixture.detectChanges();
        expect(view.selectedId()).toBeNull();
        const el = fixture.nativeElement as HTMLElement;
        expect(el.querySelector('[data-testid="job-detail-empty"]')).toBeTruthy();
    });
});

/**
 * T6 — failed-job affordances.
 *
 * On opening a FAILED job the Log section auto-expands (default is collapsed),
 * and the failure banner exposes a "Copy error" action and a "View full log"
 * jump. Copy error must not fire until wired to the clipboard.
 */
describe('JobsScreen failed-job affordances (T6)', () => {
    afterEach(() => {
        delete (navigator as unknown as { clipboard?: unknown }).clipboard;
    });

    it('auto-expands the Log section when a FAILED job is opened', () => {
        const { fixture, view, comp } = setup();
        view.archivedJobs.set([makeJob({ status: JobStatus.FAILED, error: 'boom' })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        expect((comp as unknown as { expanded: () => { log: boolean } }).expanded().log).toBe(true);
    });

    it('does not auto-expand the Log for a non-failed job', () => {
        const { fixture, view, comp } = setup();
        view.activeJobs.set([makeJob({ status: JobStatus.RUNNING })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        expect((comp as unknown as { expanded: () => { log: boolean } }).expanded().log).toBe(false);
    });

    it('renders Copy error + View full log affordances in the failure banner', () => {
        const { fixture, view } = setup();
        view.archivedJobs.set([makeJob({ status: JobStatus.FAILED, error: 'kaboom' })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        const el = fixture.nativeElement as HTMLElement;
        expect(el.querySelector('[data-testid="copy-error"]')).toBeTruthy();
        expect(el.querySelector('[data-testid="view-full-log"]')).toBeTruthy();
    });

    it('copyError copies the failure message to the clipboard', () => {
        const writeText = vi.fn().mockResolvedValue(undefined);
        Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
        const { fixture, view, comp } = setup();
        view.archivedJobs.set([makeJob({ status: JobStatus.FAILED, error: 'kaboom' })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        (comp as unknown as { copyError: () => void }).copyError();
        expect(writeText).toHaveBeenCalledWith('kaboom');
    });
});

/**
 * Lightbox arrow navigation must follow the ORDER THE STRIP SHOWS: flat
 * newest-first in "by step", per-prompt rows (steps ascending) in "by prompt".
 * Regression (UAT 2026-07-14): nav always walked the flat step order, so in
 * by-prompt view the arrows flipped between prompts instead of stepping
 * through one prompt's timeline.
 */
describe('JobsScreen lightbox navigation follows grouping', () => {
    const TWO_PROMPT_SAMPLES = [
        { filename: 'sample_00_step000100.png', step: 100, index: 0, prompt: 'video prompt' },
        { filename: 'sample_01_step000100.png', step: 100, index: 1, prompt: 'still prompt' },
        { filename: 'sample_00_step000000.png', step: 0, index: 0, prompt: 'video prompt' },
        { filename: 'sample_01_step000000.png', step: 0, index: 1, prompt: 'still prompt' },
    ];

    type Sample = { filename: string; step?: number };
    type CompAccess = {
        samplesByJob: { set: (m: Map<string, unknown>) => void };
        sampleGrouping: { set: (v: 'step' | 'prompt') => void };
        openSample: (s: Sample) => void;
        navSample: (dir: -1 | 1) => void;
        sampleModal: () => Sample | null;
    };

    function seed(): { comp: CompAccess } {
        const { fixture, view, comp } = setup();
        view.activeJobs.set([makeJob()]);
        view.selectedId.set(JOB_ID);
        const c = comp as unknown as CompAccess;
        c.samplesByJob.set(new Map([[JOB_ID, TWO_PROMPT_SAMPLES]]));
        fixture.detectChanges();
        return { comp: c };
    }

    it('by-prompt: arrows walk one prompt timeline (steps), not across prompts', () => {
        const { comp } = seed();
        comp.sampleGrouping.set('prompt');
        comp.openSample(TWO_PROMPT_SAMPLES[2]); // p0 step 0
        comp.navSample(1);
        expect(comp.sampleModal()?.filename).toBe('sample_00_step000100.png'); // p0 step 100
        comp.navSample(1);
        expect(comp.sampleModal()?.filename).toBe('sample_01_step000000.png'); // next row: p1 step 0
        comp.navSample(-1);
        expect(comp.sampleModal()?.filename).toBe('sample_00_step000100.png');
    });

    it('by-step: arrows keep the flat strip order (regression pin)', () => {
        const { comp } = seed();
        comp.sampleGrouping.set('step');
        comp.openSample(TWO_PROMPT_SAMPLES[0]); // first in flat order
        comp.navSample(1);
        expect(comp.sampleModal()?.filename).toBe('sample_01_step000100.png');
        comp.navSample(1);
        expect(comp.sampleModal()?.filename).toBe('sample_00_step000000.png');
    });
});

/**
 * T11 — adaptive layer targeting status chip.
 *
 * The chip surfaces the newest `{"adapt": {...}}` event in the LIVE log
 * stream (job.logs is live-session-only; the durable history endpoint is a
 * later task). It must render only when an adapt event exists, and must
 * disappear again when there is none.
 */
describe('JobsScreen adaptive status chip (T11)', () => {
    const adaptLine = (data: object) => JSON.stringify({ adapt: data });

    it('renders "5/8 layers" when the log stream has an adapt event', () => {
        const { fixture, view } = setup();
        view.activeJobs.set([
            makeJob({
                logs: [adaptLine({ step: 20, kind: 'narrow', active_count: 5, total_count: 8 })],
            }),
        ]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        const chip = fixture.nativeElement.querySelector('[data-testid="adapt-chip"]');
        expect(chip).toBeTruthy();
        expect(chip!.textContent).toContain('5/8 layers');
    });

    it('does not render the chip when there is no adapt event', () => {
        const { fixture, view } = setup();
        view.activeJobs.set([makeJob({ logs: [] })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="adapt-chip"]')).toBeNull();
    });

    /**
     * UAT-3.2 — the chip used to sit in a row of its own ABOVE the curves card,
     * where it read as a separate section rather than as a property of the
     * curve. The pair of tests above passed either way, which is why this one
     * pins the position and not just the existence.
     */
    it('sits inside the curves card head, immediately after the live chip', () => {
        const { fixture, view } = setup();
        view.activeJobs.set([
            makeJob({
                status: JobStatus.RUNNING,
                logs: [adaptLine({ step: 20, kind: 'narrow', active_count: 5, total_count: 8 })],
            }),
        ]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();

        const chip = fixture.nativeElement.querySelector('[data-testid="adapt-chip"]') as HTMLElement;
        expect(chip).toBeTruthy();

        // In the curves head, not floating above the card.
        const head = chip.closest('.curves-head');
        expect(head).toBeTruthy();
        expect(head!.querySelector('.card-title')!.textContent).toContain('Training Curves');

        // Same height as "live" is a CSS concern; what the DOM can pin is that
        // it wears the same size class and follows the live chip in order.
        expect(chip.classList.contains('head-chip')).toBe(true);
        const chips = [...head!.querySelectorAll('.chip')] as HTMLElement[];
        const live = chips.find(c => c.textContent!.trim().endsWith('live'));
        expect(live).toBeTruthy();
        expect(chips.indexOf(chip)).toBe(chips.indexOf(live!) + 1);
    });
});

/**
 * LANE-35 — the chip must survive log-buffer eviction.
 *
 * `job.logs` is a bounded 1000-entry FIFO (`job_manager.py` appends then
 * `pop(0)`). Adapt events are emitted only at adaptation moments while
 * ordinary step lines stream into that same buffer continuously, so a rare
 * event ages out BY CONSTRUCTION — the chip was guaranteed to vanish on any
 * sufficiently long run, which is what the user saw past step 1000.
 *
 * The three T11 tests above all pass with the log-scan-only derivation, and
 * so does the UAT-3.2 position pin: the chip was never wrong in POSITION, it
 * was wrong in EXISTENCE, and a DOM-order test cannot see that. These tests
 * reproduce the eviction — a full window with no `"adapt"` line left in it —
 * and are the guard that fails when the chip is absent for a run that HAS
 * adapted.
 */
describe('JobsScreen adaptive chip survives log eviction (LANE-35)', () => {
    const adaptLine = (data: object) => JSON.stringify({ adapt: data });

    /**
     * A saturated 1000-entry FIFO of ordinary step lines — no adapt event left
     * in the window. `loss` is deliberately omitted: it is orthogonal to the
     * property under test (the absence of an `"adapt"` line), and 1000 loss
     * points would drive a real uPlot draw that throws in jsdom.
     */
    const evictedLogs = (fromStep: number) =>
        Array.from({ length: 1000 }, (_, i) =>
            JSON.stringify({ step: fromStep + i, status: 'training', step_time: 2.7 }),
        );

    it('renders the durable n/m after the adapt event has aged out of job.logs', () => {
        const { fixture, view } = setup({
            events: [
                { step: 1450, kind: 'narrow', active_count: 181, total_count: 224 },
                { step: 1700, kind: 'narrow', active_count: 176, total_count: 224 },
            ],
        });
        // Window covers steps 2000..2999: both adapt events have been evicted.
        view.activeJobs.set([
            makeJob({ status: JobStatus.RUNNING, logs: evictedLogs(2000) }),
        ]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();

        const chip = fixture.nativeElement.querySelector('[data-testid="adapt-chip"]');
        expect(chip).toBeTruthy();
        expect(chip!.textContent).toContain('176/224 layers');
    });

    it('a live event newer than the durable record wins (higher step)', () => {
        const { fixture, view } = setup({
            events: [{ step: 1450, kind: 'narrow', active_count: 181, total_count: 224 }],
        });
        view.activeJobs.set([
            makeJob({
                status: JobStatus.RUNNING,
                logs: [adaptLine({ step: 1700, kind: 'narrow', active_count: 176, total_count: 224 })],
            }),
        ]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();

        const chip = fixture.nativeElement.querySelector('[data-testid="adapt-chip"]');
        expect(chip!.textContent).toContain('176/224 layers');
    });

    it('a latched live event is not lost when that line later evicts', () => {
        const { fixture, view } = setup({ events: [] }); // durable served nothing yet
        view.activeJobs.set([
            makeJob({
                status: JobStatus.RUNNING,
                logs: [adaptLine({ step: 1700, kind: 'narrow', active_count: 176, total_count: 224 })],
            }),
        ]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="adapt-chip"]')).toBeTruthy();

        // The FIFO rolls past it — same job, same component, no refetch.
        view.activeJobs.set([
            makeJob({ status: JobStatus.RUNNING, logs: evictedLogs(2000) }),
        ]);
        fixture.detectChanges();

        const chip = fixture.nativeElement.querySelector('[data-testid="adapt-chip"]');
        expect(chip).toBeTruthy();
        expect(chip!.textContent).toContain('176/224 layers');
    });

    it('still renders nothing for a run that never adapted', () => {
        const { fixture, view } = setup({ events: [] });
        view.activeJobs.set([
            makeJob({ status: JobStatus.RUNNING, logs: evictedLogs(2000) }),
        ]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="adapt-chip"]')).toBeNull();
    });
});
