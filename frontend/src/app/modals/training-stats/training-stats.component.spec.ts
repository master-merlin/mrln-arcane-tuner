import { TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';
import { signal } from '@angular/core';
import { vi } from 'vitest';
import { TrainingStatsModalComponent } from './training-stats.component';
import { JobService, type TrainingStats } from '../../services/job';
import { ProjectService } from '../../services/project.service';
import { OverlayStore } from '../../state/overlay.store';

export function makeStats(over: Partial<TrainingStats> = {}): TrainingStats {
    return {
        total_jobs: 5, completed: 3, failed: 1, stopped: 1, running: 0, paused: 0,
        success_rate: 60.0, total_steps: 5000, total_runtime_sec: 7200,
        total_training_sec: 6000, avg_steps: 1666, avg_loss: 0.12, avg_min_loss: 0.09,
        avg_step_time_sec: 0.45, avg_runtime_sec: 2400,
        optimizers: [{ name: 'adamw', count: 5 }], unique_datasets: 3,
        last_job: { lora_name: 'x', definition_id: 'flux', status: 'completed', created_at: 1 },
        activity: [{ week_start: '2026-07-13', completed: 3, failed: 1, stopped: 1, other: 0 }],
        gpu_hours: 1.67, overhead_pct: 16.7, lora_count: 3, lora_bytes: 3_000_000_000,
        lora_on_disk: 2, lora_size_known: 3,
        checkpoint_count: 12,
        families: [{ id: 'flux', count: 5, completed: 3, success_rate: 60.0, avg_step_time: 0.45, best_loss: 0.08 }],
        loss_histogram: { edges: [0.05, 0.1], counts: [3] },
        hyperparams: { optimizer_type: [{ value: 'adamw', count: 5 }], network_rank: [], lr_scheduler: [], timestep_sampling: [], quantization: [], mixed_precision: [], ema_enabled: [{ value: 'off', count: 5 }], batch_size: [] },
        resume_rate: 20.0,
        top_datasets: [{ name: 'ds1', count: 4 }],
        records: {
            longest_run: { job_id: 'j1', lora_name: 'x', definition_id: 'flux', value: 3600 },
            most_steps: { job_id: 'j1', lora_name: 'x', definition_id: 'flux', value: 2000 },
            best_loss: { job_id: 'j2', lora_name: 'y', definition_id: 'flux', value: 0.08 },
        },
        ...over,
    };
}

describe('TrainingStatsModalComponent', () => {
    let stats$: Subject<TrainingStats>;
    let getTrainingStats: ReturnType<typeof vi.fn>;
    let recompute$: Subject<unknown>;
    let recomputeStats: ReturnType<typeof vi.fn>;
    let runs$: Subject<unknown>;
    let listFamilyRuns: ReturnType<typeof vi.fn>;
    let adaptive$: Subject<unknown>;
    let getJobAdaptiveHistory: ReturnType<typeof vi.fn>;
    let metrics$: Subject<unknown>;
    let getJobMetrics: ReturnType<typeof vi.fn>;

    function setup() {
        stats$ = new Subject<TrainingStats>();
        getTrainingStats = vi.fn().mockReturnValue(stats$.asObservable());
        recompute$ = new Subject<unknown>();
        recomputeStats = vi.fn().mockReturnValue(recompute$.asObservable());
        runs$ = new Subject<unknown>();
        listFamilyRuns = vi.fn().mockImplementation(() => runs$.asObservable());
        adaptive$ = new Subject<unknown>();
        getJobAdaptiveHistory = vi.fn().mockImplementation(() => adaptive$.asObservable());
        metrics$ = new Subject<unknown>();
        getJobMetrics = vi.fn().mockImplementation(() => metrics$.asObservable());
        TestBed.configureTestingModule({
            imports: [TrainingStatsModalComponent],
            providers: [
                {
                    provide: JobService,
                    useValue: {
                        getTrainingStats, recomputeStats, listFamilyRuns,
                        getJobAdaptiveHistory, getJobMetrics,
                    },
                },
                { provide: ProjectService, useValue: { allProjects: signal([{ id: 'p1', name: 'P1' }]) } },
                { provide: OverlayStore, useValue: { topModal: () => undefined, closeModal: vi.fn() } },
            ],
        });
        const fixture = TestBed.createComponent(TrainingStatsModalComponent);
        fixture.detectChanges();
        return fixture;
    }

    function openTab(fixture: ReturnType<typeof setup>, index: number) {
        const tabs = fixture.nativeElement.querySelectorAll('[data-testid="stats-tabs"] .tab');
        (tabs[index] as HTMLButtonElement).click();
        fixture.detectChanges();
    }

    /** Drive to "family expanded, one run row visible" — the Adaptive
     *  section's entry point (a specific run's job id, the only per-job
     *  identity this cross-job modal has in scope). */
    function expandOneRun(fixture: ReturnType<typeof setup>, jobId = 'j1') {
        openTab(fixture, 1);
        (fixture.nativeElement.querySelector('[data-testid="stats-family-row"]') as HTMLElement).click();
        fixture.detectChanges();
        runs$.next([
            { id: jobId, lora_name: 'x_lora', status: 'completed', created_at: 1752624000,
              completed_steps: 1500, avg_step_time: 1.70523, min_loss: 0.005319 },
        ]);
        runs$.complete();
        fixture.detectChanges();
        (fixture.nativeElement.querySelector('[data-testid="stats-run-row"]') as HTMLElement).click();
        fixture.detectChanges();
    }

    it('fetches global stats on open and renders KPI tiles', () => {
        const fixture = setup();
        expect(getTrainingStats).toHaveBeenCalledWith('all');
        stats$.next(makeStats());
        stats$.complete();
        fixture.detectChanges();
        const el: HTMLElement = fixture.nativeElement;
        expect(el.querySelector('[data-testid="stats-kpi-total"]')?.textContent).toContain('5');
        expect(el.querySelector('[data-testid="stats-kpi-success"]')?.textContent).toContain('60');
    });

    it('refetches with the selected project id', () => {
        const fixture = setup();
        stats$.next(makeStats()); stats$.complete();
        fixture.detectChanges();
        stats$ = new Subject(); getTrainingStats.mockReturnValue(stats$.asObservable());
        const sel: HTMLSelectElement = fixture.nativeElement.querySelector('[data-testid="stats-project-filter"]');
        sel.value = sel.options[1].value; // 'p1'
        sel.dispatchEvent(new Event('change'));
        fixture.detectChanges();
        expect(getTrainingStats).toHaveBeenLastCalledWith('p1');
    });

    it('shows the empty state when the scope has no jobs', () => {
        const fixture = setup();
        stats$.next(makeStats({ total_jobs: 0 })); stats$.complete();
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="stats-empty"]')).toBeTruthy();
    });

    it('ignores a stale response that resolves after a newer filter change', () => {
        const fixture = setup();
        // initial (S1) request left pending — do not resolve yet

        const stats2$ = new Subject<TrainingStats>();
        getTrainingStats.mockReturnValue(stats2$.asObservable());
        const sel: HTMLSelectElement = fixture.nativeElement.querySelector('[data-testid="stats-project-filter"]');
        sel.value = sel.options[1].value; // 'p1'
        sel.dispatchEvent(new Event('change'));
        fixture.detectChanges();

        // newer request (S2) resolves first
        stats2$.next(makeStats({ total_jobs: 7 }));
        fixture.detectChanges();

        // stale request (S1) resolves late
        stats$.next(makeStats({ total_jobs: 5 }));
        fixture.detectChanges();

        const el: HTMLElement = fixture.nativeElement;
        expect(el.querySelector('[data-testid="stats-kpi-total"]')?.textContent).toContain('7');
        expect(el.querySelector('[data-testid="stats-kpi-total"]')?.textContent).not.toContain('5');
    });

    it('renders three tabs with Activity active by default; KPI row stays visible', () => {
        const fixture = setup();
        stats$.next(makeStats()); stats$.complete();
        fixture.detectChanges();
        const el: HTMLElement = fixture.nativeElement;
        const tabs = el.querySelectorAll('[data-testid="stats-tabs"] .tab');
        expect(Array.from(tabs).map(t => t.textContent?.trim()))
            .toEqual(['Activity', 'Quality & Families', 'Config & Data']);
        expect(tabs[0].classList.contains('active')).toBe(true);
        expect(el.querySelector('[data-testid="stats-kpi-total"]')).toBeTruthy();
        expect(el.querySelector('app-stats-uplot')).toBeTruthy();           // activity chart
        expect(el.querySelector('[data-testid="stats-family-row"]')).toBeFalsy(); // other tab
    });

    it('switches sections when a tab is clicked, keeping the KPI row', () => {
        const fixture = setup();
        stats$.next(makeStats()); stats$.complete();
        fixture.detectChanges();
        openTab(fixture, 1);
        const el: HTMLElement = fixture.nativeElement;
        expect(el.querySelector('[data-testid="stats-family-row"]')).toBeTruthy();
        expect(el.querySelector('[data-testid="stats-kpi-total"]')).toBeTruthy();
        openTab(fixture, 2);
        expect(el.querySelector('[data-testid="stats-hp-row"]')).toBeTruthy();
        expect(el.querySelector('[data-testid="stats-datasets"]')).toBeTruthy();
        expect(el.querySelector('[data-testid="stats-family-row"]')).toBeFalsy();
    });

    it('renders the activity section when there is activity, and the histogram fallback note when not', () => {
        const fixture = setup();
        stats$.next(makeStats({ loss_histogram: { edges: [], counts: [] } }));
        stats$.complete();
        fixture.detectChanges();
        const el: HTMLElement = fixture.nativeElement;
        expect(el.querySelector('app-stats-uplot')).toBeTruthy();   // activity chart host
        openTab(fixture, 1);
        expect(el.textContent).toContain('Not enough completed runs');
    });

    it('renders the family table with success-rate bars', () => {
        const fixture = setup();
        stats$.next(makeStats()); stats$.complete();
        fixture.detectChanges();
        openTab(fixture, 1);
        const rows = fixture.nativeElement.querySelectorAll('[data-testid="stats-family-row"]');
        expect(rows.length).toBe(1);
        expect(rows[0].textContent).toContain('flux');
        expect(rows[0].textContent).toContain('60');
    });

    it('renders hyperparameter bars only for populated dimensions', () => {
        const fixture = setup();
        stats$.next(makeStats()); stats$.complete();
        fixture.detectChanges();
        openTab(fixture, 2);
        const bars = fixture.nativeElement.querySelectorAll('[data-testid="stats-hp-row"]');
        expect(bars.length).toBe(2); // optimizer_type + ema_enabled populated in makeStats
    });

    it('shows the true best loss (not the average of per-job minima) in the Quality tile', () => {
        const fixture = setup();
        stats$.next(makeStats()); stats$.complete();
        fixture.detectChanges();
        openTab(fixture, 1);
        const el: HTMLElement = fixture.nativeElement;
        const tile = el.querySelector('[data-testid="stats-kpi-best-loss"]');
        expect(tile?.textContent).toContain('0.08');
        expect(tile?.textContent).not.toContain('0.09');
    });

    it('renders records and top datasets', () => {
        const fixture = setup();
        stats$.next(makeStats()); stats$.complete();
        fixture.detectChanges();
        openTab(fixture, 2);
        const el: HTMLElement = fixture.nativeElement;
        expect(el.querySelector('[data-testid="stats-records"]')?.textContent).toContain('y'); // best-loss lora
        expect(el.querySelector('[data-testid="stats-datasets"]')?.textContent).toContain('ds1');
    });

    it('gives the Total steps tile a warning accent', () => {
        const fixture = setup();
        stats$.next(makeStats()); stats$.complete();
        fixture.detectChanges();
        const steps = fixture.nativeElement.querySelector('[data-testid="stats-kpi-steps"]');
        expect(steps?.querySelector('.kpi-accent.warning')).toBeTruthy();
    });

    it('LoRAs tile shows a compact sub-line with the full detail as hover title', () => {
        const fixture = setup();
        stats$.next(makeStats()); stats$.complete();
        fixture.detectChanges();
        const tile = fixture.nativeElement.querySelector('[data-testid="stats-kpi-loras"]');
        expect(tile?.querySelector('[data-testid="kpi-tile-value"]')?.textContent).toContain('3');
        expect(tile?.textContent).toContain('2 on disk · 2.8 GB'); // 3e9 bytes, 1-decimal face
        expect(tile?.textContent).not.toContain('sized');          // coverage lives in the title
        expect(tile?.textContent).not.toContain('checkpoints');    // so does the ckpt count
        expect(tile?.getAttribute('title')).toBe('2 on disk · 2.79 GB · 12 checkpoints');
    });

    it('LoRAs tile keeps partial size coverage in the hover title only', () => {
        const fixture = setup();
        stats$.next(makeStats({ lora_count: 10, lora_size_known: 4 }));
        stats$.complete();
        fixture.detectChanges();
        const tile = fixture.nativeElement.querySelector('[data-testid="stats-kpi-loras"]');
        expect(tile?.textContent).not.toContain('sized');
        expect(tile?.getAttribute('title')).toBe('2 on disk · 2.79 GB (4/10 sized) · 12 checkpoints');
    });

    it('Total-jobs tile drops zero-count outcomes from the sub-line, full detail in title', () => {
        const fixture = setup();
        stats$.next(makeStats({ failed: 0, stopped: 2 })); stats$.complete();
        fixture.detectChanges();
        const tile = fixture.nativeElement.querySelector('[data-testid="stats-kpi-total"]');
        expect(tile?.textContent).toContain('3 done · 2 stopped');
        expect(tile?.textContent).not.toContain('failed');
        expect(tile?.getAttribute('title')).toBe('3 done · 0 failed · 2 stopped');
    });

    it('expands a family row into a per-run table with its own header (no avg columns)', () => {
        const fixture = setup();
        stats$.next(makeStats()); stats$.complete();
        fixture.detectChanges();
        openTab(fixture, 1);
        (fixture.nativeElement.querySelector('[data-testid="stats-family-row"]') as HTMLElement).click();
        fixture.detectChanges();
        expect(listFamilyRuns).toHaveBeenCalledWith('flux', 'all');

        runs$.next([
            { id: 'j1', lora_name: 'x_lora', status: 'completed', created_at: 1752624000,
              completed_steps: 1500, avg_step_time: 1.70523, min_loss: 0.005319 },
            { id: 'j2', lora_name: 'y_lora', status: 'failed', created_at: 1752537600, completed_steps: 0 },
        ]);
        runs$.complete();
        fixture.detectChanges();

        const head = fixture.nativeElement.querySelector('[data-testid="stats-run-head"]');
        expect(head?.textContent).toContain('step time');
        expect(head?.textContent?.toLowerCase()).not.toContain('avg'); // per-run values, not averages
        const rows = fixture.nativeElement.querySelectorAll('[data-testid="stats-run-row"]');
        expect(rows.length).toBe(2);
        expect(rows[0].textContent).toContain('x_lora');
        expect(rows[0].textContent).toContain('1.705s');
        expect(rows[0].textContent).toContain('0.005319');
        expect(rows[1].textContent).toContain('—'); // failed run: no step time / loss
    });

    it('collapses an expanded family on second click', () => {
        const fixture = setup();
        stats$.next(makeStats()); stats$.complete();
        fixture.detectChanges();
        openTab(fixture, 1);
        const row = fixture.nativeElement.querySelector('[data-testid="stats-family-row"]') as HTMLElement;
        row.click();
        runs$.next([]); runs$.complete();
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="stats-family-runs"]')).toBeTruthy();
        row.click();
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="stats-family-runs"]')).toBeFalsy();
    });

    it('reconcile button triggers the disk backfill and refetches stats on success', () => {
        const fixture = setup();
        stats$.next(makeStats()); stats$.complete();
        fixture.detectChanges();
        openTab(fixture, 2);
        const btn: HTMLButtonElement = fixture.nativeElement.querySelector('[data-testid="stats-reconcile"]');
        btn.click();
        fixture.detectChanges();
        expect(recomputeStats).toHaveBeenCalledTimes(1);
        expect(btn.disabled).toBe(true);            // busy while pending

        stats$ = new Subject(); getTrainingStats.mockReturnValue(stats$.asObservable());
        recompute$.next({ rows_updated: 5 }); recompute$.complete();
        fixture.detectChanges();
        expect(getTrainingStats).toHaveBeenCalledTimes(2);  // refetched
    });

    it('reconcile failure shows an inline error and re-enables the button', () => {
        const fixture = setup();
        stats$.next(makeStats()); stats$.complete();
        fixture.detectChanges();
        openTab(fixture, 2);
        const btn: HTMLButtonElement = fixture.nativeElement.querySelector('[data-testid="stats-reconcile"]');
        btn.click();
        recompute$.error(new Error('boom'));
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="stats-reconcile-error"]')).toBeTruthy();
        expect(btn.disabled).toBe(false);
    });

    // ── Adaptive section (Task 12) ───────────────────────────────────────
    describe('Adaptive section', () => {
        it('fetches a clicked run\'s adaptive history + metrics by job id', () => {
            const fixture = setup();
            stats$.next(makeStats()); stats$.complete();
            fixture.detectChanges();
            expandOneRun(fixture, 'j1');
            expect(getJobAdaptiveHistory).toHaveBeenCalledWith('j1');
            expect(getJobMetrics).toHaveBeenCalledWith('j1');
        });

        it('shows a caret affordance on the run row that flips on expand/collapse', () => {
            const fixture = setup();
            stats$.next(makeStats()); stats$.complete();
            fixture.detectChanges();
            openTab(fixture, 1);
            (fixture.nativeElement.querySelector('[data-testid="stats-family-row"]') as HTMLElement).click();
            fixture.detectChanges();
            runs$.next([
                { id: 'j1', lora_name: 'x_lora', status: 'completed', created_at: 1752624000,
                  completed_steps: 1500, avg_step_time: 1.70523, min_loss: 0.005319 },
            ]);
            runs$.complete();
            fixture.detectChanges();

            const row = fixture.nativeElement.querySelector('[data-testid="stats-run-row"]') as HTMLElement;
            expect(row.textContent).toContain('▸'); // collapsed
            row.click();
            fixture.detectChanges();
            expect(row.textContent).toContain('▾'); // expanded
        });

        it('renders the Adaptive header + event table when the run has adaptive events', () => {
            const fixture = setup();
            stats$.next(makeStats()); stats$.complete();
            fixture.detectChanges();
            expandOneRun(fixture, 'j1');

            adaptive$.next({
                events: [
                    { step: 100, event_index: 0, kind: 'narrow', active_count: 200, total_count: 248,
                      hot_count: 40, active_param_pct: 80.60000000000001, earliest_active_block: 2 },
                    { step: 300, event_index: 1, kind: 'narrow', active_count: 120, total_count: 248,
                      hot_count: 30, active_param_pct: 48.4, earliest_active_block: 5 },
                ],
                modules: [], heat: {},
            });
            adaptive$.complete();
            metrics$.next({ curve: [], summary: {} });
            metrics$.complete();
            fixture.detectChanges();

            const el: HTMLElement = fixture.nativeElement;
            const section = el.querySelector('[data-testid="stats-adaptive-section"]');
            expect(section).toBeTruthy();
            expect(section?.textContent).toContain('Adaptive');
            const rows = el.querySelectorAll('[data-testid="stats-adapt-row"]');
            expect(rows.length).toBe(2);
            expect(rows[0].textContent).toContain('narrow');
            expect(rows[0].textContent).toContain('200/248');
            // fmtPct rounds to 1 decimal — an unrounded float must never leak
            // through as "80.60000000000001%".
            expect(rows[0].textContent).toContain('80.6%');
            expect(rows[0].textContent).not.toContain('80.60000000000001');
            expect(rows[1].textContent).toContain('120/248');
        });

        it('is genuinely absent from the DOM (not CSS-hidden) for a run with no adaptive history — and does NOT show the error branch', () => {
            const fixture = setup();
            stats$.next(makeStats()); stats$.complete();
            fixture.detectChanges();
            expandOneRun(fixture, 'j1');

            adaptive$.next({ events: [], modules: [], heat: {} });
            adaptive$.complete();
            metrics$.next({ curve: [], summary: {} });
            metrics$.complete();
            fixture.detectChanges();

            const el: HTMLElement = fixture.nativeElement;
            expect(el.querySelector('[data-testid="stats-adaptive-section"]')).toBeFalsy();
            expect(el.querySelector('[data-testid="stats-adapt-empty"]')).toBeTruthy();
            expect(el.querySelector('[data-testid="stats-adapt-error"]')).toBeFalsy();
        });

        // The empty-200 shape ("this run never used the feature") and a genuine
        // fetch failure (500 / network drop / the 404 the backend reserves for
        // an unknown job id) must never collapse into the same "no data" message
        // — that would silently misreport an error as a plausible default.
        it('surfaces a failed adaptive fetch as an error, distinct from the empty-200 "no data" message', () => {
            const fixture = setup();
            stats$.next(makeStats()); stats$.complete();
            fixture.detectChanges();
            expandOneRun(fixture, 'j1');

            adaptive$.error(new Error('boom'));
            metrics$.next({ curve: [], summary: {} });
            metrics$.complete();
            fixture.detectChanges();

            const el: HTMLElement = fixture.nativeElement;
            expect(el.querySelector('[data-testid="stats-adapt-error"]')).toBeTruthy();
            expect(el.querySelector('[data-testid="stats-adapt-empty"]')).toBeFalsy();
            expect(el.querySelector('[data-testid="stats-adaptive-section"]')).toBeFalsy();
        });

        it('renders the chart host when the run has a curve', () => {
            const fixture = setup();
            stats$.next(makeStats()); stats$.complete();
            fixture.detectChanges();
            expandOneRun(fixture, 'j1');

            adaptive$.next({
                events: [{ step: 10, event_index: 0, kind: 'narrow', active_count: 5, total_count: 8 }],
                modules: [], heat: {},
            });
            adaptive$.complete();
            metrics$.next({
                curve: [
                    { step: 1, loss: 0.5, lr: 1e-4, grad_norm: null, timestep_mean: null, epoch: null, active_layers: 8 },
                    { step: 2, loss: 0.4, lr: 1e-4, grad_norm: null, timestep_mean: null, epoch: null, active_layers: null },
                    { step: 3, loss: 0.3, lr: 1e-4, grad_norm: null, timestep_mean: null, epoch: null, active_layers: 5 },
                ],
                summary: {},
            });
            metrics$.complete();
            fixture.detectChanges();

            const section = fixture.nativeElement.querySelector('[data-testid="stats-adaptive-section"]');
            expect(section?.querySelector('app-stats-uplot')).toBeTruthy();
        });

        // `StatsUplotComponent` short-circuits before constructing uPlot under
        // jsdom (no 2D canvas context), so the DOM has no observable trace of
        // WHICH points were plotted — the render-presence test above cannot
        // pin the NULL-skip contract. Assert on the component's own
        // `adaptiveData()` (which wraps `buildAdaptiveSeries`, already unit-pinned
        // in stats-charts.spec.ts) so a regression that started plotting NULL as
        // 0 fails here too, not only one layer down.
        it('feeds the chart a NULL-skipped series, never a zero-filled one', () => {
            const fixture = setup();
            stats$.next(makeStats()); stats$.complete();
            fixture.detectChanges();
            expandOneRun(fixture, 'j1');

            adaptive$.next({
                events: [{ step: 10, event_index: 0, kind: 'narrow', active_count: 5, total_count: 8 }],
                modules: [], heat: {},
            });
            adaptive$.complete();
            metrics$.next({
                curve: [
                    { step: 1, loss: 0.5, lr: 1e-4, grad_norm: null, timestep_mean: null, epoch: null, active_layers: 8 },
                    { step: 2, loss: 0.4, lr: 1e-4, grad_norm: null, timestep_mean: null, epoch: null, active_layers: null },
                    { step: 3, loss: 0.3, lr: 1e-4, grad_norm: null, timestep_mean: null, epoch: null, active_layers: 5 },
                ],
                summary: {},
            });
            metrics$.complete();
            fixture.detectChanges();

            const data = (fixture.componentInstance as any).adaptiveData();
            expect(data[0]).toEqual([1, 3]);   // step 2 (NULL) skipped, not plotted as 0
            expect(data[1]).toEqual([8, 5]);
        });

        it('collapses the run (and its Adaptive fetches) on a second click', () => {
            const fixture = setup();
            stats$.next(makeStats()); stats$.complete();
            fixture.detectChanges();
            expandOneRun(fixture, 'j1');
            adaptive$.next({ events: [{ step: 10, event_index: 0, kind: 'narrow', active_count: 5, total_count: 8 }], modules: [], heat: {} });
            adaptive$.complete();
            metrics$.next({ curve: [], summary: {} }); metrics$.complete();
            fixture.detectChanges();
            expect(fixture.nativeElement.querySelector('[data-testid="stats-adaptive-section"]')).toBeTruthy();

            (fixture.nativeElement.querySelector('[data-testid="stats-run-row"]') as HTMLElement).click();
            fixture.detectChanges();
            expect(fixture.nativeElement.querySelector('[data-testid="stats-adaptive-section"]')).toBeFalsy();
        });
    });
});
