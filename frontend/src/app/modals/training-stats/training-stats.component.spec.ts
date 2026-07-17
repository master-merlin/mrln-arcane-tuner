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

    function setup() {
        stats$ = new Subject<TrainingStats>();
        getTrainingStats = vi.fn().mockReturnValue(stats$.asObservable());
        recompute$ = new Subject<unknown>();
        recomputeStats = vi.fn().mockReturnValue(recompute$.asObservable());
        TestBed.configureTestingModule({
            imports: [TrainingStatsModalComponent],
            providers: [
                { provide: JobService, useValue: { getTrainingStats, recomputeStats } },
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
});
