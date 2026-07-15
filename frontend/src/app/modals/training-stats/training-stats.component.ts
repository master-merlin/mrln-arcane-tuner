import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';
import uPlot from 'uplot';
import { JobService, type TrainingStats } from '../../services/job';
import { ProjectService } from '../../services/project.service';
import { OverlayStore } from '../../state/overlay.store';
import { KpiTileComponent } from '../../ui/kpi-tile/kpi-tile.component';
import { formatDuration } from '../../shared/job-metrics';
import { buildActivityChart, buildHistogramChart } from './stats-charts';
import { StatsUplotComponent } from './stats-uplot.component';

/**
 * Cross-job training statistics modal — the redesign successor of the legacy
 * collapsible "Training Statistics" card (TrainingStatsComponent, 59f992bd).
 * Global by default, narrowable to a project; server-side aggregation via
 * `GET /jobs/history/stats`.
 */
@Component({
    selector: 'app-modal-training-stats',
    standalone: true,
    imports: [KpiTileComponent, StatsUplotComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">JOBS</div>
                <div class="modal-title">Training statistics</div>
            </div>
            <div class="ts-head-actions">
                <select data-testid="stats-project-filter"
                        [value]="projectFilter()" (change)="onFilterChange($event)"
                        class="ts-filter">
                    <option value="all">All projects</option>
                    @for (p of projectService.allProjects(); track p.id) {
                        <option [value]="p.id">{{ p.name }}</option>
                    }
                </select>
                <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
            </div>
        </div>

        <div class="modal-body">
            @if (loading()) {
                <div class="ts-empty">Loading statistics…</div>
            } @else if (stats(); as s) {
                @if (s.total_jobs === 0) {
                    <div class="ts-empty" data-testid="stats-empty">
                        {{ projectFilter() === 'all' ? 'No trainings yet.' : 'No trainings in this project.' }}
                    </div>
                } @else {
                    <!-- ── KPI row ─────────────────────────────────── -->
                    <div class="ts-kpis">
                        <div data-testid="stats-kpi-total">
                            <app-kpi-tile label="Total jobs" [value]="s.total_jobs" accent="brand"
                                          [sub]="s.completed + ' done · ' + s.failed + ' failed · ' + s.stopped + ' stopped'"/>
                        </div>
                        <div data-testid="stats-kpi-success">
                            <app-kpi-tile label="Success rate" [value]="s.success_rate" unit="%"
                                          [accent]="s.success_rate >= 50 ? 'success' : 'warning'"/>
                        </div>
                        <app-kpi-tile label="Total steps" [value]="fmtCount(s.total_steps)"
                                      [sub]="'avg ' + fmtCount(s.avg_steps) + ' / run'"/>
                        <app-kpi-tile label="GPU time" [value]="fmtHours(s.gpu_hours)" unit="h"
                                      [sub]="s.overhead_pct + '% overhead'" accent="violet"/>
                        <app-kpi-tile label="LoRAs produced" [value]="s.lora_count"
                                      [sub]="fmtGB(s.lora_bytes) + ' · ' + s.checkpoint_count + ' checkpoints'" accent="teal"/>
                    </div>

                    <!-- ── Activity ────────────────────────────────── -->
                    @if (activityData(); as ad) {
                        <div class="card ts-section">
                            <div class="card-head"><div class="card-title">Activity · jobs per week</div>
                                <div class="ts-legend">
                                    <span><i class="dot success"></i> completed</span>
                                    <span><i class="dot danger"></i> failed</span>
                                    <span><i class="dot warning"></i> stopped/other</span>
                                </div>
                            </div>
                            <div class="card-body">
                                <app-stats-uplot [data]="ad" [opts]="activityOpts" [height]="150"/>
                            </div>
                        </div>
                    }

                    <!-- ── Quality ─────────────────────────────────── -->
                    <div class="card ts-section">
                        <div class="card-head"><div class="card-title">Quality · completed runs</div></div>
                        <div class="card-body ts-quality">
                            @if (histData(); as hd) {
                                <app-stats-uplot [data]="hd" [opts]="histOpts" [height]="130"/>
                            } @else {
                                <div class="ts-note">Not enough completed runs for a loss distribution.</div>
                            }
                            <div class="ts-quality-tiles">
                                <app-kpi-tile label="Avg loss" [value]="stats()!.avg_loss" [compact]="true"/>
                                <app-kpi-tile label="Best loss" [value]="stats()!.avg_min_loss" [compact]="true" accent="success"/>
                                <app-kpi-tile label="Avg step time" [value]="stats()!.avg_step_time_sec" unit="s" [compact]="true"/>
                                <app-kpi-tile label="Avg runtime" [value]="fmtDur(stats()!.avg_runtime_sec)" [compact]="true"/>
                            </div>
                        </div>
                    </div>
                }
            }
        </div>

        <div class="modal-foot">
            <button class="btn ghost" type="button" (click)="overlay.closeModal()">Close</button>
        </div>
    `,
    styles: [`
        :host { display: contents; }
        .modal-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .ts-head-actions { display: flex; align-items: center; gap: 10px; }
        .ts-filter {
            background: var(--color-surface-high); color: var(--color-text-secondary);
            border: 1px solid var(--color-surface-mid); border-radius: 4px;
            font-size: 11px; padding: 3px 6px; outline: none;
        }
        .ts-filter:focus { border-color: var(--color-brand); }
        .ts-empty {
            display: flex; align-items: center; justify-content: center;
            padding: 48px; color: var(--color-text-muted); font-size: 13px;
        }
        .ts-kpis {
            display: grid; grid-template-columns: repeat(5, 1fr);
            gap: 10px; margin-bottom: 16px;
        }
        @media (max-width: 900px) { .ts-kpis { grid-template-columns: repeat(2, 1fr); } }
        .ts-section { margin-bottom: 14px; }
        .ts-legend { display: flex; gap: 12px; font-size: 10.5px; color: var(--color-text-muted); }
        .ts-legend .dot { width: 8px; height: 8px; border-radius: 2px; display: inline-block; margin-right: 4px; }
        .ts-legend .dot.success { background: var(--color-success); }
        .ts-legend .dot.danger { background: var(--color-danger); }
        .ts-legend .dot.warning { background: var(--color-warning); }
        .ts-quality { display: grid; grid-template-columns: 1fr 220px; gap: 14px; align-items: start; }
        .ts-quality-tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .ts-note { color: var(--color-text-muted); font-size: 12px; padding: 20px 0; }
        @media (max-width: 900px) { .ts-quality { grid-template-columns: 1fr; } }
    `],
})
export class TrainingStatsModalComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    protected projectService = inject(ProjectService);
    private jobService = inject(JobService);

    protected loading = signal(false);
    protected stats = signal<TrainingStats | null>(null);
    protected projectFilter = signal<string>('all');

    private reloadSeq = 0;

    protected readonly activityChart = computed(() => {
        const s = this.stats();
        return s ? buildActivityChart(s.activity) : null;
    });
    protected readonly activityData = computed<uPlot.AlignedData | null>(() => {
        const c = this.activityChart();
        if (!c || !c.xs.length) return null;
        return [c.xs, c.stoppedCum, c.failedCum, c.completedCum];
    });
    protected readonly activityOpts: Omit<uPlot.Options, 'width' | 'height'>;

    protected readonly histData = computed<uPlot.AlignedData | null>(() => {
        const s = this.stats();
        const h = s ? buildHistogramChart(s.loss_histogram) : null;
        return h ? [h.xs, h.counts] : null;
    });
    protected readonly histOpts: Omit<uPlot.Options, 'width' | 'height'>;

    constructor() {
        this.activityOpts = {
            legend: { show: false },
            cursor: { show: false },
            scales: { x: { time: true } },
            axes: [
                {},
                { size: 36, incrs: [1, 2, 5, 10, 25, 50, 100] },
            ],
            series: [
                {},
                // draw order bottom layer first: full cumulative in "stopped" color
                { paths: uPlot.paths.bars!({ size: [0.6, 100] }), fill: this.cssVar('--color-warning'), stroke: 'transparent', points: { show: false } },
                { paths: uPlot.paths.bars!({ size: [0.6, 100] }), fill: this.cssVar('--color-danger'), stroke: 'transparent', points: { show: false } },
                { paths: uPlot.paths.bars!({ size: [0.6, 100] }), fill: this.cssVar('--color-success'), stroke: 'transparent', points: { show: false } },
            ],
        };
        this.histOpts = {
            legend: { show: false },
            cursor: { show: false },
            scales: { x: { time: false } },
            axes: [{}, { size: 36 }],
            series: [
                {},
                { paths: uPlot.paths.bars!({ size: [0.8, 100] }), fill: this.cssVar('--color-brand'), stroke: 'transparent', points: { show: false } },
            ],
        };
    }

    private cssVar(name: string): string {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
    }

    ngOnInit(): void { this.reload(); }

    protected onFilterChange(ev: Event): void {
        this.projectFilter.set((ev.target as HTMLSelectElement).value);
        this.reload();
    }

    protected reload(): void {
        const seq = ++this.reloadSeq;
        this.loading.set(true);
        this.jobService.getTrainingStats(this.projectFilter()).subscribe({
            next: s => { if (seq !== this.reloadSeq) return; this.stats.set(s); this.loading.set(false); },
            error: () => { if (seq !== this.reloadSeq) return; this.stats.set(null); this.loading.set(false); },
        });
    }

    protected fmtCount(n: number): string {
        return n >= 10_000 ? `${(n / 1000).toFixed(1)}k` : String(n);
    }
    protected fmtHours(h: number): string { return h.toFixed(h >= 100 ? 0 : 1); }
    protected fmtGB(bytes: number): string { return (bytes / 1024 ** 3).toFixed(2) + ' GB'; }
    /**
     * `formatDuration(startedAtSec, endMs)` computes elapsed time between an
     * epoch-seconds start and an epoch-ms end (`Math.floor((endMs -
     * startedAtSec*1000)/1000)`), and treats a falsy `startedAtSec` as "no
     * start" → '0:00'. To format a plain duration-in-seconds (e.g.
     * `total_runtime_sec`) without re-deriving the h:mm:ss logic, anchor at
     * epoch second 1 (truthy) and place the end at `(sec + 1) * 1000` — the
     * 1s offset cancels out, leaving exactly `sec` elapsed seconds.
     */
    protected fmtDur(sec: number): string { return formatDuration(1, (sec + 1) * 1000); }
}
