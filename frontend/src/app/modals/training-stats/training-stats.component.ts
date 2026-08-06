import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';
import uPlot from 'uplot';
import { JobService, type Job, type TrainingStats } from '../../services/job';
import { ProjectService } from '../../services/project.service';
import { OverlayStore } from '../../state/overlay.store';
import { KpiTileComponent } from '../../ui/kpi-tile/kpi-tile.component';
import { formatDuration, type AdaptEvent } from '../../shared/job-metrics';
import {
    buildActivityChart, buildHistogramChart, buildAdaptiveSeries,
    readAxisTheme, buildActivityOpts, buildHistogramOpts, buildAdaptiveOpts,
} from './stats-charts';
import { StatsUplotComponent } from './stats-uplot.component';
import { TabsComponent, type TabItem } from '../../ui/tabs/tabs.component';

/**
 * Cross-job training statistics modal — the redesign successor of the legacy
 * collapsible "Training Statistics" card (TrainingStatsComponent, 59f992bd).
 * Global by default, narrowable to a project; server-side aggregation via
 * `GET /jobs/history/stats`.
 */
@Component({
    selector: 'app-modal-training-stats',
    standalone: true,
    imports: [KpiTileComponent, StatsUplotComponent, TabsComponent],
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
                        <div data-testid="stats-kpi-total" [title]="jobsSubFull(s)">
                            <app-kpi-tile label="Total jobs" [value]="s.total_jobs" accent="brand" [animate]="true"
                                          [sub]="jobsSubShort(s)"/>
                        </div>
                        <div data-testid="stats-kpi-success">
                            <app-kpi-tile label="Success rate" [value]="s.success_rate" unit="%" [animate]="true"
                                          [accent]="s.success_rate >= 50 ? 'success' : 'warning'"/>
                        </div>
                        <div data-testid="stats-kpi-steps">
                            <app-kpi-tile label="Total steps" [value]="fmtCount(s.total_steps)" accent="warning"
                                          [sub]="'avg ' + fmtCount(s.avg_steps) + ' / run'"/>
                        </div>
                        <app-kpi-tile label="GPU time" [value]="fmtHours(s.gpu_hours)" unit="h"
                                      [sub]="s.overhead_pct + '% overhead'" accent="violet"/>
                        <div data-testid="stats-kpi-loras" [title]="loraSub(s)">
                            <app-kpi-tile label="LoRAs produced" [value]="s.lora_count" [animate]="true"
                                          [sub]="loraSubShort(s)" accent="teal"/>
                        </div>
                    </div>

                    <div data-testid="stats-tabs" class="ts-tabs">
                        <app-tabs [tabs]="TAB_ITEMS" [active]="activeTab()" (changed)="activeTab.set($event)"/>
                    </div>

                    @switch (activeTab()) {
                        @case ('activity') {
                            <!-- ── Activity ────────────────────────────────── -->
                            @if (activityData(); as ad) {
                                @if (activityOpts(); as ao) {
                                    <div class="card ts-section">
                                        <div class="card-head"><div class="card-title">Activity · jobs per week</div>
                                            <div class="ts-legend">
                                                <span><i class="dot success"></i> completed</span>
                                                <span><i class="dot danger"></i> failed</span>
                                                <span><i class="dot warning"></i> stopped/other</span>
                                            </div>
                                        </div>
                                        <div class="card-body">
                                            <app-stats-uplot [data]="ad" [opts]="ao" [height]="150"/>
                                        </div>
                                    </div>
                                }
                            }
                        }
                        @case ('quality') {
                            <!-- ── Quality ─────────────────────────────────── -->
                            <div class="card ts-section">
                                <div class="card-head"><div class="card-title">Quality · completed runs</div></div>
                                <div class="card-body ts-quality">
                                    @if (histData(); as hd) {
                                        @if (histOpts(); as ho) {
                                            <app-stats-uplot [data]="hd" [opts]="ho" [height]="130"/>
                                        }
                                    } @else {
                                        <div class="ts-note">Not enough completed runs for a loss distribution.</div>
                                    }
                                    <div class="ts-quality-tiles">
                                        <app-kpi-tile label="Avg loss" [value]="stats()!.avg_loss" [compact]="true"/>
                                        <div data-testid="stats-kpi-best-loss">
                                            <app-kpi-tile label="Best loss" [value]="s.records.best_loss?.value ?? '—'" [compact]="true" accent="success"/>
                                        </div>
                                        <app-kpi-tile label="Avg step time" [value]="stats()!.avg_step_time_sec" unit="s" [compact]="true"/>
                                        <app-kpi-tile label="Avg runtime" [value]="fmtDur(stats()!.avg_runtime_sec)" [compact]="true"/>
                                    </div>
                                </div>
                            </div>

                            <!-- ── Model families ──────────────────────────── -->
                            <div class="card ts-section">
                                <div class="card-head"><div class="card-title">Model families</div></div>
                                <div class="card-body">
                                    <div class="ts-fam-grid ts-fam-head mono">
                                        <span>family</span><span>jobs</span><span>success</span><span>avg step</span><span>best loss</span>
                                    </div>
                                    @for (f of s.families; track f.id) {
                                        <div class="ts-fam-grid" data-testid="stats-family-row"
                                             [class.ts-fam-expandable]="!!f.id"
                                             (click)="f.id && toggleFamily(f.id)">
                                            <span class="mono">
                                                @if (f.id) { <i class="ts-fam-caret">{{ expandedFamily() === f.id ? '▾' : '▸' }}</i> }
                                                {{ f.id || '—' }}
                                            </span>
                                            <span class="mono">{{ f.count }}</span>
                                            <span class="ts-rate">
                                                <i class="ts-rate-bar"><b [style.width.%]="f.success_rate"></b></i>
                                                <span class="mono">{{ f.success_rate }}%</span>
                                            </span>
                                            <span class="mono">{{ f.avg_step_time !== null ? f.avg_step_time + 's' : '—' }}</span>
                                            <span class="mono">{{ f.best_loss ?? '—' }}</span>
                                        </div>
                                        @if (f.id && expandedFamily() === f.id) {
                                            <div class="ts-fam-runs" data-testid="stats-family-runs">
                                                @if (familyRunsLoading()) {
                                                    <div class="ts-note">Loading runs…</div>
                                                } @else if (familyRuns(); as runs) {
                                                    <!-- per-run header: single-run values, not averages -->
                                                    <div class="ts-run-grid ts-fam-head mono" data-testid="stats-run-head">
                                                        <span>run</span><span>date</span><span>status</span>
                                                        <span>steps</span><span>step time</span><span>min loss</span>
                                                    </div>
                                                    @for (r of runs; track r.id) {
                                                        <div class="ts-run-grid ts-run-expandable" data-testid="stats-run-row"
                                                             (click)="toggleRunAdaptive(r.id)">
                                                            <span class="mono ts-run-name" [title]="r.lora_name">{{ r.lora_name }}</span>
                                                            <span class="mono">{{ fmtDate(r.created_at) }}</span>
                                                            <span class="mono ts-run-status" [class]="'ts-run-status ' + r.status">{{ r.status }}</span>
                                                            <span class="mono">{{ fmtCount(r.completed_steps ?? 0) }}</span>
                                                            <span class="mono">{{ fmtStepTime(r.avg_step_time) }}</span>
                                                            <span class="mono">{{ fmtLoss(r.min_loss) }}</span>
                                                        </div>
                                                        @if (expandedRun() === r.id) {
                                                            <!-- ── Adaptive layer targeting (Task 12) ──────────
                                                                 Durable per-run history — absent from the DOM
                                                                 entirely (not CSS-hidden) when the run never
                                                                 used the feature. -->
                                                            @if (runAdaptiveLoading()) {
                                                                <div class="ts-note">Loading adaptive data…</div>
                                                            } @else if (adaptiveEvents().length > 0) {
                                                                <div class="card ts-section ts-adapt-card" data-testid="stats-adaptive-section">
                                                                    <div class="card-head"><div class="card-title">Adaptive</div>
                                                                        <span class="mono ts-sub">layer targeting · {{ adaptiveEvents().length }} events</span>
                                                                    </div>
                                                                    <div class="card-body">
                                                                        @if (adaptiveData(); as ad) {
                                                                            @if (adaptiveOpts(); as ao) {
                                                                                <app-stats-uplot [data]="ad" [opts]="ao" [height]="110"/>
                                                                            }
                                                                        }
                                                                        <div class="ts-adapt-grid ts-fam-head mono">
                                                                            <span>step</span><span>kind</span><span>active/total</span>
                                                                            <span>active %</span><span>earliest block</span>
                                                                        </div>
                                                                        @for (e of adaptiveEvents(); track e.event_index ?? e.step) {
                                                                            <div class="ts-adapt-grid" data-testid="stats-adapt-row">
                                                                                <span class="mono">{{ e.step }}</span>
                                                                                <span class="mono">{{ e.kind }}</span>
                                                                                <span class="mono">{{ e.active_count }}/{{ e.total_count }}</span>
                                                                                <span class="mono">{{ e.active_param_pct != null ? e.active_param_pct + '%' : '—' }}</span>
                                                                                <span class="mono">{{ e.earliest_active_block ?? '—' }}</span>
                                                                            </div>
                                                                        }
                                                                    </div>
                                                                </div>
                                                            } @else {
                                                                <div class="ts-note" data-testid="stats-adapt-empty">No adaptive layer-targeting data for this run.</div>
                                                            }
                                                        }
                                                    } @empty { <div class="ts-note">No runs found.</div> }
                                                }
                                            </div>
                                        }
                                    }
                                </div>
                            </div>
                        }
                        @case ('config') {
                            <!-- ── Hyperparameters ─────────────────────────── -->
                            @if (hpRows().length) {
                                <div class="card ts-section">
                                    <div class="card-head"><div class="card-title">Hyperparameters</div>
                                        <span class="mono ts-sub">{{ s.resume_rate }}% of jobs resumed</span>
                                    </div>
                                    <div class="card-body">
                                        @for (row of hpRows(); track row.key) {
                                            <div class="ts-hp" data-testid="stats-hp-row">
                                                <span class="ts-hp-label">{{ row.label }}</span>
                                                <div class="ts-hp-bar">
                                                    @for (seg of row.segments; track seg.value) {
                                                        <i [style.flex]="seg.count" [style.background]="toneColor(seg.tone)"
                                                           [attr.title]="seg.value + ' · ' + seg.count"></i>
                                                    }
                                                </div>
                                                <span class="ts-hp-legend mono">
                                                    @for (seg of row.segments; track seg.value) {
                                                        <span><i class="dot" [style.background]="toneColor(seg.tone)"></i>{{ seg.value }} ({{ seg.count }})</span>
                                                    }
                                                </span>
                                            </div>
                                        }
                                    </div>
                                </div>
                            }

                            <!-- ── Datasets & records ──────────────────────── -->
                            <div class="card ts-section">
                                <div class="card-head"><div class="card-title">Datasets &amp; records</div>
                                    <span class="mono ts-sub">{{ s.unique_datasets }} unique datasets</span>
                                </div>
                                <div class="card-body ts-bottom">
                                    <div data-testid="stats-datasets">
                                        <div class="ts-mini-title">Most trained on</div>
                                        @for (d of s.top_datasets; track d.name) {
                                            <div class="ts-ds-row mono"><span>{{ d.name }}</span><span>{{ d.count }}</span></div>
                                        } @empty { <div class="ts-note">No dataset linkage yet.</div> }
                                    </div>
                                    <div data-testid="stats-records">
                                        <div class="ts-mini-title">Records</div>
                                        @if (s.records.longest_run; as r) {
                                            <div class="ts-rec-row"><span>Longest run</span><span class="mono">{{ r.lora_name }} · {{ fmtDur(r.value) }}</span></div>
                                        }
                                        @if (s.records.most_steps; as r) {
                                            <div class="ts-rec-row"><span>Most steps</span><span class="mono">{{ r.lora_name }} · {{ fmtCount(r.value) }}</span></div>
                                        }
                                        @if (s.records.best_loss; as r) {
                                            <div class="ts-rec-row"><span>Best loss</span><span class="mono">{{ r.lora_name }} · {{ r.value }}</span></div>
                                        }
                                    </div>
                                </div>
                            </div>

                            <div class="ts-reconcile">
                                <button class="btn ghost" type="button" data-testid="stats-reconcile"
                                        [disabled]="reconciling()" (click)="reconcile()">
                                    {{ reconciling() ? 'Reconciling…' : 'Reconcile from disk' }}
                                </button>
                                <span class="ts-note">Recovers LoRA files &amp; sizes from run output folders for runs recorded before live persistence.</span>
                                @if (reconcileError()) {
                                    <span class="ts-reconcile-err" data-testid="stats-reconcile-error">Reconcile failed — see server logs.</span>
                                }
                            </div>
                        }
                    }
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
            display: grid;
            /* minmax(0, 1fr): a plain 1fr track's min-content floor lets a
               long nowrap sub-line (LoRAs tile) widen its column — pin the
               floor to 0 so all five tiles stay equal and the sub ellipsizes
               (global .kpi-sub already carries overflow/ellipsis). */
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px; margin-bottom: 16px;
        }
        @media (max-width: 900px) { .ts-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
        .ts-section { margin-bottom: 14px; }
        .ts-tabs { margin-bottom: 14px; }
        .ts-legend { display: flex; gap: 12px; font-size: 10.5px; color: var(--color-text-muted); }
        .ts-legend .dot { width: 8px; height: 8px; border-radius: 2px; display: inline-block; margin-right: 4px; }
        .ts-legend .dot.success { background: var(--color-success); }
        .ts-legend .dot.danger { background: var(--color-danger); }
        .ts-legend .dot.warning { background: var(--color-warning); }
        .ts-quality { display: grid; grid-template-columns: 1fr 220px; gap: 14px; align-items: start; }
        .ts-quality-tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .ts-note { color: var(--color-text-muted); font-size: 12px; padding: 20px 0; }
        @media (max-width: 900px) { .ts-quality { grid-template-columns: 1fr; } }
        .ts-fam-grid { display: grid; grid-template-columns: 1.4fr 0.5fr 1.4fr 0.7fr 0.7fr; gap: 8px; align-items: center; padding: 4px 0; font-size: 12px; }
        .ts-fam-head { color: var(--color-text-muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; border-bottom: 1px solid var(--color-border-subtle); padding-bottom: 6px; }
        .ts-fam-expandable { cursor: pointer; }
        .ts-fam-expandable:hover { background: var(--color-surface-high); }
        .ts-fam-caret { font-style: normal; color: var(--color-text-muted); margin-right: 4px; }
        .ts-fam-runs { margin: 2px 0 8px; padding: 4px 8px 6px 18px; border-left: 2px solid var(--color-border-subtle); }
        .ts-run-grid { display: grid; grid-template-columns: 2fr 0.8fr 0.8fr 0.6fr 0.7fr 0.8fr; gap: 8px; align-items: center; padding: 3px 0; font-size: 11.5px; }
        .ts-run-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .ts-run-status.failed { color: var(--color-danger); }
        .ts-run-status.stopped { color: var(--color-warning); }
        .ts-run-expandable { cursor: pointer; }
        .ts-run-expandable:hover { background: var(--color-surface-high); }
        .ts-adapt-card { margin: 2px 0 8px 18px; }
        /* minmax(0, …): same overflow guard as .ts-kpis — "kind" values like
           rebuild_request are the longest cell and must not force the grid wide. */
        .ts-adapt-grid {
            display: grid;
            grid-template-columns: minmax(0, 0.6fr) minmax(0, 1.3fr) minmax(0, 1fr) minmax(0, 0.9fr) minmax(0, 1fr);
            gap: 8px; align-items: center; padding: 3px 0; font-size: 11.5px;
        }
        .ts-rate { display: flex; align-items: center; gap: 8px; }
        .ts-rate-bar { flex: 1; height: 6px; border-radius: 3px; background: var(--color-surface-mid); overflow: hidden; display: block; }
        .ts-rate-bar b { display: block; height: 100%; background: var(--color-success); }
        .ts-hp { display: grid; grid-template-columns: 130px 1fr; gap: 4px 12px; margin-bottom: 10px; }
        .ts-hp-label { font-size: 11px; color: var(--color-text-secondary); align-self: center; }
        .ts-hp-bar { display: flex; height: 12px; border-radius: 3px; overflow: hidden; background: var(--color-surface-mid); }
        .ts-hp-bar i { display: block; height: 100%; }
        .ts-hp-legend { grid-column: 2; display: flex; flex-wrap: wrap; gap: 10px; font-size: 10px; color: var(--color-text-muted); }
        .ts-hp-legend .dot { width: 7px; height: 7px; border-radius: 2px; display: inline-block; margin-right: 3px; }
        .ts-sub { font-size: 11px; color: var(--color-text-muted); }
        .ts-bottom { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
        .ts-mini-title { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--color-text-muted); margin-bottom: 6px; }
        .ts-ds-row, .ts-rec-row { display: flex; justify-content: space-between; font-size: 12px; padding: 3px 0; }
        @media (max-width: 900px) { .ts-bottom { grid-template-columns: 1fr; } }
        .ts-reconcile { display: flex; align-items: center; gap: 12px; margin-top: 4px; }
        .ts-reconcile .ts-note { padding: 0; }
        .ts-reconcile-err { color: var(--color-danger); font-size: 11px; }
    `],
})
export class TrainingStatsModalComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    protected projectService = inject(ProjectService);
    private jobService = inject(JobService);

    protected loading = signal(false);
    protected stats = signal<TrainingStats | null>(null);
    protected projectFilter = signal<string>('all');
    protected reconciling = signal(false);
    protected reconcileError = signal(false);

    protected readonly TAB_ITEMS: TabItem<'activity' | 'quality' | 'config'>[] = [
        { value: 'activity', label: 'Activity' },
        { value: 'quality', label: 'Quality & Families' },
        { value: 'config', label: 'Config & Data' },
    ];
    protected activeTab = signal<'activity' | 'quality' | 'config'>('activity');

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
    protected readonly activityOpts = computed<Omit<uPlot.Options, 'width' | 'height'> | null>(() => {
        const c = this.activityChart();
        if (!c) return null;
        return buildActivityOpts(readAxisTheme(), {
            success: this.cssVar('--color-success'),
            danger: this.cssVar('--color-danger'),
            warning: this.cssVar('--color-warning'),
            brand: this.cssVar('--color-brand'),
        }, c);
    });

    protected readonly histData = computed<uPlot.AlignedData | null>(() => {
        const s = this.stats();
        const h = s ? buildHistogramChart(s.loss_histogram) : null;
        return h ? [h.xs, h.counts] : null;
    });
    protected readonly histOpts = computed<Omit<uPlot.Options, 'width' | 'height'> | null>(() => {
        const s = this.stats();
        if (!s || !s.loss_histogram.edges.length) return null;
        return buildHistogramOpts(readAxisTheme(), this.cssVar('--color-brand'), s.loss_histogram.edges);
    });

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
        this.expandedFamily.set(null); // scope changed — stale run list must not survive
        this.collapseRunAdaptive(); // and any run drilled into under the old scope
        this.jobService.getTrainingStats(this.projectFilter()).subscribe({
            next: s => { if (seq !== this.reloadSeq) return; this.stats.set(s); this.loading.set(false); },
            error: () => { if (seq !== this.reloadSeq) return; this.stats.set(null); this.loading.set(false); },
        });
    }

    // ── Family drill-down ────────────────────────────────────────────
    protected expandedFamily = signal<string | null>(null);
    protected familyRuns = signal<Job[] | null>(null);
    protected familyRunsLoading = signal(false);
    private runsSeq = 0;

    /** Expand one family into its per-run table (lazy fetch); click again collapses. */
    protected toggleFamily(id: string): void {
        if (this.expandedFamily() === id) { this.expandedFamily.set(null); this.collapseRunAdaptive(); return; }
        this.expandedFamily.set(id);
        this.collapseRunAdaptive(); // a run drilled into under the old family must not survive
        const seq = ++this.runsSeq;
        this.familyRuns.set(null);
        this.familyRunsLoading.set(true);
        this.jobService.listFamilyRuns(id, this.projectFilter()).subscribe({
            next: runs => { if (seq !== this.runsSeq) return; this.familyRuns.set(runs); this.familyRunsLoading.set(false); },
            error: () => { if (seq !== this.runsSeq) return; this.familyRuns.set([]); this.familyRunsLoading.set(false); },
        });
    }

    // ── Adaptive layer targeting drill-down (Task 12) ──────────────────
    // The cross-job aggregate has no notion of "the current run" — the only
    // per-job identity in scope is a `Job.id` from the family run table above,
    // so a run row is the entry point: click it to fetch that one job's
    // durable adaptive-targeting history (`/adaptive`) + metrics curve
    // (`/metrics`, for the staircase chart) and render both inline.
    protected expandedRun = signal<string | null>(null);
    protected runAdaptiveLoading = signal(false);
    protected adaptiveEvents = signal<AdaptEvent[]>([]);
    private adaptiveCurve = signal<{ step: number; active_layers: number | null }[]>([]);
    private adaptiveSeq = 0;

    protected readonly adaptiveData = computed<uPlot.AlignedData | null>(() => {
        const s = buildAdaptiveSeries(this.adaptiveCurve());
        return s.steps.length ? [s.steps, s.counts] : null;
    });
    protected readonly adaptiveOpts = computed<Omit<uPlot.Options, 'width' | 'height'> | null>(() => {
        if (!this.adaptiveData()) return null;
        return buildAdaptiveOpts(readAxisTheme(), this.cssVar('--color-warning'));
    });

    /** Expand one run's Adaptive section (lazy fetch); click again collapses. */
    protected toggleRunAdaptive(jobId: string): void {
        if (this.expandedRun() === jobId) { this.collapseRunAdaptive(); return; }
        this.expandedRun.set(jobId);
        const seq = ++this.adaptiveSeq;
        this.runAdaptiveLoading.set(true);
        this.adaptiveEvents.set([]);
        this.adaptiveCurve.set([]);
        this.jobService.getJobAdaptiveHistory(jobId).subscribe({
            next: h => {
                if (seq !== this.adaptiveSeq) return;
                this.adaptiveEvents.set(h.events);
                this.runAdaptiveLoading.set(false);
            },
            error: () => {
                if (seq !== this.adaptiveSeq) return;
                this.adaptiveEvents.set([]);
                this.runAdaptiveLoading.set(false);
            },
        });
        // Chart data only — a failure here still leaves the event table usable.
        this.jobService.getJobMetrics(jobId).subscribe({
            next: m => { if (seq !== this.adaptiveSeq) return; this.adaptiveCurve.set(m.curve); },
            error: () => { /* chart stays empty; table is independent */ },
        });
    }

    private collapseRunAdaptive(): void {
        this.adaptiveSeq++; // invalidate any in-flight fetch for the run being collapsed
        this.expandedRun.set(null);
        this.runAdaptiveLoading.set(false);
        this.adaptiveEvents.set([]);
        this.adaptiveCurve.set([]);
    }

    protected fmtDate(sec: number): string { return new Date(sec * 1000).toISOString().slice(0, 10); }
    protected fmtStepTime(v: number | undefined): string { return typeof v === 'number' ? v.toFixed(3) + 's' : '—'; }
    protected fmtLoss(v: number | undefined): string { return typeof v === 'number' ? v.toFixed(6) : '—'; }

    /** Run the disk backfill (recovers legacy LoRA files/sizes), then refetch. */
    protected reconcile(): void {
        if (this.reconciling()) return;
        this.reconciling.set(true);
        this.reconcileError.set(false);
        this.jobService.recomputeStats().subscribe({
            next: () => { this.reconciling.set(false); this.reload(); },
            error: () => { this.reconciling.set(false); this.reconcileError.set(true); },
        });
    }

    /** Hyperparam dimensions in display order; only populated ones render. */
    protected readonly HP_DIMS: { key: string; label: string }[] = [
        { key: 'optimizer_type', label: 'Optimizer' },
        { key: 'network_rank', label: 'Network rank' },
        { key: 'lr_scheduler', label: 'LR scheduler' },
        { key: 'timestep_sampling', label: 'Timestep sampling' },
        { key: 'quantization', label: 'Quantization' },
        { key: 'mixed_precision', label: 'Mixed precision' },
        { key: 'ema_enabled', label: 'EMA' },
        { key: 'batch_size', label: 'Batch size' },
    ];
    private readonly TONES = ['brand', 'success', 'violet', 'teal', 'warning'];

    protected hpRows = computed(() => {
        const s = this.stats();
        if (!s) return [];
        return this.HP_DIMS
            .map(d => ({ ...d, counts: s.hyperparams[d.key] ?? [] }))
            .filter(d => d.counts.length > 0)
            .map(d => ({
                ...d,
                segments: d.counts.map((c, i) => ({
                    ...c, tone: this.TONES[i % this.TONES.length],
                })),
            }));
    });

    protected toneColor(tone: string): string { return `var(--color-${tone})`; }

    protected fmtCount(n: number): string {
        return n >= 10_000 ? `${(n / 1000).toFixed(1)}k` : String(n);
    }
    protected fmtHours(h: number): string { return h.toFixed(h >= 100 ? 0 : 1); }
    protected fmtGB(bytes: number): string { return (bytes / 1024 ** 3).toFixed(2) + ' GB'; }
    /**
     * Full LoRA detail — "2 on disk · 2.79 GB (4/10 sized) · 12 checkpoints"
     * — used as the tile's hover title; the visible sub-line is the compact
     * variant below so the KPI rail never ellipsizes at normal widths.
     */
    protected loraSub(s: TrainingStats): string {
        const gb = this.fmtGB(s.lora_bytes);
        const sized = s.lora_size_known < s.lora_count
            ? `${gb} (${s.lora_size_known}/${s.lora_count} sized)` : gb;
        return `${s.lora_on_disk} on disk · ${sized} · ${s.checkpoint_count} checkpoints`;
    }
    /**
     * Compact visible variant: on-disk + 1-decimal size only — coverage and
     * the checkpoint count live in the hover title (a ~160px tile fits about
     * 22 chars of sub-line before the global .kpi-sub ellipsis kicks in).
     */
    protected loraSubShort(s: TrainingStats): string {
        return `${s.lora_on_disk} on disk · ${(s.lora_bytes / 1024 ** 3).toFixed(1)} GB`;
    }
    /** Full outcome split for the Total-jobs hover title. */
    protected jobsSubFull(s: TrainingStats): string {
        return `${s.completed} done · ${s.failed} failed · ${s.stopped} stopped`;
    }
    /** Compact visible variant: zero-count outcomes are dropped. */
    protected jobsSubShort(s: TrainingStats): string {
        const parts = [`${s.completed} done`];
        if (s.failed > 0) parts.push(`${s.failed} failed`);
        if (s.stopped > 0) parts.push(`${s.stopped} stopped`);
        return parts.join(' · ');
    }
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
