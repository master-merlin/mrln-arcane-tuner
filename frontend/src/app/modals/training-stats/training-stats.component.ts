import { ChangeDetectionStrategy, Component, OnInit, inject, signal } from '@angular/core';
import { JobService, type TrainingStats } from '../../services/job';
import { ProjectService } from '../../services/project.service';
import { OverlayStore } from '../../state/overlay.store';
import { KpiTileComponent } from '../../ui/kpi-tile/kpi-tile.component';
import { formatDuration } from '../../shared/job-metrics';

/**
 * Cross-job training statistics modal — the redesign successor of the legacy
 * collapsible "Training Statistics" card (TrainingStatsComponent, 59f992bd).
 * Global by default, narrowable to a project; server-side aggregation via
 * `GET /jobs/history/stats`.
 */
@Component({
    selector: 'app-modal-training-stats',
    standalone: true,
    imports: [KpiTileComponent],
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
                    <!-- Tasks 5 & 6 append sections here -->
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
    `],
})
export class TrainingStatsModalComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    protected projectService = inject(ProjectService);
    private jobService = inject(JobService);

    protected loading = signal(false);
    protected stats = signal<TrainingStats | null>(null);
    protected projectFilter = signal<string>('all');

    ngOnInit(): void { this.reload(); }

    protected onFilterChange(ev: Event): void {
        this.projectFilter.set((ev.target as HTMLSelectElement).value);
        this.reload();
    }

    protected reload(): void {
        this.loading.set(true);
        this.jobService.getTrainingStats(this.projectFilter()).subscribe({
            next: s => { this.stats.set(s); this.loading.set(false); },
            error: () => { this.stats.set(null); this.loading.set(false); },
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
