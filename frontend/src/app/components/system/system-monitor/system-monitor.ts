import { Component, computed, effect, inject, OnInit, OnDestroy, signal } from '@angular/core';
import { SystemService } from '../../../services/system.service';
import { JobStore } from '../../../state/job.store';
import { Job, JobStatus } from '../../../services/job';
import { SparklineComponent } from '../../../ui/sparkline/sparkline.component';

interface RailRow {
    key: string;
    label: string;
    valueText: string;
    sub?: string;
    pct: number;
    barColor: string;
    spark: number[];
    sparkColor: string;
}

interface HeatCell {
    color: string;
    title: string;
    dim: boolean;
}

const HISTORY_CAP = 80;

/**
 * System rail (Jobs screen, right pane) — per Hi-Fi `screen-jobs.jsx`.
 *
 * CPU / RAM / GPU / VRAM / Power rows each show a live value, a rolling
 * sparkline, and a utilization bar; a GPU temp chip + temp-trend mini-chart
 * and a recent-runs heatmap close out the rail.
 */
@Component({
    selector: 'app-system-monitor',
    standalone: true,
    host: { class: 'sys-rail' },
    imports: [SparklineComponent],
    template: `
        @if (svc.metrics(); as snap) {
            <div class="sys-head">
                <div class="eyebrow sys-eyebrow"><span class="sys-live"></span>SYSTEM</div>
                @if (primaryGpu(); as g) {
                    <span class="chip danger temp-chip mono">{{ g.temperature_c }}°C</span>
                }
            </div>
            @if (primaryGpu(); as g) {
                <div class="sys-gpu-name mono">{{ g.name }}</div>
            }

            <div class="sys-metrics">
                @for (row of rows(); track row.key) {
                    <div class="sys-metric" [style.--tone]="row.sparkColor">
                        <div class="sys-metric-head">
                            <span class="sys-metric-label">{{ row.label }}</span>
                            <span class="sys-metric-val mono"
                                  [style.color]="row.pct >= 85 ? row.barColor : null">{{ row.valueText }}</span>
                        </div>
                        @if (row.sub) {
                            <div class="sys-metric-sub mono">{{ row.sub }}</div>
                        }
                        <div class="sys-spark">
                            <app-sparkline [data]="row.spark" [color]="row.sparkColor" [height]="26"/>
                        </div>
                        <div class="sys-meter"><i [style.width.%]="row.pct" [style.background]="row.barColor"></i></div>
                    </div>
                }
            </div>

            <!-- VRAM by process (who's holding the card) -->
            @if (gpuProcesses().length > 0) {
                <div class="sys-section">
                    <div class="sys-subtitle">VRAM by process</div>
                    <div class="sys-procs">
                        @for (p of gpuProcesses(); track p.pid) {
                            <div class="sys-proc" [title]="p.name + ' (pid ' + p.pid + ')'">
                                <div class="sys-proc-head">
                                    <span class="sys-proc-name">{{ p.name }}</span>
                                    <span class="sys-proc-val mono">{{ fmtGb(p.used_mb) }}</span>
                                </div>
                                <div class="sys-proc-bar"><i [style.width.%]="procPct(p.used_mb)"></i></div>
                            </div>
                        }
                    </div>
                </div>
            }

            <!-- Temp trend -->
            @if (tempHist().length > 1) {
                <div class="sys-section">
                    <div class="sys-subtitle">Temp trend</div>
                    <div class="sys-temp-trend">
                        <app-sparkline [data]="tempHist()" color="var(--color-danger)" [height]="52"/>
                        <span class="sys-temp-peak mono">{{ maxTemp() }}° peak</span>
                    </div>
                </div>
            }

            <!-- Recent runs heatmap -->
            @if (recentRuns().length > 0) {
                <div class="sys-section">
                    <div class="sys-subtitle">Recent runs · {{ recentRuns().length }}</div>
                    <div class="sys-heat">
                        @for (cell of recentRuns(); track $index) {
                            <div class="sys-heat-cell"
                                 [style.background]="cell.color"
                                 [style.opacity]="cell.dim ? 0.6 : 1"
                                 [title]="cell.title"></div>
                        }
                    </div>
                    <div class="sys-heat-axis mono">
                        <span>older</span><span>now</span>
                    </div>
                </div>
            }
        } @else {
            <div class="sys-loading">Connecting to system monitor…</div>
        }
    `,
    styleUrl: './system-monitor.css',
})
export class SystemMonitorComponent implements OnInit, OnDestroy {
    svc = inject(SystemService);
    private jobStore = inject(JobStore);

    /** Rolling per-metric history keyed by row key (+ 'temp'). */
    private readonly hist = signal<Record<string, number[]>>({});

    protected readonly primaryGpu = computed(() => this.svc.metrics()?.gpus?.[0] ?? null);
    protected readonly tempHist = computed<number[]>(() => this.hist()['temp'] ?? []);
    /** Peak temperature across the rolling history window. */
    protected readonly maxTemp = computed<number>(() => {
        const h = this.tempHist();
        return h.length ? Math.round(Math.max(...h)) : 0;
    });
    /** Top VRAM-holding processes on the primary GPU (ComfyUI, browser, …). */
    protected readonly gpuProcesses = computed(() => this.primaryGpu()?.processes ?? []);

    /** A process's share of total card VRAM, as a percentage (for its bar). */
    protected procPct(usedMb: number): number {
        const total = this.primaryGpu()?.vram_total_mb ?? 0;
        return total > 0 ? Math.min(100, (usedMb / total) * 100) : 0;
    }

    constructor() {
        // Accumulate a rolling buffer each time the metrics snapshot changes.
        // (Writing signals inside effects is an established pattern here.)
        effect(() => {
            const snap = this.svc.metrics();
            if (!snap) return;
            const g = snap.gpus?.[0];
            const push = (rec: Record<string, number[]>, key: string, v: number | undefined) => {
                if (v == null || !Number.isFinite(v)) return;
                const arr = [...(rec[key] ?? []), v];
                rec[key] = arr.length > HISTORY_CAP ? arr.slice(-HISTORY_CAP) : arr;
            };
            this.hist.update((prev) => {
                const next = { ...prev };
                push(next, 'cpu', snap.system?.cpu_percent);
                push(next, 'ram', snap.system?.ram_percent);
                if (g) {
                    push(next, 'gpu', g.gpu_utilization);
                    push(next, 'vram', g.vram_percent);
                    push(next, 'power', g.power_limit_w > 0 ? (g.power_draw_w / g.power_limit_w) * 100 : 0);
                    push(next, 'temp', g.temperature_c);
                }
                return next;
            });
        });
    }

    private tone(pct: number): string {
        if (pct > 90) return 'var(--color-danger)';
        if (pct > 70) return 'var(--color-warning)';
        return 'var(--color-success)';
    }

    protected readonly rows = computed<RailRow[]>(() => {
        const snap = this.svc.metrics();
        if (!snap) return [];
        const h = this.hist();
        const g = snap.gpus?.[0];
        const out: RailRow[] = [];

        const cpu = snap.system?.cpu_percent ?? 0;
        out.push({
            key: 'cpu',
            label: 'CPU',
            valueText: `${cpu.toFixed(0)}%`,
            pct: cpu,
            barColor: this.tone(cpu),
            spark: h['cpu'] ?? [],
            sparkColor: 'var(--color-chart-lr)',
        });

        const ramPct = snap.system?.ram_percent ?? 0;
        out.push({
            key: 'ram',
            label: 'RAM',
            valueText: `${this.fmtGb(snap.system?.ram_used_mb ?? 0)} / ${this.fmtGb(snap.system?.ram_total_mb ?? 0)}`,
            pct: ramPct,
            barColor: this.tone(ramPct),
            spark: h['ram'] ?? [],
            sparkColor: 'var(--color-brand)',
        });

        if (g) {
            out.push({
                key: 'gpu',
                label: 'GPU',
                valueText: `${g.gpu_utilization}%`,
                pct: g.gpu_utilization,
                barColor: this.tone(g.gpu_utilization),
                spark: h['gpu'] ?? [],
                sparkColor: 'var(--color-warning)',
            });
            const powerPct = g.power_limit_w > 0 ? (g.power_draw_w / g.power_limit_w) * 100 : 0;
            out.push({
                key: 'power',
                label: 'Power',
                valueText: `${g.power_draw_w.toFixed(0)} W`,
                pct: powerPct,
                barColor: this.tone(powerPct),
                spark: h['power'] ?? [],
                sparkColor: 'var(--color-success)',
            });
            const vramFree = g.vram_free_mb ?? Math.max(g.vram_total_mb - g.vram_used_mb, 0);
            out.push({
                key: 'vram',
                label: 'VRAM',
                valueText: `${this.fmtGb(g.vram_used_mb)} / ${this.fmtGb(g.vram_total_mb)}`,
                sub: `${this.fmtGb(vramFree)} free`,
                pct: g.vram_percent,
                barColor: this.tone(g.vram_percent),
                spark: h['vram'] ?? [],
                sparkColor: 'var(--color-violet)',
            });
        }
        return out;
    });

    /** Last 24 jobs (oldest→newest) as a status heatmap. */
    protected readonly recentRuns = computed<HeatCell[]>(() => {
        const jobs = [...this.jobStore.entities()].sort((a, b) => a.created_at - b.created_at);
        const last = jobs.slice(-24);
        return last.map((j) => ({
            color: this.statusColor(j.status),
            title: this.heatTitle(j),
            dim: j.status === JobStatus.COMPLETED,
        }));
    });

    private statusColor(status: JobStatus): string {
        switch (status) {
            case JobStatus.RUNNING:
                return 'var(--color-success)';
            case JobStatus.PENDING:
            case JobStatus.PAUSED:
                return 'var(--color-warning)';
            case JobStatus.FAILED:
                return 'var(--color-danger)';
            case JobStatus.STOPPED:
                return 'var(--color-text-disabled)';
            default:
                return 'oklch(0.40 0.06 155)';
        }
    }

    private heatTitle(j: Job): string {
        const name = j.config?.['lora_name'] || j.id.slice(0, 8);
        return `${name} · ${j.status}`;
    }

    ngOnInit() {
        this.svc.subscribeMetrics(2.0);
        // Seed history for the recent-runs heatmap.
        void this.jobStore.loadHistory();
    }

    ngOnDestroy() {
        this.svc.unsubscribeMetrics();
    }

    protected fmtGb(mb: number): string {
        if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
        return `${mb.toFixed(0)} MB`;
    }
}
