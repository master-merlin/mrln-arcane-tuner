import { Component, OnInit, signal, inject } from '@angular/core';
import { JobService, TrainingStats } from '../../../services/job';

@Component({
  selector: 'app-training-stats',
  standalone: true,
  template: `
    @if (stats(); as s) {
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">

        <!-- Total Jobs -->
        <div class="bg-surface-mid/50 rounded-theme-lg p-4 border border-surface-high/30">
          <div class="text-xs uppercase tracking-widest text-text-subtle font-bold mb-1">Total Jobs</div>
          <div class="text-2xl font-bold text-white">{{ s.total_jobs }}</div>
          <div class="mt-2 flex gap-2 flex-wrap">
            @if (s.completed > 0) {
              <span class="text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full border border-green-500/30 bg-green-500/10 text-green-400">{{ s.completed }} done</span>
            }
            @if (s.failed > 0) {
              <span class="text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full border border-red-500/30 bg-red-500/10 text-red-400">{{ s.failed }} failed</span>
            }
            @if (s.stopped > 0) {
              <span class="text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full border border-amber-500/30 bg-amber-500/10 text-amber-400">{{ s.stopped }} stopped</span>
            }
          </div>
        </div>

        <!-- Success Rate -->
        <div class="bg-surface-mid/50 rounded-theme-lg p-4 border border-surface-high/30">
          <div class="text-xs uppercase tracking-widest text-text-subtle font-bold mb-1">Success Rate</div>
          <div class="text-2xl font-bold" [class]="s.success_rate >= 80 ? 'text-green-400' : s.success_rate >= 50 ? 'text-amber-400' : 'text-red-400'">
            {{ s.success_rate }}%
          </div>
          <div class="mt-2 w-full bg-surface-high/50 rounded-full h-1.5">
            <div class="h-full rounded-full transition-all duration-500"
              [class]="s.success_rate >= 80 ? 'bg-green-500' : s.success_rate >= 50 ? 'bg-amber-500' : 'bg-red-500'"
              [style.width.%]="s.success_rate"></div>
          </div>
        </div>

        <!-- Total Steps -->
        <div class="bg-surface-mid/50 rounded-theme-lg p-4 border border-surface-high/30">
          <div class="text-xs uppercase tracking-widest text-text-subtle font-bold mb-1">Total Steps</div>
          <div class="text-2xl font-bold text-white">{{ formatNumber(s.total_steps) }}</div>
          <div class="text-xs text-text-muted mt-1">avg {{ formatNumber(s.avg_steps) }} / job</div>
        </div>

        <!-- Total Runtime -->
        <div class="bg-surface-mid/50 rounded-theme-lg p-4 border border-surface-high/30">
          <div class="text-xs uppercase tracking-widest text-text-subtle font-bold mb-1">Total Runtime</div>
          <div class="text-2xl font-bold text-white">{{ formatDuration(s.total_training_sec) }}</div>
          <div class="text-xs text-text-muted mt-1">avg {{ formatDuration(s.avg_runtime_sec) }} / job</div>
        </div>

        <!-- Avg Loss -->
        <div class="bg-surface-mid/50 rounded-theme-lg p-4 border border-surface-high/30">
          <div class="text-xs uppercase tracking-widest text-text-subtle font-bold mb-1">Avg Loss</div>
          <div class="text-2xl font-bold text-white font-mono">{{ s.avg_loss.toFixed(4) }}</div>
          <div class="text-xs text-text-muted mt-1">best avg {{ s.avg_min_loss.toFixed(4) }}</div>
        </div>

        <!-- Step Time -->
        <div class="bg-surface-mid/50 rounded-theme-lg p-4 border border-surface-high/30">
          <div class="text-xs uppercase tracking-widest text-text-subtle font-bold mb-1">Avg Step Time</div>
          <div class="text-2xl font-bold text-white font-mono">{{ s.avg_step_time_sec.toFixed(2) }}s</div>
          <div class="text-xs text-text-muted mt-1">per training step</div>
        </div>

        <!-- Datasets -->
        <div class="bg-surface-mid/50 rounded-theme-lg p-4 border border-surface-high/30">
          <div class="text-xs uppercase tracking-widest text-text-subtle font-bold mb-1">Datasets Used</div>
          <div class="text-2xl font-bold text-white">{{ s.unique_datasets }}</div>
          <div class="text-xs text-text-muted mt-1">unique datasets</div>
        </div>

        <!-- Model Families -->
        <div class="bg-surface-mid/50 rounded-theme-lg p-4 border border-surface-high/30">
          <div class="text-xs uppercase tracking-widest text-text-subtle font-bold mb-1">Model Families</div>
          <div class="mt-1 flex flex-col gap-1">
            @for (f of s.model_families; track f.id) {
              <div class="flex items-center justify-between">
                <span class="text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full border border-brand/30 bg-brand/10 text-brand truncate max-w-[120px]" [title]="f.id">{{ f.id }}</span>
                <span class="text-xs text-text-muted font-mono">{{ f.count }}</span>
              </div>
            }
            @if (s.model_families.length === 0) {
              <span class="text-xs text-text-subtle italic">No data</span>
            }
          </div>
        </div>

      </div>

      <!-- Optimizers Row -->
      @if (s.optimizers.length > 0) {
        <div class="mt-4 flex items-center gap-2 flex-wrap">
          <span class="text-xs uppercase tracking-widest text-text-subtle font-bold mr-1">Optimizers:</span>
          @for (o of s.optimizers; track o.name) {
            <span class="text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full border border-violet-500/30 bg-violet-500/10 text-violet-400">
              {{ o.name }} <span class="text-text-subtle">({{ o.count }})</span>
            </span>
          }
        </div>
      }
    } @else if (loading()) {
      <div class="flex items-center justify-center py-8 text-text-subtle">
        <div class="w-5 h-5 border-2 border-brand/20 border-t-brand rounded-full animate-spin mr-3"></div>
        Loading statistics…
      </div>
    } @else {
      <div class="py-6 text-center text-text-subtle text-sm">
        No training data yet. Complete your first job to see statistics.
      </div>
    }
  `,
})
export class TrainingStatsComponent implements OnInit {
  private jobService = inject(JobService);

  stats = signal<TrainingStats | null>(null);
  loading = signal(true);

  ngOnInit() {
    this.loadStats();
  }

  loadStats() {
    this.loading.set(true);
    this.jobService.getStats().subscribe({
      next: (data) => {
        this.stats.set(data.total_jobs > 0 ? data : null);
        this.loading.set(false);
      },
      error: () => this.loading.set(false),
    });
  }

  formatNumber(n: number): string {
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
    if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
    return n.toString();
  }

  formatDuration(seconds: number): string {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    const h = Math.floor(seconds / 3600);
    const m = Math.round((seconds % 3600) / 60);
    return m > 0 ? `${h}h ${m}m` : `${h}h`;
  }
}
