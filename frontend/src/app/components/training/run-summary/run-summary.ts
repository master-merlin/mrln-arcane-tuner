import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';
import { Job } from '../../../services/job';

/**
 * Presentational metric-strip for a single training run/job.
 *
 * Renders the same value set as the Jobs → Archive "Training Summary" card so
 * the two surfaces stay identical (PR8 will adopt this component in the Archive
 * view). Purely data-driven from a {@link Job}'s summary fields — no stores,
 * no actions, no expanders. The host owns those.
 */
@Component({
    selector: 'app-run-summary',
    standalone: true,
    imports: [DatePipe, DecimalPipe],
    templateUrl: './run-summary.html',
    styleUrl: './run-summary.css',
    changeDetection: ChangeDetectionStrategy.OnPush,
})
export class RunSummaryComponent {
    /** The run to summarise. */
    readonly job = input.required<Job>();
    /** When false, the identity header (name · model · status · when) is hidden. */
    readonly showHeader = input(true);

    protected readonly loraName = computed<string>(() => {
        const j = this.job();
        return (j.config?.['lora_name'] as string) || j.lora_name || 'UNNAMED';
    });

    protected readonly model = computed<string>(() => {
        const j = this.job();
        return (j.config?.['definition_id'] as string) || j.definition_id || j.plugin_id || '—';
    });

    /** Tone class consumed by the global `.chip` styles. Mirrors projects-screen. */
    protected readonly statusTone = computed<string>(() => {
        switch (this.job().status) {
            case 'running':
            case 'completed': return 'success';
            case 'failed': return 'danger';
            case 'stopped':
            case 'paused': return 'warning';
            case 'pending': return 'teal';
            default: return '';
        }
    });

    protected readonly metrics = computed(() => {
        const j = this.job();
        const cfg = (j.config ?? {}) as Record<string, unknown>;
        const avg = j.avg_loss;
        const min = j.min_loss;
        const improvement = avg != null && min != null && avg > 0 ? (1 - min / avg) * 100 : null;
        return {
            steps: j.completed_steps ?? null,
            epoch: this.finalEpoch(j),
            finalLoss: avg ?? null,
            bestLoss: min ?? null,
            bestLossStep: j.min_loss_step ?? null,
            improvement,
            improvementGood: improvement != null && min != null && avg != null && min < avg * 0.9,
            avgStep: j.avg_step_time ?? null,
            optimizer: (cfg['optimizer_type'] as string) || 'AdamW',
            lr: this.formatLR(cfg['learning_rate']),
            batch: (cfg['train_batch_size'] as number) ?? 1,
            trainTime: this.formatTrainingTime(j.training_seconds),
        };
    });

    /** Prefer the exact persisted epoch; fall back to em-dash. */
    private finalEpoch(job: Job): string {
        if (!job.completed_steps) return '—';
        if (job.completed_epochs != null) return job.completed_epochs.toFixed(2);
        return '—';
    }

    private formatLR(lr: unknown): string {
        if (lr == null || lr === 0) return '—';
        const n = Number(lr);
        if (Number.isNaN(n)) return '—';
        if (n < 0.0001) return n.toExponential(1);
        return n.toString();
    }

    private formatTrainingTime(seconds: number | null | undefined): string {
        if (!seconds || seconds <= 0) return '—';
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        if (h > 0) return `${h}h ${m}m`;
        return `${m}m`;
    }
}
