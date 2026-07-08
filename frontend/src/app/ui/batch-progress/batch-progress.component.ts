import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

/**
 * Reusable determinate batch-progress panel. Replaces the per-modal
 * `.xx-progress` bars that were copy-pasted across the batch-run modals
 * (mass-caption / mass-mask / mass-edit).
 *
 * The per-domain accent is driven by the `--batch-accent` CSS custom property
 * — callers either set it directly or pass the convenience `accent` input
 * (e.g. `[accent]="'var(--color-success)'"`). The progress bar and the accent
 * eyebrow both read that property, so a single value re-tints the whole panel.
 *
 * Percent is either supplied explicitly (`percent`, so a caller can keep its
 * own rounding as the source of truth) or derived from `current`/`total`.
 *
 * A11y: the bar carries `role="progressbar"` with aria-valuenow/min/max and an
 * aria-label taken from the panel label.
 *
 * Slots: projected `<ng-content/>` at the foot for optional actions (e.g. a
 * Stop button rendered inside the panel).
 */
@Component({
    selector: 'app-batch-progress',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    host: {
        style: 'display: block;',
        '[style.--batch-accent]': 'accent()',
    },
    template: `
        <div class="batch-progress" data-testid="batch-progress">
            <div class="batch-progress-head">
                <div>
                    <div class="eyebrow accent">{{ label() }}</div>
                    <div class="batch-progress-pct" data-testid="batch-progress-pct">{{ pct() }}%</div>
                </div>
                <div class="batch-progress-queue">
                    <div class="eyebrow">{{ queueLabel() }}</div>
                    <span class="mono">{{ current() }} / {{ total() }}</span>
                </div>
            </div>
            <div class="batch-progress-bar"
                 data-testid="batch-progress-bar"
                 role="progressbar"
                 [attr.aria-valuenow]="pct()"
                 aria-valuemin="0"
                 aria-valuemax="100"
                 [attr.aria-label]="label()">
                <div class="batch-progress-bar-fill" data-testid="batch-progress-fill"
                     [style.width.%]="pct()"></div>
            </div>
            @if (showCurrent()) {
                <div class="batch-progress-cur">
                    <span class="eyebrow">{{ currentLabel() }}</span>
                    <span class="mono">{{ currentItem() }}</span>
                </div>
            }
            @if (hint(); as h) {
                <div class="batch-progress-hint" data-testid="batch-progress-hint">{{ h }}</div>
            }
            <ng-content/>
        </div>
    `,
    styles: [`
        .batch-progress {
            padding: 20px 22px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-default);
            border-radius: var(--radius-theme-2xl);
        }
        .eyebrow {
            font-size: 10px; font-weight: 700; letter-spacing: 0.14em;
            text-transform: uppercase; color: var(--color-text-subtle);
        }
        .eyebrow.accent { color: var(--batch-accent, var(--color-brand)); }
        .mono { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }

        .batch-progress-head {
            display: flex; justify-content: space-between; align-items: flex-end;
            margin-bottom: 12px;
        }
        .batch-progress-pct {
            font-size: 28px; font-weight: 900; font-style: italic;
            margin-top: 4px; color: var(--color-text-primary);
            font-variant-numeric: tabular-nums;
        }
        .batch-progress-queue { text-align: right; }

        .batch-progress-bar {
            height: 10px;
            background: var(--color-base);
            border: 1px solid var(--color-border-subtle);
            border-radius: 999px;
            overflow: hidden;
            padding: 2px;
        }
        .batch-progress-bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg,
                var(--batch-accent, var(--color-brand)),
                color-mix(in oklab, var(--batch-accent, var(--color-brand)) 60%, white));
            box-shadow: 0 0 10px color-mix(in oklab, var(--batch-accent, var(--color-brand)) 50%, transparent);
            transition: width 300ms;
        }

        .batch-progress-cur { display: flex; align-items: center; gap: 10px; margin-top: 12px; }
        .batch-progress-cur .mono { font-size: 11.5px; color: var(--color-text-secondary); }
        .batch-progress-hint {
            font-size: 11px; color: var(--color-text-muted); margin-top: 8px; font-style: italic;
        }

        @media (prefers-reduced-motion: reduce) {
            .batch-progress-bar-fill { transition: none; }
        }
    `],
})
export class BatchProgressComponent {
    /** Accent eyebrow above the percent readout (e.g. "NEURAL PROCESSING"). */
    label = input.required<string>();
    /** Items processed so far — shown in the queue readout. */
    current = input<number>(0);
    /** Total items in the batch — shown in the queue readout. */
    total = input<number>(0);
    /**
     * Explicit completion percentage (0–100). When null, the panel derives it
     * from `current`/`total`. Callers that already compute a rounded pct pass it
     * here so the shared panel matches their existing readout exactly.
     */
    percent = input<number | null>(null);
    /** Current item label (e.g. the frame filename). */
    currentItem = input<string>('');
    /** Whether to render the "current item" row. */
    showCurrent = input<boolean>(true);
    /** Eyebrow above the queue count. */
    queueLabel = input<string>('QUEUE STATUS');
    /** Eyebrow beside the current item. */
    currentLabel = input<string>('CURRENT FRAME');
    /** Optional italic footnote (e.g. "Runs in the background…"). */
    hint = input<string | null>(null);
    /** Per-domain accent color; sets the `--batch-accent` custom property. */
    accent = input<string | null>(null);

    protected pct = computed<number>(() => {
        const explicit = this.percent();
        if (explicit !== null && explicit !== undefined) return explicit;
        const total = this.total();
        return total > 0 ? Math.round((this.current() / total) * 100) : 0;
    });
}
