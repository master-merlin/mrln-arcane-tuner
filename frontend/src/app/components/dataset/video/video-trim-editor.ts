import { ChangeDetectionStrategy, Component, computed, effect, input, output, signal } from '@angular/core';
import { IcoComponent } from '../../../icons/ico.component';
import { FRAME_FAMILIES, estimateFrames, passesFamily } from './frame-rules';

/** Committed trim window (seconds); nulls mean "no bound on that side". */
export interface TrimChange { start: number | null; end: number | null }

/**
 * Dual-thumb trim editor for a single video clip.
 *
 * Two range inputs select the effective [start, end] window inside the clip's
 * duration. The editor shows the effective frame count and per-family pass/fail
 * chips (4n+1 / 8n+1) computed inline from the window, plus "set from playhead"
 * buttons that snap a bound to the supplied `currentTime`.
 *
 * Commit discipline: {@link trimChanged} fires ONLY on commit (pointerup / blur),
 * never per-drag — the parent persists on commit and the inline chips give live
 * feedback while dragging. A bound equal to the clip extent (0 / duration) is
 * emitted as `null` so a "full clip" trim clears the stored bound.
 */
@Component({
    selector: 'app-video-trim-editor',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="vte" data-testid="video-trim-editor">
            <div class="vte-head">
                <span class="eyebrow">TRIM</span>
                <span class="mono vte-window" data-testid="vte-window">
                    {{ fmt(start()) }} – {{ fmt(end()) }}
                </span>
            </div>

            <div class="vte-track">
                <input type="range" class="vte-range start"
                       data-testid="vte-range-start"
                       [min]="0" [max]="duration()" step="0.05"
                       [value]="start()"
                       (input)="onStartInput($any($event.target).value)"
                       (pointerup)="commit()" (keyup.enter)="commit()" (blur)="commit()"
                       aria-label="Trim start">
                <input type="range" class="vte-range end"
                       data-testid="vte-range-end"
                       [min]="0" [max]="duration()" step="0.05"
                       [value]="end()"
                       (input)="onEndInput($any($event.target).value)"
                       (pointerup)="commit()" (keyup.enter)="commit()" (blur)="commit()"
                       aria-label="Trim end">
            </div>

            <div class="vte-playhead">
                <button type="button" class="btn-tiny"
                        data-testid="vte-set-start"
                        [disabled]="currentTime() == null"
                        title="Set start from the current playhead"
                        (click)="setStartFromPlayhead()">
                    <app-ico name="ChevronFirst" [size]="12"/> Set in
                </button>
                <button type="button" class="btn-tiny"
                        data-testid="vte-set-end"
                        [disabled]="currentTime() == null"
                        title="Set end from the current playhead"
                        (click)="setEndFromPlayhead()">
                    Set out <app-ico name="ChevronLast" [size]="12"/>
                </button>
                <button type="button" class="btn-tiny ghost"
                        data-testid="vte-reset"
                        title="Reset to the full clip"
                        (click)="resetFull()">
                    <app-ico name="RotateCcw" [size]="12"/> Full
                </button>
            </div>

            <div class="vte-stats">
                <div class="vte-frames">
                    <span class="eyebrow">FRAMES</span>
                    <span class="mono" data-testid="vte-frames">{{ effectiveFrames() || '—' }}</span>
                </div>
                <div class="vte-chips">
                    @for (f of families; track f.label; let i = $index) {
                        <span class="chip"
                              [class.pass]="familyPass()[i]"
                              [class.fail]="!familyPass()[i]"
                              [attr.data-testid]="'vte-chip-' + f.modulus"
                              [title]="f.label + (familyPass()[i] ? ' OK' : ' fails')">
                            {{ f.label }}
                        </span>
                    }
                </div>
            </div>
        </div>
    `,
    styles: [`
        :host { display: block; }
        .vte {
            padding: 12px 14px;
            background: var(--color-surface-low);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-lg);
            display: flex; flex-direction: column; gap: 10px;
        }
        .vte-head { display: flex; align-items: center; justify-content: space-between; }
        .eyebrow {
            font-size: 9.5px; font-weight: 700; letter-spacing: 0.12em;
            text-transform: uppercase; color: var(--color-text-subtle);
        }
        .vte-window { font-size: 12px; color: var(--color-text-primary); }
        .vte-track { position: relative; height: 22px; }
        .vte-range {
            position: absolute; left: 0; right: 0; top: 0;
            width: 100%; margin: 0; background: transparent;
            accent-color: var(--color-chart-lr);
            pointer-events: auto;
        }
        .vte-range.end { accent-color: var(--color-brand); }
        .vte-playhead { display: flex; gap: 6px; }
        .btn-tiny {
            display: inline-flex; align-items: center; gap: 4px;
            padding: 4px 8px; font-size: 11px; font-weight: 600;
            border-radius: var(--radius-theme-md);
            border: 1px solid var(--color-border-subtle);
            background: var(--color-surface-mid); color: var(--color-text-secondary);
            cursor: pointer;
        }
        .btn-tiny:hover:not(:disabled) { color: var(--color-text-primary); }
        .btn-tiny:disabled { opacity: 0.45; cursor: not-allowed; }
        .btn-tiny.ghost { margin-left: auto; }
        .vte-stats { display: flex; align-items: center; justify-content: space-between; }
        .vte-frames { display: flex; align-items: center; gap: 8px; }
        .vte-frames .mono { font-size: 13px; color: var(--color-text-primary); }
        .vte-chips { display: inline-flex; gap: 6px; }
        .chip {
            font-size: 10px; font-weight: 700; letter-spacing: 0.04em;
            padding: 2px 7px; border-radius: 999px;
            border: 1px solid var(--color-border-subtle);
        }
        .chip.pass {
            color: var(--color-success);
            background: color-mix(in oklab, var(--color-success) 12%, transparent);
            border-color: color-mix(in oklab, var(--color-success) 35%, transparent);
        }
        .chip.fail { color: var(--color-text-muted); background: var(--color-surface-mid); }
    `],
})
export class VideoTrimEditorComponent {
    /** Clip duration in seconds — the slider extent. */
    duration = input.required<number>();
    /** Source fps — drives the effective frame count + family verdicts. */
    fps = input<number | undefined>(undefined);
    /** Stored trim bounds (seconds); null = no bound (full clip on that side). */
    trimStartS = input<number | null>(null);
    trimEndS = input<number | null>(null);
    /** Optional live playhead time (seconds) from the host `<video>`; enables
     *  the "set in/out" buttons. */
    currentTime = input<number | null>(null);

    /** Fires ONLY on commit (pointerup / Enter / blur / button click). */
    trimChanged = output<TrimChange>();

    protected readonly families = FRAME_FAMILIES;

    /** Live (drag) window — initialized from the inputs, updated per-input,
     *  committed on pointerup. */
    protected start = signal<number>(0);
    protected end = signal<number>(0);

    constructor() {
        // Re-seed the live window whenever the bound inputs or duration change
        // (navigating to another clip, or an external save reconciles the pair).
        effect(() => {
            const dur = this.duration();
            const s = this.trimStartS();
            const e = this.trimEndS();
            this.start.set(this.clamp(s ?? 0, dur));
            this.end.set(this.clamp(e ?? dur, dur));
        });
    }

    protected effectiveFrames = computed<number>(() =>
        estimateFrames(this.start(), this.end(), this.fps()),
    );

    protected familyPass = computed<boolean[]>(() => {
        const frames = this.effectiveFrames();
        return this.families.map(f => passesFamily(frames, f));
    });

    protected fmt(s: number): string {
        if (!Number.isFinite(s) || s < 0) return '0:00.0';
        const m = Math.floor(s / 60);
        const sec = s - m * 60;
        return `${m}:${sec.toFixed(1).padStart(4, '0')}`;
    }

    private clamp(v: number, dur: number): number {
        if (!Number.isFinite(v)) return 0;
        return Math.max(0, Math.min(v, dur));
    }

    /** Start can't cross end. */
    protected onStartInput(value: string | number): void {
        const v = typeof value === 'string' ? parseFloat(value) : value;
        this.start.set(Math.min(this.clamp(v, this.duration()), this.end()));
    }

    /** End can't cross start. */
    protected onEndInput(value: string | number): void {
        const v = typeof value === 'string' ? parseFloat(value) : value;
        this.end.set(Math.max(this.clamp(v, this.duration()), this.start()));
    }

    protected setStartFromPlayhead(): void {
        const t = this.currentTime();
        if (t == null) return;
        this.start.set(Math.min(this.clamp(t, this.duration()), this.end()));
        this.commit();
    }

    protected setEndFromPlayhead(): void {
        const t = this.currentTime();
        if (t == null) return;
        this.end.set(Math.max(this.clamp(t, this.duration()), this.start()));
        this.commit();
    }

    protected resetFull(): void {
        this.start.set(0);
        this.end.set(this.duration());
        this.commit();
    }

    /**
     * Emit the committed window. A bound equal to the clip extent collapses to
     * `null` (clears the stored bound — "full clip on that side"), matching the
     * backend's nullable trim contract.
     */
    protected commit(): void {
        const dur = this.duration();
        const s = this.start();
        const e = this.end();
        this.trimChanged.emit({
            start: s <= 0 ? null : s,
            end: e >= dur ? null : e,
        });
    }
}
