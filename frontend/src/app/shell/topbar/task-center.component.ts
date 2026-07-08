import { ChangeDetectionStrategy, Component, ElementRef, HostListener, inject } from '@angular/core';
import { IcoComponent } from '../../icons/ico.component';
import { Task, TaskStore } from '../../state/task.store';
import { TopbarPanelStore } from '../../state/topbar-panel.store';

/** How a task row presents: a human kind label + an accent color token. */
interface TaskView { kind: string; subject: string; accent: string; }

/**
 * Per-type presentation. `kind` is the tier-1 label; `accent` is a CSS color
 * token reused from the KPI accent rails (brand / success / warning / danger /
 * teal=chart-lr / violet). New task types (crop/scoring/rescan/adjustments)
 * slot in here as the background-task framework absorbs them.
 */
const TASK_KINDS: Record<string, { kind: string; accent: string }> = {
    caption_batch:       { kind: 'Captioning',  accent: 'var(--color-brand)' },
    // Lighter than --color-violet (oklch L=0.65, too dark to read as a tiny
    // uppercase label); L=0.74 matches the readability of brand.
    rescan_batch:        { kind: 'Rescan',       accent: 'oklch(0.74 0.15 295)' },
    // Mass masking offload — generate (AI mask) then apply (composite). Green
    // to match the mass-mask modal's success theme.
    mask_generate_batch: { kind: 'Masking',      accent: 'var(--color-success)' },
    mask_apply_batch:    { kind: 'Apply Masks',  accent: 'var(--color-success)' },
    // Video curation — splitting a source video into clips + scene detection.
    video_split:         { kind: 'Clip Split',   accent: 'var(--color-chart-lr)' },
    scene_detect:        { kind: 'Scene Detect',  accent: 'oklch(0.74 0.12 200)' },
    // Future task types slot in here as the background-task framework absorbs
    // them — e.g. scoring (var(--color-warning)), crop (var(--color-chart-lr)).
    // Until a type is mapped it falls back to a neutral rail + a label parsed
    // from the "<Kind> · <subject>" title.
};

@Component({
    selector: 'app-task-center',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
            <div class="tc">
                <button class="tc-pill" type="button" (click)="toggle()"
                        data-testid="task-center-trigger"
                        [class.busy]="count() > 0" [class.idle]="count() === 0"
                        [attr.aria-label]="count() > 0
                            ? count() + ' background tasks running'
                            : 'Activity — background tasks'"
                        [attr.aria-expanded]="open()" title="Background tasks">
                    <app-ico [name]="count() > 0 ? 'Loader2' : 'Activity'" [size]="14"/>
                    @if (count() > 0) { <span class="tc-count mono">{{ count() }}</span> }
                </button>
                @if (open()) {
                    <div class="tc-pop">
                        <div class="tc-head">
                            <span>Activity</span>
                            @if (recent().length > 0) {
                                <button class="tc-clear" type="button" (click)="clearRecent()"
                                        data-testid="task-center-clear"
                                        aria-label="Clear recent activity">Clear</button>
                            }
                        </div>
                        @if (active().length === 0 && recent().length === 0) {
                            <div class="tc-empty" data-testid="task-center-empty">
                                All clear · no recent activity
                            </div>
                        }
                        @for (t of active(); track t.id) {
                            @let v = view(t);
                            <div class="tc-row" [style.--accent]="v.accent">
                                <div class="tc-kind">{{ v.kind }}
                                    @if (t.status === 'pending') { <span class="tc-q">· Queued</span> }
                                </div>
                                <div class="tc-subject" [title]="v.subject">{{ v.subject }}</div>
                                <div class="bar"><i [style.width.%]="pct(t)"></i></div>
                                <div class="tc-meta mono">
                                    <span>{{ t.current }} / {{ t.total }}</span>
                                    <button class="tc-cancel" type="button" (click)="cancel(t.id)">Cancel</button>
                                </div>
                                @if (t.current_item) { <div class="tc-item mono">› {{ t.current_item }}</div> }
                            </div>
                        }
                        @if (recent().length > 0) { <div class="tc-sep"></div> }
                        @for (t of recent(); track t.id) {
                            @let v = view(t);
                            <div class="tc-row done" data-testid="task-center-row"
                                 [class.failed]="t.status !== 'completed'"
                                 [style.--accent]="v.accent">
                                <div class="tc-kind" data-testid="task-center-kind">{{ v.kind }}</div>
                                <div class="tc-subject" data-testid="task-center-subject" [title]="v.subject">{{ v.subject }}</div>
                                <div class="tc-detail mono">
                                    <app-ico class="tc-glyph"
                                        [name]="t.status === 'completed' ? 'Check' : 'X'" [size]="12"/>
                                    @if (t.failed > 0) {
                                        <span class="ok">{{ t.ok }} ok</span>
                                        <span class="fail">· {{ t.failed }} failed</span>
                                    } @else {
                                        <span class="ok">{{ t.ok }} done</span>
                                    }
                                    @if (t.status === 'cancelled') { <span class="cancelled">· cancelled</span> }
                                </div>
                                @if (t.status === 'failed' && t.error) {
                                    <div class="tc-error" data-testid="task-center-error"
                                         [title]="t.error">{{ t.error }}</div>
                                }
                            </div>
                        }
                    </div>
                }
            </div>
    `,
    styles: [`
        .tc { position: relative; }
        .tc-pill { display: inline-flex; align-items: center; gap: 5px; padding: 5px 8px;
            border: 1px solid var(--color-border-subtle); border-radius: var(--radius-theme-md);
            background: var(--color-surface-mid); color: var(--color-text-secondary); cursor: pointer; }
        .tc-pill.busy { color: var(--color-brand-light); }
        /* Idle: dimmed but always present — the stable entry point to activity. */
        .tc-pill.idle { color: var(--color-text-subtle); opacity: 0.65; }
        .tc-pill.idle:hover { opacity: 1; color: var(--color-text-secondary); }
        .tc-pill:focus-visible { outline: 2px solid var(--color-brand); outline-offset: 2px; }
        .tc-count { font-size: 11px; font-weight: 700; }
        .tc-pop { position: absolute; right: 0; top: calc(100% + 6px); width: 320px; z-index: 50;
            background: var(--color-surface-low); border: 1px solid var(--color-border-default);
            border-radius: var(--radius-theme-lg); box-shadow: var(--shadow-lg); padding: 10px; }
        .tc-head { display: flex; align-items: center; justify-content: space-between;
            font-size: 10px; font-weight: 700; letter-spacing: 0.10em; text-transform: uppercase;
            color: var(--color-text-subtle); margin-bottom: 8px; }
        .tc-clear { background: none; border: none; cursor: pointer; font: inherit;
            font-size: 10px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
            color: var(--color-text-muted); padding: 2px 4px; border-radius: var(--radius-theme-sm); }
        .tc-clear:hover { color: var(--color-text-secondary); }
        .tc-clear:focus-visible { outline: 2px solid var(--color-brand); outline-offset: 1px; }
        .tc-empty { font-size: 12px; color: var(--color-text-muted); padding: 6px 2px; }
        /* Each row is a 3-tier card — kind / subject / detail — with a left
           accent rail colored per task type (--accent, set inline). Mirrors
           the KPI-tile accent rails. */
        .tc-row { position: relative; padding: 6px 0 6px 12px; }
        .tc-row::before { content: ''; position: absolute; left: 0; top: 5px; bottom: 5px;
            width: 2px; border-radius: 1px; background: var(--accent, var(--color-border-default)); }
        /* Tier 1 — task kind: small, uppercase, in the type's accent tone. */
        .tc-kind { font-size: 9.5px; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
            color: var(--accent, var(--color-text-subtle)); }
        .tc-q { color: var(--color-warning); font-weight: 600; }
        /* Tier 2 — subject (dataset): the primary line, truncates not wraps. */
        .tc-subject { font-size: 12.5px; font-weight: 600; color: var(--color-text-primary);
            margin-top: 1px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .bar { height: 6px; background: var(--color-surface-mid); border-radius: 3px; overflow: hidden; margin: 5px 0; }
        /* Progress fill is always brand — the per-type accent lives on the rail
           and kind label, not the bar, so progress reads consistently. */
        .bar i { display: block; height: 100%; background: var(--color-brand); }
        /* Tier 3 — detail line. */
        .tc-meta { display: flex; justify-content: space-between; font-size: 10.5px; color: var(--color-text-muted); }
        .tc-detail { display: flex; align-items: center; gap: 5px; margin-top: 3px;
            font-size: 10.5px; color: var(--color-text-muted); }
        .tc-glyph { flex-shrink: 0; color: var(--color-success); }
        .tc-row.done.failed .tc-glyph { color: var(--color-danger); }
        .tc-detail .ok { color: var(--color-text-muted); }
        .tc-detail .fail { color: var(--color-danger); font-weight: 600; }
        .tc-detail .cancelled { color: var(--color-text-disabled); }
        .tc-cancel { background: none; border: none; color: var(--color-danger); cursor: pointer; font: inherit; }
        .tc-item { font-size: 10px; color: var(--color-text-disabled); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        /* Failed-task error — surfaced so a failure is inspectable in the panel. */
        .tc-error { font-size: 10.5px; color: var(--color-danger); margin-top: 3px;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .tc-sep { height: 1px; background: var(--color-border-subtle); margin: 8px 0; }
    `],
})
export class TaskCenterComponent {
    private store = inject(TaskStore);
    private host = inject(ElementRef<HTMLElement>);
    private panels = inject(TopbarPanelStore);
    protected active = this.store.active;
    protected recent = this.store.recent;
    protected count = this.store.activeCount;
    protected open = this.panels.isOpen('tasks');

    protected toggle(): void { this.panels.toggle('tasks'); }

    /**
     * Split a task into its three display tiers. Prefers the structured
     * `type`/`dataset_name` fields; falls back to parsing the composed
     * `"<Kind> · <subject>"` title for unmapped types or dataset-less tasks.
     */
    protected view(t: Task): TaskView {
        const known = TASK_KINDS[t.type];
        const [head, ...rest] = (t.title ?? '').split(' · ');
        const tail = rest.join(' · ');
        return {
            kind: known?.kind ?? head ?? t.type,
            subject: t.dataset_name ?? tail ?? head ?? '',
            accent: known?.accent ?? 'var(--color-border-default)',
        };
    }

    protected pct(t: { current: number; total: number }): number {
        return t.total > 0 ? Math.round((t.current / t.total) * 100) : 0;
    }
    protected cancel(id: string): void { this.store.cancel(id); }
    /** Dismiss the recent list; leaves active tasks running. */
    protected clearRecent(): void { this.store.clearRecent(); }

    @HostListener('document:mousedown', ['$event'])
    protected onOutsidePointer(event: MouseEvent): void {
        if (!this.open()) return;
        if (!this.host.nativeElement.contains(event.target as Node)) {
            this.panels.close('tasks');
        }
    }

    @HostListener('document:keydown.escape')
    protected onEsc(): void {
        if (this.open()) this.panels.close('tasks');
    }
}
