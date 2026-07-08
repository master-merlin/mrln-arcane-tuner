import {
    ChangeDetectionStrategy,
    Component,
    computed,
    effect,
    ElementRef,
    inject,
    input,
    signal,
    viewChild,
} from '@angular/core';

import { ToastService } from '../../services/toast';
import type { LogLine } from '../../shared/job-metrics';

/**
 * T5 — a proper, lightweight job-log viewer (replaces the old 14-line tail).
 *
 * Renders a scrollable log region with real scrollback plus a free-text filter,
 * Copy + Download actions and an auto-Follow (stick-to-bottom) toggle that
 * disengages the moment the user scrolls up and re-engages once they scroll
 * back to the bottom. It borrows the presentation *idea* of the server-log
 * viewer (`components/system/live-log-viewer`) but is a self-contained,
 * dependency-light component local to the Jobs screen — it takes pre-classified
 * `LogLine[]` and owns no streaming/service machinery.
 */

/** Distance (px) from the bottom within which we consider the view "at bottom". */
const FOLLOW_THRESHOLD = 24;

@Component({
    selector: 'app-job-log-viewer',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="jlog">
            <div class="jlog-controls">
                <div class="jlog-search">
                    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                    <input
                        class="input mono jlog-search-input"
                        type="text"
                        placeholder="filter log…"
                        aria-label="Filter log lines"
                        [value]="query()"
                        (input)="onQuery($event)"
                        data-testid="job-log-filter"
                    />
                </div>
                <span class="jlog-count mono">{{ filtered().length }}/{{ lines().length }}</span>
                <span class="jlog-spacer"></span>
                <button
                    type="button"
                    class="btn sm jlog-follow"
                    [class.on]="follow()"
                    [attr.aria-pressed]="follow()"
                    (click)="toggleFollow()"
                    [title]="follow() ? 'Following new lines — click to stop' : 'Click to follow new lines'"
                    data-testid="job-log-follow"
                >
                    <span class="follow-dot" [class.on]="follow()"></span> Follow
                </button>
                <button
                    type="button"
                    class="btn sm"
                    (click)="copy()"
                    [disabled]="filtered().length === 0"
                    title="Copy the shown log lines to the clipboard"
                    data-testid="job-log-copy"
                >
                    Copy
                </button>
                <button
                    type="button"
                    class="btn sm"
                    (click)="download()"
                    [disabled]="filtered().length === 0"
                    title="Download the shown log lines as a .txt file"
                    data-testid="job-log-download"
                >
                    Download
                </button>
            </div>

            <div class="jlog-body mono" #logBody (scroll)="onScroll($event)" data-testid="job-log-body">
                @for (line of filtered(); track $index) {
                    <div class="jlog-line" data-testid="job-log-line">
                        <span
                            class="tag"
                            [class.teal]="line.tone === 'teal'"
                            [class.warning]="line.tone === 'warning'"
                            [class.danger]="line.tone === 'danger'"
                            >{{ line.level }}</span
                        >
                        <span class="jlog-text">{{ line.text }}</span>
                    </div>
                } @empty {
                    <p class="jlog-empty">
                        @if (lines().length) {
                            No log lines match the filter.
                        } @else {
                            No log output yet.
                        }
                    </p>
                }
            </div>
        </div>
    `,
    styles: [
        `
            .jlog {
                display: flex;
                flex-direction: column;
                gap: 8px;
                padding: 10px 14px 12px;
            }
            .jlog-controls {
                display: flex;
                align-items: center;
                gap: 8px;
                flex-wrap: wrap;
            }
            .jlog-search {
                position: relative;
                display: flex;
                align-items: center;
            }
            .jlog-search svg {
                position: absolute;
                left: 8px;
                color: var(--color-text-muted);
                pointer-events: none;
            }
            .jlog-search-input {
                padding-left: 26px;
                font-size: 11px;
                height: 26px;
                width: 190px;
            }
            .jlog-count {
                font-size: 10px;
                color: var(--color-text-muted);
                font-weight: 500;
            }
            .jlog-spacer {
                flex: 1 1 auto;
            }
            .jlog-follow.on {
                background: oklch(0.68 0.14 155 / 0.14);
                color: var(--color-success);
                border-color: oklch(0.68 0.14 155 / 0.3);
            }
            .follow-dot {
                display: inline-block;
                width: 6px;
                height: 6px;
                border-radius: 999px;
                background: var(--color-text-muted);
                margin-right: 2px;
            }
            .follow-dot.on {
                background: var(--color-success);
                box-shadow: 0 0 6px var(--color-success);
            }

            .jlog-body {
                background: oklch(0.05 0.01 265);
                border: 1px solid var(--color-border-subtle);
                border-radius: var(--radius-theme-md);
                font-size: 11px;
                line-height: 1.7;
                /* Real scrollback — much taller than the old ~240px 14-line tail. */
                height: 420px;
                max-height: 60vh;
                overflow: auto;
                padding: 8px 12px;
                color: var(--color-text-secondary);
                scroll-behavior: smooth;
            }
            @media (prefers-reduced-motion: reduce) {
                .jlog-body {
                    scroll-behavior: auto;
                }
            }
            .jlog-line {
                display: flex;
                gap: 8px;
                align-items: flex-start;
            }
            .jlog-line .tag {
                min-width: 46px;
                justify-content: center;
                flex-shrink: 0;
            }
            .jlog-text {
                flex: 1;
                white-space: pre-wrap;
                word-break: break-all;
            }
            .jlog-empty {
                margin: 0;
                padding: 18px 4px;
                text-align: center;
                color: var(--color-text-disabled);
                font-style: italic;
            }
        `,
    ],
})
export class JobLogViewerComponent {
    /** Pre-classified log lines (oldest → newest). */
    readonly lines = input<LogLine[]>([]);
    /** Basename for the downloaded file (`<name>-log-<ts>.txt`). */
    readonly downloadName = input<string>('job');

    protected readonly query = signal<string>('');
    /** Auto stick-to-bottom; disengages when the user scrolls up. */
    protected readonly follow = signal<boolean>(true);

    private readonly logBody = viewChild<ElementRef<HTMLElement>>('logBody');
    private readonly toast = inject(ToastService);

    protected readonly filtered = computed<LogLine[]>(() => {
        const q = this.query().trim().toLowerCase();
        const src = this.lines();
        if (!q) return src;
        return src.filter((l) => l.text.toLowerCase().includes(q));
    });

    constructor() {
        // Stick to the bottom whenever new (filtered) lines arrive and Follow is on.
        effect(() => {
            this.filtered(); // track new content
            if (this.follow()) this.scrollToBottom();
        });
    }

    protected onQuery(e: Event): void {
        this.query.set((e.target as HTMLInputElement).value);
    }

    protected toggleFollow(): void {
        const next = !this.follow();
        this.follow.set(next);
        if (next) this.scrollToBottom();
    }

    /** Disengage follow when the user scrolls up; re-engage at the bottom. */
    protected onScroll(e: Event): void {
        const el = e.target as HTMLElement;
        const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < FOLLOW_THRESHOLD;
        if (atBottom) {
            if (!this.follow()) this.follow.set(true);
        } else if (this.follow()) {
            this.follow.set(false);
        }
    }

    protected copy(): void {
        const text = this.filtered().map((l) => l.text).join('\n');
        if (!text) return;
        const clip = navigator.clipboard;
        if (clip?.writeText) {
            clip.writeText(text).then(
                () => this.toast.success(`Copied ${this.filtered().length} log line(s)`),
                () => this.toast.error('Copy failed'),
            );
        } else {
            this.toast.error('Clipboard unavailable');
        }
    }

    protected download(): void {
        const rows = this.filtered();
        if (!rows.length) return;
        const blob = new Blob([rows.map((l) => l.text).join('\n') + '\n'], {
            type: 'text/plain;charset=utf-8',
        });
        const url = URL.createObjectURL(blob);
        // Filesystem-safe timestamp: 2026-07-08T23-40-12
        const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${this.downloadName()}-log-${ts}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    private scrollToBottom(): void {
        setTimeout(() => {
            const el = this.logBody()?.nativeElement;
            if (el) el.scrollTop = el.scrollHeight;
        });
    }
}
