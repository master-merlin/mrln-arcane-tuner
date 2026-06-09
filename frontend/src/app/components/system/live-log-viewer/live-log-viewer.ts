import { Component, OnInit, ElementRef, viewChild, signal, inject, effect, input, computed, DestroyRef, ChangeDetectionStrategy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { ToastService } from '../../../services/toast';
import { of, catchError } from 'rxjs';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { WebSocketService } from '../../../services/websocket.service';
import { IcoComponent } from '../../../icons/ico.component';

type LogLevel = 'INFO' | 'ERROR' | 'WARNING' | 'DEBUG' | 'CRITICAL' | 'UNKNOWN';

interface LogSegment {
    text: string;
    kind: 'bracket' | 'num' | 'text';
}

interface ParsedLog {
    raw: string;
    level: LogLevel;
    formatted: string;
    segments: LogSegment[];
}

const LEVEL_CHIPS: { key: Exclude<LogLevel, 'UNKNOWN'>; tone: string }[] = [
    { key: 'INFO', tone: 'success' },
    { key: 'WARNING', tone: 'warning' },
    { key: 'ERROR', tone: 'danger' },
    { key: 'DEBUG', tone: 'brand' },
    { key: 'CRITICAL', tone: 'danger' },
];

/**
 * Live server-log viewer, redesigned to the Hi-Fi `ServerLogs` card
 * (audit 09 Theme-F): DS `.card` chrome with a control header (free-text
 * filter, per-level chips, follow-tail + word-wrap toggles, clear) and a
 * dark monospace body with per-level row accents + lightweight token
 * colouring. Streaming / history-fetch / parsing logic is unchanged from
 * the legacy component; only presentation + the new controls are added.
 */
@Component({
    selector: 'app-live-log-viewer',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [IcoComponent],
    template: `
    <div class="card log-card">
        <div class="card-head">
            <div class="card-title">
                <app-ico name="ScrollText" [size]="11" /> Server logs
                <span class="log-count">{{ filteredLogs().length }}/{{ parsedLogs().length }}</span>
            </div>
            <div class="log-controls">
                <div class="log-search">
                    <app-ico name="Search" [size]="11" />
                    <input class="input mono log-search-input" placeholder="filter…"
                           [value]="query()" (input)="onQuery($event)" />
                </div>
                <div class="log-level-chips">
                    @for (lv of levels; track lv.key) {
                        <button type="button" class="chip log-level-chip"
                                [style.opacity]="filters()[lv.key] ? 1 : 0.4"
                                (click)="toggleFilter(lv.key)">
                            <span class="dot" [style.background]="filters()[lv.key] ? toneColor(lv.tone) : 'var(--color-text-disabled)'"></span>
                            {{ lv.key }}
                        </button>
                    }
                </div>
                <span class="log-sep"></span>
                <button type="button" class="btn sm log-follow" [class.on]="follow()"
                        (click)="follow.set(!follow())"
                        [title]="follow() ? 'Auto-scrolling on new entries' : 'Click to follow tail'">
                    <span class="follow-dot" [class.on]="follow()"></span> Follow
                </button>
                <button type="button" class="btn sm" (click)="wrap.set(!wrap())"
                        [title]="wrap() ? 'Disable word wrap' : 'Enable word wrap'">
                    <app-ico [name]="wrap() ? 'Minus' : 'WrapText'" [size]="12" /> Wrap
                </button>
                <button type="button" class="btn sm" (click)="downloadLogs()"
                        [disabled]="parsedLogs().length === 0"
                        title="Download all logs as a file" data-testid="log-download">
                    <app-ico name="Download" [size]="12" />
                </button>
                <button type="button" class="btn sm" (click)="clearLogs()"
                        [disabled]="clearing()" title="Clear all entries">
                    <app-ico name="Trash2" [size]="12" />
                </button>
            </div>
        </div>

        <div class="log-body" [class.nowrap]="!wrap()" data-testid="log-viewer-container" #terminalContainer>
            @for (item of filteredLogs(); track $index) {
                <div class="log-row" [attr.data-level]="item.level" data-testid="log-line">
                    <span class="ln">{{ $index + 1 }}</span>
                    <span class="lvl" [style.color]="toneColor(levelTone(item.level))">{{ levelLabel(item.level) }}</span>
                    <span class="msg">@for (seg of item.segments; track $index) {<span [class]="'seg-' + seg.kind">{{ seg.text }}</span>}</span>
                </div>
            }
            @if (filteredLogs().length === 0 && parsedLogs().length > 0) {
                <div class="log-empty">No log entries match the current filters.</div>
            }
            @if (parsedLogs().length === 0) {
                <div class="log-empty" data-testid="log-empty">Connecting to server logs…</div>
            }
        </div>
    </div>
  `,
    styles: [`
        /* Fill the host slot so the body can grow to all available height.
           Falls back gracefully (min-height on the body) if ever dropped into
           a container without a definite height. */
        :host { display: flex; flex-direction: column; flex: 1 1 auto; min-height: 0; }
        .log-card { padding: 0; display: flex; flex-direction: column; flex: 1 1 auto; min-height: 0; }
        .log-controls { display: flex; align-items: center; gap: 8px; }
        .log-search { position: relative; display: flex; align-items: center; }
        .log-search app-ico { position: absolute; left: 8px; color: var(--color-text-muted); pointer-events: none; display: flex; }
        .log-search-input { padding-left: 24px; font-size: 11px; height: 26px; width: 150px; }
        .log-count { font-size: 10px; color: var(--color-text-muted); font-weight: 500; font-family: var(--font-mono); margin-left: 6px; }
        .log-level-chips { display: flex; align-items: center; gap: 4px; }
        .log-level-chip { padding: 2px 7px; font-size: 10px; cursor: pointer; user-select: none; }
        .log-sep { width: 1px; height: 18px; background: var(--color-border-subtle); }
        .log-follow.on { background: oklch(0.68 0.14 155 / 0.14); color: var(--color-success); border-color: oklch(0.68 0.14 155 / 0.30); }
        .follow-dot { width: 6px; height: 6px; border-radius: 999px; background: var(--color-text-muted); }
        .follow-dot.on { background: var(--color-success); box-shadow: 0 0 6px var(--color-success); }

        .log-body {
            padding: 4px 0 6px;
            background: var(--color-terminal-bg);
            /* Grow to fill the card; min-height is a floor for the case where
               an ancestor provides no definite height. */
            flex: 1 1 auto;
            min-height: 240px;
            overflow-y: auto;
            scroll-behavior: smooth;
            border-bottom-left-radius: var(--radius-lg, 10px);
            border-bottom-right-radius: var(--radius-lg, 10px);
        }
        .log-row {
            display: flex;
            align-items: baseline;
            gap: 10px;
            padding: 2px 14px;
            font-family: var(--font-mono);
            font-size: 11.5px;
            line-height: 1.55;
            border-left: 2px solid transparent;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .log-body.nowrap .log-row { white-space: nowrap; }
        .log-body.nowrap { overflow-x: auto; }
        .log-row:nth-child(even) { background: var(--color-terminal-row-alt); }
        .log-row[data-level="WARNING"]  { border-left-color: var(--color-warning); }
        .log-row[data-level="ERROR"]    { border-left-color: var(--color-danger); }
        .log-row[data-level="CRITICAL"] { border-left-color: var(--color-danger); background: oklch(0.70 0.17 25 / 0.10); }
        .log-row .ln { flex: 0 0 auto; color: var(--color-text-disabled); font-size: 10px; user-select: none; min-width: 26px; text-align: right; }
        .log-row .lvl { flex: 0 0 auto; font-size: 9.5px; font-weight: 700; letter-spacing: 0.04em; min-width: 62px; white-space: nowrap; }
        .log-row .msg { flex: 1 1 auto; min-width: 0; color: var(--color-text-secondary); }
        .seg-bracket { color: var(--color-chart-lr); }
        .seg-num { color: var(--color-brand); font-weight: 600; }

        .log-empty { padding: 22px 16px; text-align: center; color: var(--color-text-disabled); font-size: 11px; font-family: var(--font-mono); font-style: italic; }
    `],
})
export class LiveLogViewerComponent implements OnInit {
    readonly mode = input<'server' | 'training'>('server');

    logs = signal<string[]>([]);
    readonly levels = LEVEL_CHIPS;

    // Filters State
    filters = signal<{ INFO: boolean; ERROR: boolean; WARNING: boolean; DEBUG: boolean; CRITICAL: boolean }>({
        INFO: true,
        ERROR: true,
        WARNING: true,
        DEBUG: false,
        CRITICAL: true,
    });

    query = signal('');
    follow = signal(true);
    wrap = signal(false);
    clearing = signal(false);

    // Processed logs (parsed once for performance)
    parsedLogs = computed<ParsedLog[]>(() =>
        this.logs().map(line => {
            const { level, formatted } = this.parseLine(line);
            return { raw: line, level, formatted, segments: this.tokenize(formatted) };
        }),
    );

    // Filtered view (level chips + free-text query)
    filteredLogs = computed<ParsedLog[]>(() => {
        const f = this.filters();
        const q = this.query().trim().toLowerCase();
        return this.parsedLogs().filter(item => {
            if (item.level !== 'UNKNOWN' && !f[item.level]) return false;
            if (q && !item.formatted.toLowerCase().includes(q)) return false;
            return true;
        });
    });

    private http = inject(HttpClient);
    private wsService = inject(WebSocketService);
    private rtc = inject(RuntimeConfigService);
    private toast = inject(ToastService);
    private destroyRef = inject(DestroyRef);

    terminalContainer = viewChild<ElementRef>('terminalContainer');

    constructor() {
        effect(() => {
            const logs = this.filteredLogs(); // track filtered logs
            if (logs.length > 0 && this.follow()) {
                this.scrollToBottom();
            }
        });
    }

    ngOnInit() {
        // Initial fetch for history
        this.fetchLogs().subscribe(logs => this.logs.set(logs));

        // Subscribe to real-time logs
        this.wsService.on<{ message: string }>('server_log').pipe(
            takeUntilDestroyed(this.destroyRef),
        ).subscribe(payload => {
            // payload.message is the raw JSON string from structlog
            this.handleLogEvent(payload.message);
        });
    }

    handleLogEvent(logLine: string) {
        this.logs.update(current => [...current, logLine]);
    }

    fetchLogs() {
        return this.http.get<string[]>(`${this.rtc.apiUrl}/system/logs?lines=200`).pipe(
            catchError(err => {
                console.error(err);
                return of([]);
            }),
        );
    }

    onQuery(e: Event) {
        this.query.set((e.target as HTMLInputElement).value);
    }

    toggleFilter(type: 'INFO' | 'ERROR' | 'WARNING' | 'DEBUG' | 'CRITICAL') {
        this.filters.update(f => ({ ...f, [type]: !f[type] }));
    }

    /**
     * Export the full in-memory log buffer (raw structlog lines, unaffected
     * by the on-screen level/text filters) as a downloadable file. Lets a
     * user hand off complete logs for debugging when a remote session into
     * the pod isn't available.
     */
    downloadLogs() {
        const lines = this.logs();
        if (lines.length === 0) {
            this.toast.error('No logs to download');
            return;
        }
        const blob = new Blob([lines.join('\n') + '\n'], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        // Filesystem-safe timestamp: 2026-06-09T07-12-30
        const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const a = document.createElement('a');
        a.href = url;
        a.download = `mrln-${this.mode()}-logs-${ts}.log`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        this.toast.success(`Downloaded ${lines.length} log line${lines.length === 1 ? '' : 's'}`);
    }

    clearLogs() {
        if (!confirm('Are you sure you want to clear the server logs?')) return;
        this.clearing.set(true);
        this.http.post<{ message?: string; error?: string }>(`${this.rtc.apiUrl}/system/logs/clear`, {}).subscribe({
            next: (res) => {
                this.clearing.set(false);
                if (res.error) {
                    this.toast.error(`Error: ${res.error}`);
                } else {
                    this.logs.set([]);
                    this.toast.success(res.message || 'Logs cleared successfully');
                }
            },
            error: (err) => {
                this.clearing.set(false);
                this.toast.error('Failed to clear logs');
                console.error(err);
            },
        });
    }

    toneColor(tone: string): string {
        return `var(--color-${tone === 'brand' ? 'brand' : tone})`;
    }

    levelTone(level: LogLevel): string {
        switch (level) {
            case 'INFO': return 'success';
            case 'WARNING': return 'warning';
            case 'ERROR':
            case 'CRITICAL': return 'danger';
            case 'DEBUG': return 'brand';
            default: return 'text-muted';
        }
    }

    levelLabel(level: LogLevel): string {
        return level === 'UNKNOWN' ? 'LOG' : level;
    }

    /** Split a formatted line into bracket / number / text segments for colouring. */
    private tokenize(line: string): LogSegment[] {
        const out: LogSegment[] = [];
        const re = /(\[[^\]]+\])|(\b\d[\d.,:/eE+\-]*\b)/g;
        let last = 0;
        let m: RegExpExecArray | null;
        while ((m = re.exec(line)) !== null) {
            if (m.index > last) out.push({ text: line.slice(last, m.index), kind: 'text' });
            out.push({ text: m[0], kind: m[1] ? 'bracket' : 'num' });
            last = m.index + m[0].length;
        }
        if (last < line.length) out.push({ text: line.slice(last), kind: 'text' });
        return out.length ? out : [{ text: line, kind: 'text' }];
    }

    parseLine(line: string): { level: LogLevel, formatted: string } {
        let level: LogLevel = 'UNKNOWN';
        let formatted = line;

        // 1. Try JSON Parse
        if (line.trim().startsWith('{')) {
            try {
                const obj = JSON.parse(line);

                // Extract Level
                if (obj.level) {
                    const l = obj.level.toUpperCase();
                    if (l.includes('INFO')) level = 'INFO';
                    else if (l.includes('CRIT') || l.includes('FATAL')) level = 'CRITICAL';
                    else if (l.includes('ERR')) level = 'ERROR';
                    else if (l.includes('WARN')) level = 'WARNING';
                    else if (l.includes('DEBUG')) level = 'DEBUG';
                    else level = 'INFO';
                }

                // Format
                const timestamp = obj.timestamp ? `[${obj.timestamp.split('T')[1].split('.')[0]}] ` : '';
                const lvlStr = level !== 'UNKNOWN' ? `[${level}] ` : (obj.level ? `[${obj.level.toUpperCase()}] ` : '');
                const event = obj.event || obj.message || JSON.stringify(obj);

                const { timestamp: _, level: __, event: ___, ...rest } = obj;
                const restParam = Object.keys(rest).length > 0 ? ` ${JSON.stringify(rest)}` : '';

                formatted = `${timestamp}${lvlStr}${event}${restParam}`;

                if (level === 'UNKNOWN') {
                    // Check common strings in event if validation failed
                    if (event.toLowerCase().includes('error')) level = 'ERROR';
                }

            } catch (e) {
                // Fallback text parsing
                formatted = line;
            }
        }

        // 2. Fallback Text Parsing if still UNKNOWN (or was not JSON)
        if (level === 'UNKNOWN') {
            const lower = line.toLowerCase();
            if (lower.includes('error') || lower.includes('failed') || lower.includes('exception')) level = 'ERROR';
            else if (lower.includes('warn')) level = 'WARNING';
            else if (lower.includes('debug')) level = 'DEBUG';
            else level = 'INFO'; // Default to INFO for standard lines
        }

        return { level, formatted };
    }

    scrollToBottom() {
        setTimeout(() => {
            const el = this.terminalContainer();
            if (el) {
                el.nativeElement.scrollTop = el.nativeElement.scrollHeight;
            }
        }, 100);
    }
}
