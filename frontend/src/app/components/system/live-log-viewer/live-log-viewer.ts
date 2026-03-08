import { Component, OnInit, OnDestroy, ElementRef, viewChild, signal, inject, effect, input, computed } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { Subscription, of, catchError } from 'rxjs';
import { WebSocketService } from '../../../services/websocket.service';

type LogLevel = 'INFO' | 'ERROR' | 'WARNING' | 'DEBUG' | 'CRITICAL' | 'UNKNOWN';

@Component({
    selector: 'app-live-log-viewer',
    standalone: true,
    imports: [],
    template: `
    <div class="flex flex-col gap-4">
        <!-- Filter Controls -->
        <div class="flex flex-wrap gap-2 text-xs font-mono">
            <button (click)="toggleFilter('INFO')" 
                class="px-3 py-1.5 rounded border transition-colors flex items-center gap-2"
                [class.bg-blue-900_30]="filters().INFO" 
                [class.border-blue-500]="filters().INFO"
                [class.text-blue-400]="filters().INFO"
                [class.border-border-default]="!filters().INFO"
                [class.text-text-subtle]="!filters().INFO">
                <div class="w-2 h-2 rounded-full" [class.bg-blue-500]="filters().INFO" [class.bg-surface-high]="!filters().INFO"></div>
                INFO
            </button>
            <button (click)="toggleFilter('WARNING')" 
                class="px-3 py-1.5 rounded border transition-colors flex items-center gap-2"
                [class.bg-yellow-900_30]="filters().WARNING" 
                [class.border-yellow-500]="filters().WARNING"
                [class.text-yellow-400]="filters().WARNING"
                [class.border-border-default]="!filters().WARNING"
                [class.text-text-subtle]="!filters().WARNING">
                <div class="w-2 h-2 rounded-full" [class.bg-yellow-500]="filters().WARNING" [class.bg-surface-high]="!filters().WARNING"></div>
                WARNING
            </button>
            <button (click)="toggleFilter('ERROR')" 
                class="px-3 py-1.5 rounded border transition-colors flex items-center gap-2"
                [class.bg-red-900_30]="filters().ERROR" 
                [class.border-red-500]="filters().ERROR"
                [class.text-red-400]="filters().ERROR"
                [class.border-border-default]="!filters().ERROR"
                [class.text-text-subtle]="!filters().ERROR">
                <div class="w-2 h-2 rounded-full" [class.bg-red-500]="filters().ERROR" [class.bg-surface-high]="!filters().ERROR"></div>
                ERROR
            </button>
            <button (click)="toggleFilter('DEBUG')" 
                class="px-3 py-1.5 rounded border transition-colors flex items-center gap-2"
                [class.bg-brand/30_30]="filters().DEBUG" 
                [class.border-brand]="filters().DEBUG"
                [class.text-brand-light]="filters().DEBUG"
                [class.border-border-default]="!filters().DEBUG"
                [class.text-text-subtle]="!filters().DEBUG">
                <div class="w-2 h-2 rounded-full" [class.bg-brand]="filters().DEBUG" [class.bg-surface-high]="!filters().DEBUG"></div>
                DEBUG
            </button>
            <button (click)="toggleFilter('CRITICAL')" 
                class="px-3 py-1.5 rounded border transition-colors flex items-center gap-2"
                [class.bg-red-900]="filters().CRITICAL" 
                [class.border-red-600]="filters().CRITICAL"
                [class.text-red-100]="filters().CRITICAL"
                [class.font-bold]="filters().CRITICAL"
                [class.animate-pulse]="filters().CRITICAL"
                [class.border-border-default]="!filters().CRITICAL"
                [class.text-text-subtle]="!filters().CRITICAL">
                <div class="w-2 h-2 rounded-full" [class.bg-red-600]="filters().CRITICAL" [class.bg-surface-high]="!filters().CRITICAL"></div>
                CRITICAL
            </button>
        </div>

        <!-- Log Container -->
        <div class="bg-overlay text-brand font-mono p-4 rounded-theme-xl h-96 overflow-y-auto shadow-2xl border border-surface-mid scroll-smooth backdrop-blur-md" 
            data-testid="log-viewer-container"
            #terminalContainer>
        <div class="flex flex-col gap-1">
            @for (item of filteredLogs(); track $index) {
            <div class="flex gap-3 items-start group" data-testid="log-line">
                <span class="text-text-disabled text-[10px] select-none mt-1 min-w-[20px]">{{ $index + 1 }}</span>
                <div [class]="getLevelClass(item.level)"
                    class="whitespace-pre-wrap text-xs leading-relaxed font-medium break-all selection:bg-brand/30">
                {{ item.formatted }}
                </div>
            </div>
            }
            @if (filteredLogs().length === 0 && logs().length > 0) {
                 <div class="flex flex-col items-center justify-center h-full text-text-disabled gap-3">
                    <div class="italic text-sm tracking-wide">All logs hidden by filters.</div>
                </div>
            }
            @if (logs().length === 0) {
            <div class="flex flex-col items-center justify-center h-full text-text-disabled gap-3" data-testid="log-empty">
                <div class="italic text-sm tracking-wide">Connecting to server logs...</div>
            </div>
            }
        </div>
        </div>
    </div>
  `,
    styles: []
})
export class LiveLogViewerComponent implements OnInit, OnDestroy {
    readonly mode = input<'server' | 'training'>('server');

    logs = signal<string[]>([]);

    // Filters State
    filters = signal<{ INFO: boolean, ERROR: boolean, WARNING: boolean, DEBUG: boolean, CRITICAL: boolean }>({
        INFO: true,
        ERROR: true,
        WARNING: true,
        DEBUG: true,
        CRITICAL: true
    });

    // Processed logs (parsed once for performance)
    parsedLogs = computed(() => {
        return this.logs().map(line => {
            const { level, formatted } = this.parseLine(line);
            return { raw: line, level, formatted };
        });
    });

    // Filtered view
    filteredLogs = computed(() => {
        const f = this.filters();
        return this.parsedLogs().filter(item => {
            if (item.level === 'INFO') return f.INFO;
            if (item.level === 'ERROR') return f.ERROR;
            if (item.level === 'WARNING') return f.WARNING;
            if (item.level === 'DEBUG') return f.DEBUG;
            if (item.level === 'CRITICAL') return f.CRITICAL;
            return true; // Show Unknown by default? or map to INFO
        });
    });

    private http = inject(HttpClient);
    private wsService = inject(WebSocketService);
    private rtc = inject(RuntimeConfigService);
    private wsSub?: Subscription;

    terminalContainer = viewChild<ElementRef>('terminalContainer');

    constructor() {
        effect(() => {
            const logs = this.filteredLogs(); // track filtered logs
            if (logs.length > 0) {
                this.scrollToBottom();
            }
        });
    }

    ngOnInit() {
        // Initial fetch for history
        this.fetchLogs().subscribe(logs => this.logs.set(logs));

        // Subscribe to real-time logs
        this.wsSub = this.wsService.on<any>('server_log').subscribe(payload => {
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
            })
        );
    }

    toggleFilter(type: 'INFO' | 'ERROR' | 'WARNING' | 'DEBUG' | 'CRITICAL') {
        this.filters.update(f => ({ ...f, [type]: !f[type] }));
    }

    getLevelClass(level: LogLevel): string {
        switch (level) {
            case 'INFO': return 'text-blue-400';
            case 'ERROR': return 'text-red-400';
            case 'WARNING': return 'text-yellow-400';
            case 'DEBUG': return 'text-brand-light';
            case 'CRITICAL': return 'text-red-500 font-bold bg-red-900/20 px-1 rounded';
            default: return 'text-text-secondary';
        }
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

    ngOnDestroy() {
        this.wsSub?.unsubscribe();
    }
}
