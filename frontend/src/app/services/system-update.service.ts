import { Injectable, computed, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { RuntimeConfigService } from './runtime-config.service';
import { WebSocketService } from './websocket.service';
import { ToastService } from './toast';

export type UpdateState =
    | 'idle' | 'pulling' | 'building' | 'pending_restart' | 'restarting' | 'error';

export interface UpdateStatus {
    state: UpdateState;
    available: boolean;
    branch: string | null;
    commit: string | null;
    dirty: boolean;
    is_repo: boolean;
    behind: number | null;
    active: number;
    error: string | null;
}

export interface CheckResult {
    behind: number;
    commits: string[];
}

@Injectable({ providedIn: 'root' })
export class SystemUpdateService {
    private http = inject(HttpClient);
    private apiUrl = inject(RuntimeConfigService).apiUrl;
    private ws = inject(WebSocketService);
    private toast = inject(ToastService);

    readonly status = signal<UpdateStatus | null>(null);
    readonly available = computed(() => this.status()?.available ?? false);
    readonly state = computed<UpdateState>(() => this.status()?.state ?? 'idle');
    readonly isBusy = computed(() => {
        const s = this.state();
        return s !== 'idle' && s !== 'error';
    });
    readonly updateReady = computed(() => this.available() && (this.status()?.behind ?? 0) > 0);

    private lastBehind = 0;

    constructor() {
        this.ws.on<UpdateStatus>('update.status').subscribe(s => this.ingest(s));
        this.refreshStatus();
    }

    private ingest(s: UpdateStatus): void {
        const behind = s.behind ?? 0;
        if (this.lastBehind === 0 && behind > 0 && s.available) {
            this.toast.info(`Update available — ${behind} new commit(s). Open the update menu to apply.`);
        }
        this.lastBehind = behind;
        this.status.set(s);
    }

    refreshStatus(): void {
        this.http.get<UpdateStatus>(`${this.apiUrl}/system/update/status`)
            .subscribe({ next: s => this.ingest(s), error: () => { /* keep last */ } });
    }

    check(): Observable<CheckResult> {
        return this.http.post<CheckResult>(`${this.apiUrl}/system/update/check`, {});
    }

    apply(): Observable<{ message: string }> {
        return this.http.post<{ message: string }>(`${this.apiUrl}/system/update/apply`, {});
    }
}
