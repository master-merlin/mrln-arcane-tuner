import { Injectable, effect, inject, signal, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, Subscription } from 'rxjs';
import { RuntimeConfigService } from './runtime-config.service';
import { WebSocketService } from './websocket.service';

// ── Types ──────────────────────────────────────────────────────────────

export interface GpuProcess {
    pid: number;
    name: string;
    used_mb: number;
}

export interface GPUStatus {
    index: number;
    name: string;
    vram_used_mb: number;
    vram_total_mb: number;
    /** total − used (device-wide free VRAM). */
    vram_free_mb?: number;
    vram_percent: number;
    temperature_c: number;
    power_draw_w: number;
    power_limit_w: number;
    gpu_utilization: number;
    memory_utilization: number;
    clock_graphics_mhz: number;
    clock_memory_mhz: number;
    /** Top VRAM-holding processes (ComfyUI, browser, this app, …). */
    processes?: GpuProcess[];
}

export interface SystemStatus {
    ram_used_mb: number;
    ram_total_mb: number;
    ram_percent: number;
    cpu_percent: number;
}

export interface SystemSnapshot {
    gpus: GPUStatus[];
    system: SystemStatus | null;
}

export interface HealthSnapshot {
    status: string;
    uptime_seconds: number;
    model_count: number;
    active_jobs: number;
}

export interface VRAMReport {
    model_weights_mb: number;
    lora_adapters_mb: number;
    optimizer_states_mb: number;
    gradients_mb: number;
    activations_mb: number;
    overhead_mb: number;
    caching_peak_mb: number;
    training_peak_mb: number;
    peak_mb: number;
    /** FREE VRAM (total − used by all processes) — drives the fit check. */
    available_mb: number;
    /** Total card capacity. */
    total_mb?: number;
    /** VRAM already held by other processes at estimate time. */
    used_mb?: number;
    fits: boolean;
    warnings: string[];
    /** True when ≥1 per-component multiplier was learned from local runs. */
    calibrated?: boolean;
    /** Which analytic components were calibrated from measured data. */
    calibrated_components?: string[];
}

// ── Service ────────────────────────────────────────────────────────────

@Injectable({
    providedIn: 'root'
})
export class SystemService implements OnDestroy {
    private http = inject(HttpClient);
    private ws = inject(WebSocketService);
    private apiUrl = inject(RuntimeConfigService).apiUrl;

    // Live metrics signal — updated by WebSocket stream
    metrics = signal<SystemSnapshot | null>(null);
    private metricsSub: Subscription | null = null;
    private reconnectSub: Subscription | null = null;

    // Track whether metrics subscription is active + last interval
    private metricsActive = false;
    private metricsInterval = 2.0;

    // Reference count of live subscribers (sidebar mini-stats + the Jobs
    // system-monitor). The WS stream stays up while ANY consumer needs it, so
    // leaving the Jobs screen no longer blanks the always-on sidebar stats.
    private subscriberCount = 0;

    constructor() {
        // Re-send the metrics subscription whenever the socket becomes
        // connected — including the INITIAL open. A subscriber that registers
        // during bootstrap (the sidebar) can otherwise lose its first
        // subscribe to a not-yet-open socket; `reconnected$` only covers
        // later reconnects, not the first connect.
        effect(() => {
            if (this.ws.isConnected() && this.metricsActive) {
                this.ws.send({ action: 'subscribe_metrics', interval_s: this.metricsInterval });
            }
        });
    }

    /** Backend health snapshot for the Server screen KPI rail. */
    getHealth(): Observable<HealthSnapshot> {
        return this.http.get<HealthSnapshot>(`${this.apiUrl}/system/health`);
    }

    /** App version shown in the sidebar footer. */
    getVersion(): Observable<{ version: string }> {
        return this.http.get<{ version: string }>(`${this.apiUrl}/system/version`);
    }

    /** Persisted server log tail (structlog lines), newest-last. */
    getLogs(lines: number = 200): Observable<string[]> {
        return this.http.get<string[]>(`${this.apiUrl}/system/logs?lines=${lines}`);
    }

    /** Clear the server-side log buffer. */
    clearLogs(): Observable<{ message?: string; error?: string }> {
        return this.http.post<{ message?: string; error?: string }>(`${this.apiUrl}/system/logs/clear`, {});
    }

    /** Start receiving live system metrics via WebSocket. Reference-counted. */
    subscribeMetrics(intervalS: number = 2.0): void {
        this.subscriberCount++;
        this.metricsActive = true;
        this.metricsInterval = intervalS;

        // Start WebSocket stream
        this.ws.send({ action: 'subscribe_metrics', interval_s: intervalS });

        // Subscribe to incoming events
        if (!this.metricsSub) {
            this.metricsSub = this.ws.on<SystemSnapshot>('system_metrics').subscribe(snap => {
                this.metrics.set(snap);
            });
        }

        // Auto-re-subscribe on reconnect (server restart loses the subscription)
        if (!this.reconnectSub) {
            this.reconnectSub = this.ws.reconnected$.subscribe(() => {
                if (this.metricsActive) {
                    this.ws.send({ action: 'subscribe_metrics', interval_s: this.metricsInterval });
                }
            });
        }
    }

    /** Release one subscriber. The stream only tears down when the last one leaves. */
    unsubscribeMetrics(): void {
        if (this.subscriberCount > 0) this.subscriberCount--;
        if (this.subscriberCount > 0) return; // other consumers still need it

        this.metricsActive = false;
        this.ws.send({ action: 'unsubscribe_metrics' });
        this.metricsSub?.unsubscribe();
        this.metricsSub = null;
        this.reconnectSub?.unsubscribe();
        this.reconnectSub = null;
        this.metrics.set(null);
    }

    ngOnDestroy(): void {
        this.unsubscribeMetrics();
    }
}
