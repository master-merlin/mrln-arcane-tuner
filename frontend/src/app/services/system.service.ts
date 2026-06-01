import { Injectable, inject, signal, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, Subscription } from 'rxjs';
import { RuntimeConfigService } from './runtime-config.service';
import { WebSocketService } from './websocket.service';

// ── Types ──────────────────────────────────────────────────────────────

export interface GPUStatus {
    index: number;
    name: string;
    vram_used_mb: number;
    vram_total_mb: number;
    vram_percent: number;
    temperature_c: number;
    power_draw_w: number;
    power_limit_w: number;
    gpu_utilization: number;
    memory_utilization: number;
    clock_graphics_mhz: number;
    clock_memory_mhz: number;
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
    available_mb: number;
    fits: boolean;
    warnings: string[];
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

    /** One-shot system snapshot (REST). */
    getSystemStatus(): Observable<SystemSnapshot> {
        return this.http.get<SystemSnapshot>(`${this.apiUrl}/system/status`);
    }

    /** One-shot GPU-only snapshot (REST). */
    getGPUStatus(): Observable<{ gpus: GPUStatus[] }> {
        return this.http.get<{ gpus: GPUStatus[] }>(`${this.apiUrl}/system/gpu`);
    }

    /** Backend health snapshot for the Server screen KPI rail. */
    getHealth(): Observable<HealthSnapshot> {
        return this.http.get<HealthSnapshot>(`${this.apiUrl}/system/health`);
    }

    /** Estimate VRAM for a training configuration. */
    estimateVRAM(definitionId: string, config: Record<string, any>): Observable<VRAMReport> {
        return this.http.post<VRAMReport>(`${this.apiUrl}/jobs/estimate-vram`, {
            definition_id: definitionId,
            config,
        });
    }

    /** Start receiving live system metrics via WebSocket. */
    subscribeMetrics(intervalS: number = 2.0): void {
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
                    console.log('[SystemService] Re-subscribing metrics after reconnect');
                    this.ws.send({ action: 'subscribe_metrics', interval_s: this.metricsInterval });
                }
            });
        }
    }

    /** Stop receiving live system metrics. */
    unsubscribeMetrics(): void {
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
