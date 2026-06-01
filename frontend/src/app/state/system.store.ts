import { Injectable, OnDestroy, computed, inject } from '@angular/core';
import { SystemService } from '../services/system.service';

/**
 * Compact view of system metrics used by the sidebar mini-stats card.
 * Maps onto the first GPU in the snapshot — multi-GPU rigs use the same
 * fields with summed/averaged values (TODO when we have a multi-GPU box
 * to test against).
 */
export interface SidebarStats {
    gpuPct: number;
    vramUsedGB: number;
    vramTotalGB: number;
    powerW: number;
    /** GPU temperature in °C (not the CPU/system temp). */
    tempC: number;
    cpuPct: number;
    ramUsedGB: number;
    ramTotalGB: number;
}

const ZERO: SidebarStats = {
    gpuPct: 0,
    vramUsedGB: 0,
    vramTotalGB: 0,
    powerW: 0,
    tempC: 0,
    cpuPct: 0,
    ramUsedGB: 0,
    ramTotalGB: 0,
};

/**
 * Thin adapter over `SystemService.metrics` that projects the raw
 * snapshot into the 5 numbers the sidebar card needs.
 *
 * Subscribes to the WS metrics stream on construction so the sidebar
 * gets values without each consumer having to remember the boilerplate.
 * The service de-dupes subscriptions internally so repeat injection of
 * this store is safe.
 */
@Injectable({ providedIn: 'root' })
export class SystemStore implements OnDestroy {
    private system = inject(SystemService);

    readonly sidebar = computed<SidebarStats>(() => {
        const snap = this.system.metrics();
        if (!snap || !snap.gpus.length) return ZERO;
        const g = snap.gpus[0];
        const sys = snap.system;
        return {
            gpuPct: g.gpu_utilization,
            vramUsedGB: g.vram_used_mb / 1024,
            vramTotalGB: g.vram_total_mb / 1024,
            powerW: g.power_draw_w,
            tempC: g.temperature_c,
            cpuPct: sys?.cpu_percent ?? 0,
            ramUsedGB: (sys?.ram_used_mb ?? 0) / 1024,
            ramTotalGB: (sys?.ram_total_mb ?? 0) / 1024,
        };
    });

    constructor() {
        // Kick off the WS stream the first time something injects us.
        // `subscribeMetrics` is idempotent server-side.
        this.system.subscribeMetrics();
    }

    ngOnDestroy(): void {
        this.system.unsubscribeMetrics();
    }
}
