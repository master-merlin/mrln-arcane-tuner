import { Component, inject, OnInit, OnDestroy } from '@angular/core';
import { SystemService } from '../../../services/system.service';
import { DecimalPipe } from '@angular/common';

@Component({
    selector: 'app-system-monitor',
    standalone: true,
    host: { class: 'block' },
    imports: [DecimalPipe],
    template: `
    <div class="bg-surface-low/30 border border-border-default rounded-theme-xl p-6 hover:border-border-default transition-all">
        <h3 class="text-lg font-medium text-white mb-5 flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
                fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
                stroke-linejoin="round" class="text-emerald-400">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
            System Monitor
        </h3>

        @if (svc.metrics(); as snap) {
            <div class="space-y-5">

                <!-- System (CPU + RAM) -->
                <div class="grid grid-cols-2 gap-4">

                    <!-- CPU -->
                    <div class="space-y-2">
                        <div class="flex items-center justify-between">
                            <span class="text-[11px] uppercase tracking-wider text-text-subtle font-semibold">CPU</span>
                            <span class="text-xs font-mono"
                                  [class]="snap.system?.cpu_percent! > 90 ? 'text-red-400' :
                                           snap.system?.cpu_percent! > 70 ? 'text-yellow-400' : 'text-emerald-400'">
                                {{ snap.system?.cpu_percent | number:'1.0-0' }}%
                            </span>
                        </div>
                        <div class="h-1.5 bg-surface-mid rounded-full overflow-hidden">
                            <div class="h-full rounded-full transition-all duration-700 ease-out"
                                 [style.width.%]="snap.system?.cpu_percent || 0"
                                 [class]="snap.system?.cpu_percent! > 90 ? 'bg-red-500' :
                                          snap.system?.cpu_percent! > 70 ? 'bg-yellow-500' : 'bg-blue-500'">
                            </div>
                        </div>
                    </div>

                    <!-- RAM -->
                    <div class="space-y-2">
                        <div class="flex items-center justify-between">
                            <span class="text-[11px] uppercase tracking-wider text-text-subtle font-semibold">RAM</span>
                            <span class="text-xs font-mono text-text-muted">
                                {{ formatMB(snap.system?.ram_used_mb || 0) }} / {{ formatMB(snap.system?.ram_total_mb || 0) }}
                            </span>
                        </div>
                        <div class="h-1.5 bg-surface-mid rounded-full overflow-hidden">
                            <div class="h-full rounded-full transition-all duration-700 ease-out"
                                 [style.width.%]="snap.system?.ram_percent || 0"
                                 [class]="snap.system?.ram_percent! > 90 ? 'bg-red-500' :
                                          snap.system?.ram_percent! > 70 ? 'bg-yellow-500' : 'bg-brand'">
                            </div>
                        </div>
                    </div>
                </div>

                <!-- GPU Cards -->
                @for (gpu of snap.gpus; track gpu.index) {
                    <div class="border-t border-border-default pt-4 space-y-3">

                        <!-- GPU Name + Temp -->
                        <div class="flex items-center justify-between">
                            <div class="flex items-center gap-2">
                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24"
                                    fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
                                    stroke-linejoin="round" class="text-emerald-400">
                                    <rect x="2" y="6" width="20" height="12" rx="2"/>
                                    <path d="M22 10h-2"/><path d="M22 14h-2"/>
                                    <path d="M6 10h4"/><path d="M6 14h4"/>
                                </svg>
                                <span class="text-sm font-medium text-white">{{ gpu.name }}</span>
                                @if (snap.gpus.length > 1) {
                                    <span class="text-[10px] px-1.5 py-0.5 rounded bg-surface-mid text-text-subtle font-mono">{{ gpu.index }}</span>
                                }
                            </div>
                            <span class="text-xs font-mono px-2 py-0.5 rounded"
                                  [class]="gpu.temperature_c > 85 ? 'text-red-400 bg-red-500/10' :
                                           gpu.temperature_c > 70 ? 'text-yellow-400 bg-yellow-500/10' : 'text-text-muted bg-surface-mid'">
                                {{ gpu.temperature_c }}°C
                            </span>
                        </div>

                        <!-- GPU Metrics — 4-column consistent grid -->
                        <div class="grid grid-cols-4 gap-4">

                            <!-- Load -->
                            <div class="space-y-2">
                                <div class="flex items-center justify-between">
                                    <span class="text-[11px] uppercase tracking-wider text-text-subtle font-semibold">Load</span>
                                    <span class="text-xs font-mono"
                                          [class]="gpu.gpu_utilization > 90 ? 'text-red-400' :
                                                   gpu.gpu_utilization > 50 ? 'text-yellow-400' : 'text-emerald-400'">
                                        {{ gpu.gpu_utilization }}%
                                    </span>
                                </div>
                                <div class="h-1.5 bg-surface-mid rounded-full overflow-hidden">
                                    <div class="h-full rounded-full transition-all duration-700 ease-out"
                                         [style.width.%]="gpu.gpu_utilization"
                                         [class]="gpu.gpu_utilization > 90 ? 'bg-red-500' :
                                                  gpu.gpu_utilization > 50 ? 'bg-yellow-500' : 'bg-emerald-500'">
                                    </div>
                                </div>
                            </div>

                            <!-- VRAM -->
                            <div class="space-y-2">
                                <div class="flex items-center justify-between">
                                    <span class="text-[11px] uppercase tracking-wider text-text-subtle font-semibold">VRAM</span>
                                    <span class="text-xs font-mono text-text-muted">
                                        {{ formatMB(gpu.vram_used_mb) }}/{{ formatMB(gpu.vram_total_mb) }}
                                    </span>
                                </div>
                                <div class="h-1.5 bg-surface-mid rounded-full overflow-hidden">
                                    <div class="h-full rounded-full transition-all duration-700 ease-out"
                                         [style.width.%]="gpu.vram_percent"
                                         [class]="gpu.vram_percent > 90 ? 'bg-red-500' :
                                                  gpu.vram_percent > 70 ? 'bg-yellow-500' : 'bg-emerald-500'">
                                    </div>
                                </div>
                            </div>

                            <!-- Power -->
                            <div class="space-y-2">
                                <div class="flex items-center justify-between">
                                    <span class="text-[11px] uppercase tracking-wider text-text-subtle font-semibold">Power</span>
                                    <span class="text-xs font-mono text-text-muted">
                                        {{ gpu.power_draw_w | number:'1.0-0' }}W / {{ gpu.power_limit_w | number:'1.0-0' }}W
                                    </span>
                                </div>
                                <div class="h-1.5 bg-surface-mid rounded-full overflow-hidden">
                                    <div class="h-full rounded-full transition-all duration-700 ease-out bg-amber-500"
                                         [style.width.%]="gpu.power_limit_w > 0 ? (gpu.power_draw_w / gpu.power_limit_w * 100) : 0">
                                    </div>
                                </div>
                            </div>

                            <!-- Clock -->
                            <div class="space-y-2">
                                <div class="flex items-center justify-between">
                                    <span class="text-[11px] uppercase tracking-wider text-text-subtle font-semibold">Clock</span>
                                    <span class="text-xs font-mono text-text-muted">{{ gpu.clock_graphics_mhz }} MHz</span>
                                </div>
                                <div class="flex items-center justify-between mt-1">
                                    <span class="text-[11px] uppercase tracking-wider text-text-disabled font-semibold">Mem</span>
                                    <span class="text-xs font-mono text-text-subtle">{{ gpu.clock_memory_mhz }} MHz</span>
                                </div>
                            </div>

                        </div>
                    </div>
                }

            </div>
        } @else {
            <!-- Loading -->
            <div class="py-6 text-center">
                <div class="animate-pulse text-text-subtle text-sm">Connecting to system monitor…</div>
            </div>
        }
    </div>
    `
})
export class SystemMonitorComponent implements OnInit, OnDestroy {
    svc = inject(SystemService);

    ngOnInit() {
        this.svc.subscribeMetrics(2.0);
    }

    ngOnDestroy() {
        this.svc.unsubscribeMetrics();
    }

    formatMB(mb: number): string {
        if (mb >= 1024) {
            return (mb / 1024).toFixed(1) + ' GB';
        }
        return mb + ' MB';
    }
}
