import { Component, input, signal } from '@angular/core';
import { VRAMReport } from '../../../services/system.service';

@Component({
  selector: 'app-vram-budget-card',
  standalone: true,
  template: `
    @if (report()) {
      @let r = report()!;
      <div class="space-y-6 mb-8" data-testid="vram-budget-card">
        <div class="flex items-center justify-between border-b border-surface-mid/30 pb-2 mb-4 cursor-pointer select-none"
              (click)="toggleCollapsed()">
          <div class="flex items-center gap-4">
            <div class="w-1 h-6 rounded-full" [class.bg-emerald-500]="r.fits" [class.bg-red-500]="!r.fits"></div>
            <h3 class="text-sm font-black text-text-subtle uppercase tracking-[0.2em]">VRAM Budget</h3>
            @if (r.fits) {
              <span class="text-[10px] font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full">FITS</span>
            } @else {
              <span class="text-[10px] font-bold text-red-400 bg-red-500/10 border border-red-500/20 px-2 py-0.5 rounded-full">INSUFFICIENT VRAM</span>
            }
          </div>
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
               class="text-text-disabled transition-transform duration-200" [class.rotate-180]="!isCollapsed()">
            <path d="m6 9 6 6 6-6"/>
          </svg>
        </div>

        @if (!isCollapsed()) {
        <div class="animate-in fade-in slide-in-from-top-2 duration-200">
          <!-- Usage Bar -->
          <div class="mb-4">
            <div class="flex justify-between text-[10px] mb-1.5">
              <span class="text-text-muted font-medium">Peak: <span class="text-white font-bold">{{ toGB(r.peak_mb) }} GB</span></span>
              <span class="text-text-subtle">Available: {{ toGB(r.available_mb) }} GB</span>
            </div>
            <div class="h-2.5 bg-surface-mid rounded-full overflow-hidden">
              <div class="h-full rounded-full transition-all duration-500 ease-out"
                   [style.width.%]="r.available_mb > 0 ? clamp((r.peak_mb / r.available_mb) * 100, 0, 100) : 0"
                   [class.bg-gradient-to-r]="true"
                   [class.from-emerald-500]="r.fits"
                   [class.to-teal-400]="r.fits"
                   [class.from-red-500]="!r.fits"
                   [class.to-orange-500]="!r.fits"></div>
            </div>
          </div>

          <!-- Breakdown Grid -->
          <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div class="bg-surface-mid/30 border border-surface-mid/50 rounded-theme-lg px-3 py-2">
              <div class="text-[9px] text-text-subtle uppercase tracking-wider mb-0.5">Model</div>
              <div class="text-sm font-bold text-white">{{ toGB(r.model_weights_mb) }} <span class="text-[10px] text-text-subtle font-normal">GB</span></div>
            </div>
            <div class="bg-surface-mid/30 border border-surface-mid/50 rounded-theme-lg px-3 py-2">
              <div class="text-[9px] text-text-subtle uppercase tracking-wider mb-0.5">LoRA</div>
              <div class="text-sm font-bold text-white">{{ toGB(r.lora_adapters_mb) }} <span class="text-[10px] text-text-subtle font-normal">GB</span></div>
            </div>
            <div class="bg-surface-mid/30 border border-surface-mid/50 rounded-theme-lg px-3 py-2">
              <div class="text-[9px] text-text-subtle uppercase tracking-wider mb-0.5">Optimizer</div>
              <div class="text-sm font-bold text-white">{{ toGB(r.optimizer_states_mb) }} <span class="text-[10px] text-text-subtle font-normal">GB</span></div>
            </div>
            <div class="bg-surface-mid/30 border border-surface-mid/50 rounded-theme-lg px-3 py-2">
              <div class="text-[9px] text-text-subtle uppercase tracking-wider mb-0.5">Activations</div>
              <div class="text-sm font-bold text-white">{{ toGB(r.activations_mb) }} <span class="text-[10px] text-text-subtle font-normal">GB</span></div>
            </div>
          </div>

          <!-- Warnings -->
          @if (r.warnings.length > 0) {
            <div class="mt-3 space-y-1">
              @for (w of r.warnings; track w) {
                <div class="flex items-start gap-2 text-[11px] text-amber-400">
                  <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="mt-0.5 shrink-0">
                    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
                    <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                  </svg>
                  <span>{{ w }}</span>
                </div>
              }
            </div>
          }
        </div>
        }
      </div>
    }
  `
})
export class VramBudgetCardComponent {
  report = input<VRAMReport | null>(null);

  // Isolate collapse state from parent
  isCollapsed = signal<boolean>(false);

  toggleCollapsed() {
    this.isCollapsed.set(!this.isCollapsed());
  }

  /** Convert MB value to GB string with 1 decimal. */
  toGB(mb: number): string {
    return (mb / 1024).toFixed(1);
  }

  clamp(val: number, min: number, max: number): number {
    return Math.min(Math.max(val, min), max);
  }
}
