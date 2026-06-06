import { Component, ChangeDetectionStrategy, computed, input, signal } from '@angular/core';
import { VRAMReport } from '../../../services/system.service';
import { vramBreakdownParts } from '../vram-breakdown';

@Component({
  selector: 'app-vram-budget-card',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section class="card" data-testid="vram-budget-card">
      <div class="card-head cursor-pointer select-none" (click)="toggleCollapsed()">
        <div class="card-title min-w-0" style="padding:0">
          <span class="w-[3px] h-3.5 rounded-sm shrink-0"
                [class.bg-emerald-500]="report()?.fits"
                [class.bg-red-500]="report() && !report()!.fits"
                [class.bg-brand]="!report()"></span>
          <span class="shrink-0">VRAM Budget</span>
          @if (report(); as r) {
            <span class="ml-2.5 normal-case tracking-normal text-[12px] font-medium text-text-secondary truncate">{{ toGB(r.peak_mb) }} / {{ toGB(r.available_mb) }} GB</span>
          }
        </div>
        <div class="flex items-center gap-2 shrink-0">
          @if (report(); as r) {
            @if (r.fits) {
              <span class="chip success">FITS</span>
            } @else {
              <span class="chip danger">INSUFFICIENT</span>
            }
          }
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
               class="text-text-muted transition-transform duration-200" [class.rotate-180]="!isCollapsed()">
            <path d="m6 9 6 6 6-6"/>
          </svg>
        </div>
      </div>

      <div class="card-body" [class.hidden]="isCollapsed()">
        @if (report(); as r) {
          <!-- Peak / Available + segmented breakdown bar (colors shared with the Live Estimate rail) -->
          <div class="mb-3">
            <div class="flex justify-between text-[10px] mb-1.5">
              <span class="text-text-muted font-medium">Peak: <span class="text-text-primary font-bold font-mono">{{ toGB(r.peak_mb) }} GB</span></span>
              <span class="text-text-subtle">Available: <span class="font-mono">{{ toGB(r.available_mb) }} GB</span></span>
            </div>
            <div class="flex items-stretch h-2.5 rounded-full overflow-hidden border border-surface-mid bg-surface-mid">
              @for (p of barSegments(); track p.key) {
                <div [style.flex-grow]="p.mb" [style.flex-basis.px]="0" [style.background]="p.color"
                     [title]="p.label + ' ' + toGB(p.mb) + ' GB'"></div>
              }
            </div>
          </div>

          <!-- Colored legend (stacked size values — matches the rail) -->
          <div class="grid grid-cols-2 md:grid-cols-4 gap-x-4 gap-y-1.5">
            @for (p of legendParts(); track p.key) {
              <div class="flex items-center justify-between gap-2 text-[11px] min-w-0">
                <span class="flex items-center gap-1.5 text-text-muted min-w-0">
                  <span class="w-2 h-2 rounded-sm shrink-0 border border-border-subtle" [style.background]="p.color"></span>
                  <span class="truncate">{{ p.label }}</span>
                </span>
                <span class="font-mono text-text-secondary shrink-0">{{ toGB(p.mb) }} GB</span>
              </div>
            }
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
        } @else {
          <p class="text-[11px] text-text-muted">VRAM estimate pending — configure the model and datasets to see usage.</p>
        }

        <!-- Advanced block-swap settings live in the same card (projected) -->
        <div class="div"></div>
        <ng-content></ng-content>
      </div>
    </section>
  `
})
export class VramBudgetCardComponent {
  report = input<VRAMReport | null>(null);

  /**
   * Bar segments (positive-weight parts) — natural breakdown order with
   * Headroom trailing, shared color map with the rail. The bar MUST stay in
   * this order so Headroom renders as the trailing transparent segment.
   */
  protected readonly barSegments = computed(() =>
    vramBreakdownParts(this.report()).filter(p => p.mb > 0));

  /**
   * Legend rows: same parts/colors as the bar, but with Gradients and Headroom
   * positions swapped per design (legend-only — the bar keeps natural order).
   * Skips 0-MB parts, but always keeps Model + Headroom.
   */
  protected readonly legendParts = computed(() => {
    const parts = vramBreakdownParts(this.report());
    const gi = parts.findIndex(p => p.key === 'gradients');
    const hi = parts.findIndex(p => p.key === 'headroom');
    if (gi !== -1 && hi !== -1) {
      [parts[gi], parts[hi]] = [parts[hi], parts[gi]];
    }
    return parts.filter(p => p.always || p.mb > 0);
  });

  // Isolate collapse state from parent
  isCollapsed = signal<boolean>(false);

  toggleCollapsed() {
    this.isCollapsed.set(!this.isCollapsed());
  }

  /** Convert MB value to GB string with 1 decimal. */
  toGB(mb: number): string {
    return (mb / 1024).toFixed(1);
  }
}
