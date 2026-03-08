import { Component, input, output, signal, inject } from '@angular/core';
import { ModelService, BlockTopologyGroup, ModelCapabilities } from '../../../services/model.service';
import { ToastService } from '../../../services/toast';

@Component({
  selector: 'app-advanced-vram-card',
  standalone: true,
  template: `
    <div class="space-y-6 mb-8" data-testid="advanced-vram-card">
      <div class="flex items-center justify-between border-b border-surface-mid/30 pb-2 mb-4 cursor-pointer select-none"
           (click)="toggleCollapsed()">
        <div class="flex items-center gap-4">
          <div class="w-1 h-6 rounded-full bg-violet-500"></div>
          <h3 class="text-sm font-black text-text-subtle uppercase tracking-[0.2em]">Advanced VRAM</h3>
          @if (!enriched()) {
            <span class="text-[10px] font-bold text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded-full">NOT ENRICHED</span>
          } @else if (hasActiveSwaps()) {
            <span class="text-[10px] font-bold text-violet-400 bg-violet-500/10 border border-violet-500/20 px-2 py-0.5 rounded-full">BLOCK SWAP ACTIVE</span>
          }
        </div>
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
             class="text-text-disabled transition-transform duration-200" [class.rotate-180]="!isCollapsed()">
          <path d="m6 9 6 6 6-6"/>
        </svg>
      </div>

      @if (!isCollapsed()) {
        <div class="animate-in fade-in slide-in-from-top-2 duration-200">
          @if (!definitionId()) {
            <div class="text-[11px] text-text-disabled italic px-1">Select a model to configure block swapping.</div>
          } @else if (loading()) {
            <div class="flex items-center gap-2 text-[11px] text-text-muted px-1">
              <div class="w-3 h-3 border-2 border-violet-400 border-t-transparent rounded-full animate-spin"></div>
              Loading capabilities...
            </div>
          } @else if (!enriched()) {
            <!-- Not enriched — offer enrich button -->
            <div class="bg-surface-mid/30 border border-surface-mid/50 rounded-theme-lg p-4 text-center">
              <p class="text-[11px] text-text-muted mb-3">
                Block topology data not available for this model.
                Enrich the definition to enable block swapping.
              </p>
              <button type="button" (click)="enrichModel()"
                      [disabled]="enriching()"
                      class="px-4 py-1.5 text-[11px] font-bold uppercase tracking-wider
                             bg-violet-600 hover:bg-violet-500 text-white rounded-theme
                             disabled:opacity-50 disabled:cursor-wait
                             transition-colors duration-200">
                @if (enriching()) {
                  <span class="flex items-center gap-2">
                    <span class="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                    Enriching...
                  </span>
                } @else {
                  Enrich Model
                }
              </button>
            </div>
          } @else {
            <!-- Block swap sliders -->
            <div class="space-y-4">
              @for (group of topology(); track group.name) {
                <div class="bg-surface-mid/20 border border-surface-mid/40 rounded-theme-lg px-4 py-3">
                  <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center gap-2">
                      <span class="text-[11px] font-bold text-text-primary">{{ formatName(group.name) }}</span>
                      <span class="text-[10px] text-text-disabled">({{ group.count }} blocks)</span>
                    </div>
                    <span class="text-[11px] font-mono text-violet-400 min-w-[3ch] text-right">{{ getSwapValue(group.name) }}%</span>
                  </div>
                  <input type="range" min="0" max="100" step="5"
                         [value]="getSwapValue(group.name)"
                         (input)="onSliderChange(group.name, $event)"
                         class="w-full h-1.5 bg-surface-mid rounded-full appearance-none cursor-pointer
                                [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5
                                [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:rounded-full
                                [&::-webkit-slider-thumb]:bg-violet-400 [&::-webkit-slider-thumb]:border-2
                                [&::-webkit-slider-thumb]:border-violet-600 [&::-webkit-slider-thumb]:shadow-lg
                                [&::-webkit-slider-thumb]:cursor-pointer
                                [&::-webkit-slider-thumb]:transition-all [&::-webkit-slider-thumb]:duration-150
                                [&::-webkit-slider-thumb]:hover:bg-violet-300" />
                  <div class="flex justify-between text-[9px] text-text-disabled mt-1">
                    <span>GPU only</span>
                    <span>CPU offload</span>
                  </div>
                </div>
              }

              @if (topology().length > 0) {
                <div class="flex items-start gap-2 text-[10px] text-text-disabled px-1">
                  <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="mt-0.5 shrink-0">
                    <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
                  </svg>
                  <span>Higher % = more blocks offloaded to CPU. Reduces VRAM usage but slows training. Use only if you are running out of VRAM.</span>
                </div>
              }
            </div>
          }
        </div>
      }
    </div>
  `
})
export class AdvancedVramCardComponent {
  definitionId = input<string>('');
  blockSwapChanged = output<Record<string, number>>();

  private modelService = inject(ModelService);
  private toast = inject(ToastService);

  isCollapsed = signal<boolean>(true);
  loading = signal<boolean>(false);
  enriching = signal<boolean>(false);
  enriched = signal<boolean>(false);
  topology = signal<BlockTopologyGroup[]>([]);
  swapValues = signal<Record<string, number>>({});

  private lastDefinitionId = '';

  /** Called by parent to restore saved slider values (e.g. from template). */
  setSwapValues(values: Record<string, number>): void {
    this.swapValues.set(values);
  }

  /**
   * Called by the parent when definition_id changes.
   * We don't use ngOnChanges because input signals don't trigger it reliably.
   */
  loadCapabilities(defId: string): void {
    if (!defId || defId === this.lastDefinitionId) return;
    this.lastDefinitionId = defId;
    this.loading.set(true);
    this.enriched.set(false);
    this.topology.set([]);
    this.swapValues.set({});

    this.modelService.getCapabilities(defId).subscribe({
      next: (caps: ModelCapabilities) => {
        this.enriched.set(caps.enriched);
        this.topology.set(caps.block_topology || []);
        // Initialize missing sliders to 0%, preserving any values set programmatically during load
        const currentVals = this.swapValues();
        const initial: Record<string, number> = { ...currentVals };
        for (const group of caps.block_topology || []) {
          if (initial[group.name] === undefined) {
            initial[group.name] = 0;
          }
        }
        this.swapValues.set(initial);
        this.loading.set(false);
      },
      error: () => {
        this.loading.set(false);
        this.enriched.set(false);
      }
    });
  }

  enrichModel(): void {
    const defId = this.definitionId();
    if (!defId) return;
    this.enriching.set(true);

    this.modelService.enrichDefinition(defId).subscribe({
      next: (result) => {
        this.enriching.set(false);
        this.enriched.set(true);
        this.topology.set(result.block_topology || []);
        const currentVals = this.swapValues();
        const initial: Record<string, number> = { ...currentVals };
        for (const group of result.block_topology || []) {
          if (initial[group.name] === undefined) {
            initial[group.name] = 0;
          }
        }
        this.swapValues.set(initial);
        this.toast.success('Model enriched successfully');
      },
      error: (err) => {
        this.enriching.set(false);
        this.toast.error('Enrichment failed: ' + (err?.error?.detail || 'Unknown error'));
      }
    });
  }

  toggleCollapsed(): void {
    this.isCollapsed.set(!this.isCollapsed());
  }

  getSwapValue(name: string): number {
    return this.swapValues()[name] ?? 0;
  }

  onSliderChange(name: string, event: Event): void {
    const value = parseInt((event.target as HTMLInputElement).value, 10);
    const updated = { ...this.swapValues(), [name]: value };
    this.swapValues.set(updated);
    this.blockSwapChanged.emit(updated);
  }

  hasActiveSwaps(): boolean {
    return Object.values(this.swapValues()).some(v => v > 0);
  }

  formatName(name: string): string {
    return name
      .replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase());
  }
}
