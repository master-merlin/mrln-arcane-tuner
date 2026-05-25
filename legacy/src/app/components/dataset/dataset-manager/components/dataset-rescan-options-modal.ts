import { Component, output, input } from '@angular/core';

@Component({
  selector: 'app-dataset-rescan-options-modal',
  standalone: true,
  template: `
    <div class="fixed inset-0 z-[120] flex items-center justify-center p-6 backdrop-blur-md bg-base/80 bubble-in">
      <div class="bg-surface-low border border-surface-high w-full max-w-lg rounded-theme-2xl shadow-2xl overflow-hidden border-shine">
        <!-- Header -->
        <div class="p-6 border-b border-surface-high flex items-center justify-between bg-surface-mid/50">
          <div>
            <h2 class="text-xl font-bold text-white">Rescan Options</h2>
            <p class="text-xs text-text-subtle font-medium tracking-wide uppercase mt-1">Select scanning mode for {{ datasetName() }}</p>
          </div>
          <button (click)="close.emit()" class="text-text-muted hover:text-white transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
          </button>
        </div>
        
        <div class="p-6 space-y-4">
          <!-- Incremental Scan -->
          <button (click)="confirm.emit(false)" 
                class="w-full text-left p-5 rounded-theme-xl border border-surface-high hover:border-brand hover:bg-surface-mid hover:shadow-[0_0_15px_rgba(255,51,102,0.15)] transition-all group flex gap-4 items-start">
            <div class="p-3 rounded-theme-xl bg-surface-high group-hover:bg-brand/10 text-text-muted group-hover:text-brand transition-colors shrink-0">
               <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
            </div>
            <div>
              <h3 class="font-bold text-white mb-1">Incremental Scan</h3>
              <p class="text-sm text-text-muted">Fast. Only checks for new, deleted, or missing images. Keeps existing caches, manual captions, and hashes.</p>
            </div>
          </button>

          <!-- Full Scan -->
          <button (click)="confirm.emit(true)" 
                class="w-full text-left p-5 rounded-theme-xl border border-surface-high hover:border-warning hover:bg-surface-mid hover:shadow-[0_0_15px_rgba(255,170,0,0.15)] transition-all group flex gap-4 items-start">
            <div class="p-3 rounded-theme-xl bg-surface-high group-hover:bg-warning/10 text-text-muted group-hover:text-warning transition-colors shrink-0">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12a10 10 0 1 0 10-10"></path><polyline points="12 2 12 12 18 18"></polyline></svg>
            </div>
            <div>
              <h3 class="font-bold text-white mb-1">Full Scan</h3>
              <p class="text-sm text-text-muted">Slower. Completely recalculates all structural similarity hashes. Good for applying algorithm upgrades or fixing cache sync issues.</p>
            </div>
          </button>
        </div>

      </div>
    </div>
  `
})
export class DatasetRescanOptionsModalComponent {
  datasetName = input.required<string>();
  close = output<void>();
  confirm = output<boolean>(); // true if forceFull
}
