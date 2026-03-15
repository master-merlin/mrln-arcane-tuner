import { Component, output, input, signal, effect } from '@angular/core';
import { FormsModule } from '@angular/forms';

@Component({
    selector: 'app-dataset-toolbar',
    standalone: true,
    imports: [FormsModule],
    template: `
    <div class="flex flex-col md:flex-row items-center justify-between gap-4">
       <!-- Search Bar -->
       <div class="flex-1 max-w-md w-full">
          <div class="relative group">
              <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-text-subtle group-focus-within:text-brand transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
              </div>
              <input 
                  type="text" 
                  [ngModel]="searchValue()"
                  (ngModelChange)="onSearchChange($event)"
                  data-testid="search-datasets"
                  placeholder="Search by name, description or category..." 
                  class="block w-full pl-10 pr-3 py-2.5 border border-surface-mid rounded-theme-xl leading-5 bg-surface-low/80 text-text-primary placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-brand/20 focus:border-brand/50 sm:text-sm transition-all duration-200"
              >
              @if (searchValue()) {
                  <button (click)="onSearchChange('')" class="absolute inset-y-0 right-0 pr-3 flex items-center text-text-subtle hover:text-text-secondary">
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                  </button>
              }
          </div>
       </div>

       <div class="flex gap-3 w-full md:w-auto items-stretch">
           <button (click)="createRequested.emit()" 
              data-testid="btn-new-dataset"
              class="flex-1 md:flex-none min-w-[160px] h-10 bg-brand hover:bg-brand/90 text-white px-5 rounded-theme-xl text-sm font-bold transition-all flex items-center justify-center gap-2 shadow-lg shadow-brand/20 active:scale-95">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
              New Dataset
           </button>
           <!-- Rescan Split Button -->
           <div class="relative flex" data-testid="btn-rescan-library">
              <div class="flex rounded-theme-xl border border-surface-high/50 overflow-hidden min-w-[160px] h-10">
                <button (click)="rescanLibraryRequested.emit(false)" 
                  class="flex-1 bg-surface-mid hover:bg-surface-high text-text-primary px-4 py-2.5 text-sm font-bold transition-all flex items-center justify-center gap-2 active:scale-95">
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.3"/></svg>
                  Rescan
                </button>
                <button (click)="rescanLibraryRequested.emit(true)"
                  class="bg-surface-high hover:bg-surface-high/80 text-text-secondary px-3 py-2.5 text-xs font-bold transition-all border-l border-surface-high/50 active:scale-95 whitespace-nowrap"
                  title="Full Scan — re-reads all file dimensions, hashes, and metadata">
                  Full
                </button>
              </div>
           </div>
       </div>
    </div>
  `,
    styles: []
})
export class DatasetToolbarComponent {
    initialSearch = input('');
    searchValue = signal('');

    searchChanged = output<string>();
    createRequested = output<void>();
    rescanLibraryRequested = output<boolean>();

    _ = effect(() => {
        const v = this.initialSearch();
        this.searchValue.set(v);
    }, { allowSignalWrites: true });

    onSearchChange(value: string) {
        this.searchValue.set(value);
        this.searchChanged.emit(value);
    }
}

