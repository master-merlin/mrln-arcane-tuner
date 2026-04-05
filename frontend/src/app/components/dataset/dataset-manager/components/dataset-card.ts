import { Component, inject, input, output, computed, signal } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';
import { Dataset } from '../../../../services/dataset';
import { RuntimeConfigService } from '../../../../services/runtime-config.service';

@Component({
    selector: 'app-dataset-card',
    standalone: true,
    imports: [DatePipe, DecimalPipe],
    template: `
    <div [attr.data-testid]="'dataset-card-' + dataset().name"
         class="bg-surface-mid border rounded-theme-xl p-5 hover:bg-surface-high/50 transition-all group relative overflow-hidden cursor-pointer h-full flex flex-col" 
         [class.border-surface-high/50]="!dragOver()"
         [class.border-brand]="dragOver()"
         [class.ring-2]="dragOver()"
         [class.ring-brand/50]="dragOver()"
         (click)="view.emit(dataset())"
         (dragover)="onDragOver($event)"
         (dragleave)="onDragLeave($event)"
         (drop)="onDrop($event)">
       
       <!-- Drop Zone Overlay -->
       @if (dragOver()) {
           <div class="absolute inset-0 bg-brand/10 backdrop-blur-sm z-30 flex flex-col items-center justify-center gap-2 pointer-events-none animate-fadeIn">
               <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-brand">
                   <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                   <polyline points="17 8 12 3 7 8"></polyline>
                   <line x1="12" y1="3" x2="12" y2="15"></line>
               </svg>
               <span class="text-brand font-bold text-xs uppercase tracking-widest">Drop to Upload</span>
           </div>
       }

       <!-- Header Section -->
       <div class="flex gap-4 mb-4">
          <!-- Thumbnail (Left) -->
          <div class="w-28 h-28 bg-base rounded-theme-md overflow-hidden border border-surface-mid flex-shrink-0 relative group-hover:border-brand/50 transition-colors">
              @if (dataset().preview_image && !dataset().missing) {
                  <img [src]="previewUrl()" class="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" [alt]="dataset().name">
              } @else {
                  <div class="w-full h-full flex items-center justify-center text-text-disabled bg-surface-low">
                     <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                  </div>
              }
              <!-- Version Overlay -->
              @if (dataset().version) {
                  <div class="absolute bottom-1 right-1 bg-overlay backdrop-blur-sm border border-surface-high px-1.5 py-0.5 rounded text-[9px] font-mono text-text-secondary shadow-xl">
                      v{{ dataset().version }}
                  </div>
              }
          </div>

          <!-- Text (Right) -->
          <div class="flex-1 min-w-0">
              <h3 class="text-white font-semibold flex items-center gap-2 truncate text-lg">
                  {{ dataset().name }}
                  @if (dataset().missing) {
                      <span class="bg-danger/20 text-danger text-[10px] px-1.5 py-0.5 rounded border border-danger/30">MISSING</span>
                  }
              </h3>
              @if (dataset().classifier) {
                  <div class="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-brand/10 text-brand border border-brand/20 mb-1 capitalize">
                      {{ dataset().classifier }}
                  </div>
              }
              <p class="text-text-muted text-xs line-clamp-2 h-8">{{ dataset().description || 'No description provided.' }}</p>
          </div>
       </div>

       <!-- Metadata Grid -->
       <div class="flex flex-col gap-2 text-xs bg-surface-low/20 rounded-theme-md p-3 border border-surface-high/30 mb-4 flex-grow">
          <!-- Path -->
          <div class="flex items-center gap-2">
            <span class="text-text-subtle w-16 flex-shrink-0 font-medium">Path</span>
            <div class="text-text-muted truncate font-mono" [title]="dataset().path">{{ dataset().path }}</div>
          </div>

          <!-- Total Files & Size -->
          <div class="flex items-center gap-2">
              <span class="text-text-subtle w-16 font-medium">Files</span>
              <span class="text-text-primary font-mono">{{ dataset().file_count }}</span>
              <span class="text-text-subtle font-mono ml-auto" title="Total Size">{{ (dataset().total_size_bytes || 0) / 1048576 | number:'1.1-1' }} MB</span>
          </div>

          <!-- Media & Captions -->
          <div class="flex items-center gap-4">
              <div class="flex items-center gap-2">
                   <span class="text-text-subtle w-16 font-medium">Images</span>
                   <span class="text-text-primary font-mono">{{ dataset().multimedia_count || 0 }}</span>
              </div>
               <div class="flex items-center gap-2">
                  <span class="text-text-subtle font-medium">Captions</span>
                  <span class="text-text-primary font-mono">{{ dataset().caption_count || 0 }}</span>
              </div>
              @if (dataset().median_quality_score != null) {
                  <div class="flex items-center gap-2 ml-auto">
                      <span class="text-text-subtle font-medium">Median</span>
                      <span class="flex-shrink-0 px-1 py-px rounded-sm font-bold font-mono" [class]="getScoreColor(dataset().median_quality_score!)">{{ dataset().median_quality_score!.toFixed(4) }}</span>
                  </div>
              }
          </div>
          
          <!-- Indicators -->
          <div class="flex items-center justify-center gap-6 pt-2 mt-auto border-t border-surface-high/30">
              <!-- Harmonized Indicator -->
               <div class="flex items-center gap-1.5" [title]="dataset().harmonization_score ? (((dataset().harmonization_score || 0) * 100) | number:'1.0-1') + '% Harmonized' : 'Not Harmonized'">
                  @if (dataset().harmonization_score && ((dataset().harmonization_score || 0) >= 0.99)) {
                     <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="text-success"><polyline points="20 6 9 17 4 12"></polyline></svg>
                     <span class="text-success font-medium">Harmonized</span>
                  } @else if (dataset().harmonization_score && ((dataset().harmonization_score || 0) >= 0.75)) {
                     <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="text-warning"><polyline points="20 6 9 17 4 12"></polyline></svg>
                     <span class="text-warning font-medium">Harmonized</span>
                  } @else {
                      <div class="w-3.5 h-3.5 rounded-full border border-border-default"></div>
                      <span class="text-text-subtle">Harmonized</span>
                  }
              </div>

              <!-- Fully Captioned -->
              <div class="flex items-center gap-1.5">
                  @if (dataset().caption_coverage && dataset().multimedia_count > 0) {
                     <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="text-success"><polyline points="20 6 9 17 4 12"></polyline></svg>
                     <span class="text-success font-medium">Captioned</span>
                  } @else {
                      <div class="w-3.5 h-3.5 rounded-full border border-border-default"></div>
                      <span class="text-text-subtle">Captioned</span>
                  }
              </div>

              <!-- Masks Indicator -->
              <div class="flex items-center gap-1.5">
                  @if (dataset().mask_count && dataset().mask_count >= (dataset().multimedia_count || 0) && dataset().multimedia_count > 0) {
                     <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="text-success"><polyline points="20 6 9 17 4 12"></polyline></svg>
                     <span class="text-success font-medium">Masks</span>
                  } @else if (dataset().mask_count && dataset().mask_count > 0) {
                     <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="text-warning"><polyline points="20 6 9 17 4 12"></polyline></svg>
                     <span class="text-warning font-medium">Masks</span>
                  } @else {
                      <div class="w-3.5 h-3.5 rounded-full border border-border-default"></div>
                      <span class="text-text-subtle">Masks</span>
                  }
              </div>
          </div>
       </div>

       <!-- Last Scanned (outside the box, small) -->
       <div class="text-[9px] text-text-disabled italic text-right px-1 py-1">
           {{ dataset().last_scanned_at ? ('Scanned ' + ((dataset().last_scanned_at || 0) * 1000 | date:'short')) : 'Never scanned' }}
       </div>

       <!-- Action Bar -->
       <div class="mt-auto pt-3 border-t border-surface-high/50 flex items-center justify-center">
          <div class="flex justify-center gap-0.5">
             @if (hasActiveProject()) {
             @if (isInProject()) {
              <button (click)="removeFromProject.emit(dataset()); $event.stopPropagation()" 
                   data-testid="btn-remove-from-project"
                   class="text-emerald-400 hover:text-danger p-1.5 rounded transition-colors" title="Remove from project">
                   <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path><line x1="9" y1="14" x2="15" y2="14"></line></svg>
              </button>
             } @else {
              <button (click)="addToProject.emit(dataset()); $event.stopPropagation()" 
                   data-testid="btn-add-to-project"
                   class="text-text-subtle hover:text-emerald-400 p-1.5 rounded transition-colors" title="Add to project">
                   <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path><line x1="12" y1="11" x2="12" y2="17"></line><line x1="9" y1="14" x2="15" y2="14"></line></svg>
              </button>
             }
           }

             <button (click)="view.emit(dataset()); $event.stopPropagation()" 
                  data-testid="btn-view-dataset"
                  class="text-text-subtle hover:text-success p-1.5 rounded transition-colors" title="Open Viewer">
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
             </button>
             <button (click)="edit.emit(dataset()); $event.stopPropagation()" 
                  data-testid="btn-edit-dataset"
                  class="text-text-subtle hover:text-brand p-1.5 rounded transition-colors" title="Edit Metadata">
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
             </button>
             <button (click)="rescan.emit(dataset()); $event.stopPropagation()" 
                  data-testid="btn-rescan-dataset"
                  class="text-text-subtle hover:text-brand p-1.5 rounded transition-colors" title="Scan for files">
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.3"/></svg>
             </button>
              <button (click)="cache.emit(dataset()); $event.stopPropagation()" 
                   [disabled]="!dataset().has_cache"
                   data-testid="btn-cache-dataset"
                   class="text-text-subtle hover:text-brand p-1.5 rounded transition-colors"
                   [class.opacity-30]="!dataset().has_cache"
                   [class.cursor-not-allowed]="!dataset().has_cache"
                   [class.hover:text-brand]="dataset().has_cache"
                   [title]="dataset().has_cache ? 'Cache Administration' : 'No cache data'">
                   <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>
             </button>
              <button (click)="download.emit(dataset()); $event.stopPropagation()" 
                   data-testid="btn-download-dataset"
                   class="text-text-subtle hover:text-brand p-1.5 rounded transition-colors" title="Download as Zip">
                   <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 10V2"></path><path d="m8 6 4-4 4 4"></path><rect x="2" y="14" width="20" height="8" rx="2"></rect><path d="M6 18h.01"></path><path d="M10 18h.01"></path></svg>
              </button>
              <button (click)="delete.emit(dataset()); $event.stopPropagation()" 
                  data-testid="btn-delete-dataset"
                  class="text-text-subtle hover:text-danger p-1.5 rounded transition-colors" title="Delete Dataset & Files">
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"></path><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
              </button>
          </div>

          <div class="flex justify-end">
              <input type="file" multiple (change)="onFileSelected($event)" (click)="$event.stopPropagation()" class="hidden" #fileInput>
              <button (click)="fileInput.click(); $event.stopPropagation()" 
                  data-testid="btn-upload-files"
                  class="text-[10px] bg-surface-mid hover:bg-surface-high text-text-secondary px-3 py-1.5 rounded-theme-md transition-colors flex items-center gap-2 font-bold uppercase tracking-tighter">
                  <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                  Upload
              </button>
          </div>
       </div>
    </div>
  `,
    styles: []
})
export class DatasetCardComponent {
    dataset = input.required<Dataset>();
    activeProjectId = input<string | null>(null);
    isInProject = input<boolean>(false);

    /** Robust project check: handles string 'null' from select binding */
    hasActiveProject = computed(() => {
        const pid = this.activeProjectId();
        return !!pid && pid !== 'null';
    });

    view = output<Dataset>();
    edit = output<Dataset>();
    rescan = output<Dataset>();
    cache = output<Dataset>();
    download = output<Dataset>();
    delete = output<Dataset>();
    upload = output<{ datasetName: string, files: FileList }>();
    addToProject = output<Dataset>();
    removeFromProject = output<Dataset>();

    dragOver = signal(false);

    private rtc = inject(RuntimeConfigService);

    previewUrl = computed(() => {
        const ds = this.dataset();
        return `${this.rtc.mediaBaseUrl}/${encodeURIComponent(ds.name)}/${ds.preview_image}`;
    });

    onFileSelected(event: any) {
        const files = event.target.files;
        if (files.length > 0) {
            this.upload.emit({ datasetName: this.dataset().name, files });
            event.target.value = '';
        }
    }

    onDragOver(event: DragEvent) {
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer?.types.includes('Files')) {
            event.dataTransfer.dropEffect = 'copy';
            this.dragOver.set(true);
        }
    }

    onDragLeave(event: DragEvent) {
        event.preventDefault();
        event.stopPropagation();
        this.dragOver.set(false);
    }

    onDrop(event: DragEvent) {
        event.preventDefault();
        event.stopPropagation();
        this.dragOver.set(false);

        const files = event.dataTransfer?.files;
        if (files && files.length > 0) {
            this.upload.emit({ datasetName: this.dataset().name, files });
        }
    }

    getScoreColor(score: number): string {
        if (score >= 0.27) return 'bg-success/80 text-white';
        if (score >= 0.24) return 'bg-warning/80 text-black';
        return 'bg-danger/80 text-white';
    }
}
