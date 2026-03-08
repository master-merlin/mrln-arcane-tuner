import { Component, output, inject, signal, OnInit, OnDestroy } from '@angular/core';
import { DatasetService } from '../../../../services/dataset';
import { WebSocketService } from '../../../../services/websocket.service';
import { Subscription } from 'rxjs';

@Component({
    selector: 'app-viewer-rescan-modal',
    standalone: true,
    imports: [],
    template: `
        <div class="fixed inset-0 z-[110] flex items-center justify-center p-6 backdrop-blur-sm bg-base/70 bubble-in">
            <div class="bg-surface-low border border-surface-high w-full max-w-2xl rounded-theme-2xl shadow-2xl overflow-hidden border-shine">
                <!-- Header -->
                <div class="p-6 border-b border-surface-high flex items-center justify-between bg-surface-mid/50">
                    <div>
                        <h2 class="text-xl font-bold text-white">Library Rescan</h2>
                        <p class="text-xs text-text-subtle font-medium tracking-wide uppercase mt-1">Scanning all datasets for changes</p>
                    </div>
                    <!-- Close button disabled during scan unless needed -->
                </div>
                
                <div class="p-8 space-y-8 max-h-[70vh] overflow-y-auto">
                    
                    <!-- Overall Library Progress -->
                    <div class="space-y-4">
                        <div class="flex justify-between items-end">
                            <div class="space-y-1">
                                <span class="text-[10px] text-brand font-bold uppercase tracking-widest block">Library Status</span>
                                <h3 class="text-xl font-black text-white italic">
                                    {{ libraryProgress().current }} / {{ libraryProgress().total }} Datasets
                                </h3>
                            </div>
                            <div class="text-sm font-bold text-brand">
                                {{ getLibraryPercent() }}%
                            </div>
                        </div>
                        <div class="h-2 w-full bg-base/50 rounded-full overflow-hidden border border-white/5">
                            <div class="h-full bg-brand rounded-full transition-all duration-300" [style.width.%]="getLibraryPercent()"></div>
                        </div>
                    </div>

                    <!-- Current Dataset Progress -->
                     <div class="bg-surface-mid p-6 rounded-theme-2xl border border-surface-high relative overflow-hidden group">
                        <div class="absolute inset-0 bg-brand/5 animate-pulse"></div>
                        <div class="relative z-10">
                            <div class="flex justify-between items-end mb-4">
                                <div class="space-y-1">
                                    <span class="text-[10px] text-brand font-bold uppercase tracking-widest block">
                                        {{ datasetProgress().name || 'Initializing...' }}
                                    </span>
                                    <h3 class="text-2xl font-black text-white italic">
                                        {{ getDatasetPercent() }}%
                                    </h3>
                                </div>
                                <div class="text-right">
                                    <span class="text-[10px] text-text-subtle font-bold uppercase tracking-widest block mb-1">Files Scanned</span>
                                    <span class="text-xs font-mono text-text-secondary">
                                        {{ datasetProgress().current }} / {{ datasetProgress().total }}
                                    </span>
                                </div>
                            </div>
                            
                            <!-- Activity Status -->
                            <p class="text-xs font-mono text-text-muted mb-3 truncate">
                                <span class="text-brand mr-2">➜</span>{{ datasetProgress().status }}
                                <span class="text-text-disabled ml-2 italic">{{ datasetProgress().file }}</span>
                            </p>

                            <!-- Progress Bar -->
                            <div class="h-3 w-full bg-base/50 rounded-full overflow-hidden border border-white/5 p-[2px]">
                                <div class="h-full bg-gradient-to-r from-brand to-brand-bright rounded-full transition-all duration-300 shadow-[0_0_10px_rgba(255,51,102,0.5)]" 
                                     [style.width.%]="getDatasetPercent()">
                                    <div class="w-full h-full animate-shimmer bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>
                                </div>
                            </div>
                        </div>
                     </div>
                     
                     @if (isComplete()) {
                        <div class="w-full py-4 flex items-center justify-center bg-success/20 border border-success/50 text-success rounded-theme-2xl text-base font-black italic uppercase tracking-wider shadow-[0_0_15px_rgba(var(--color-success),0.2)]">
                            Scanning Complete...
                        </div>
                     }
                </div>
            </div>
        </div>
    `,
    styles: []
})
export class ViewerRescanModalComponent implements OnInit, OnDestroy {

    close = output<void>();
    completed = output<void>();

    datasetService = inject(DatasetService);
    wsService = inject(WebSocketService);

    private subs: Subscription[] = [];

    // State
    libraryProgress = signal({ current: 0, total: 0 });
    datasetProgress = signal({ name: '', current: 0, total: 0, file: '', status: 'Waiting...' });
    isComplete = signal(false);

    ngOnInit() {
        this.startRescan();
        this.subscribeToEvents();
    }

    startRescan() {
        this.datasetService.scanAllDatasets().subscribe({
            error: (err) => console.error(err)
        });
    }

    subscribeToEvents() {
        // Library Start (Estimated Total)
        this.subs.push(this.wsService.on<any>('rescan_start').subscribe(payload => {
            this.libraryProgress.set({ current: 0, total: payload.total_datasets });
        }));

        // Dataset Start
        this.subs.push(this.wsService.on<any>('dataset_start').subscribe(payload => {
            this.libraryProgress.set({ current: payload.index, total: payload.total });
            this.datasetProgress.set({
                name: payload.name,
                current: 0,
                total: 0,
                file: '',
                status: 'Starting scan...'
            });
        }));

        // File Progress
        this.subs.push(this.wsService.on<any>('scan_progress').subscribe(payload => {
            this.datasetProgress.set({
                name: payload.dataset,
                current: payload.current,
                total: payload.total,
                file: payload.file,
                status: payload.status
            });
        }));

        // Complete
        this.subs.push(this.wsService.on<any>('rescan_complete').subscribe(() => {
            this.isComplete.set(true);
            this.completed.emit();
            setTimeout(() => this.close.emit(), 1000); // Auto-close after 1 second
        }));
    }

    getLibraryPercent() {
        const { current, total } = this.libraryProgress();
        if (total === 0) return 0;
        return ((current / total) * 100).toFixed(1);
    }

    getDatasetPercent() {
        const { current, total } = this.datasetProgress();
        if (total === 0) return 0;
        return ((current / total) * 100).toFixed(1);
    }

    ngOnDestroy() {
        this.subs.forEach(s => s.unsubscribe());
    }
}
