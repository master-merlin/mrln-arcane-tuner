import { Component, OnInit, OnDestroy, inject, input, output, signal } from '@angular/core';
import { DatasetService } from '../../../../services/dataset';
import { WebSocketService } from '../../../../services/websocket.service';
import { Subscription } from 'rxjs';

@Component({
    selector: 'app-dataset-single-rescan-modal',
    standalone: true,
    template: `
    <div class="fixed inset-0 z-[130] flex items-center justify-center p-6 backdrop-blur-sm bg-base/70 bubble-in">
        <div class="bg-surface-low border border-surface-high w-full max-w-lg rounded-theme-2xl shadow-2xl overflow-hidden border-shine">
            <!-- Header -->
            <div class="p-6 border-b border-surface-high flex items-center justify-between bg-surface-mid/50">
                <div>
                    <h2 class="text-xl font-bold text-white">Rescanning Dataset</h2>
                    <p class="text-xs text-text-subtle font-medium tracking-wide uppercase mt-1">{{ datasetName() }}</p>
                </div>
            </div>
            
            <div class="p-8 space-y-6">
                 <!-- Current Dataset Progress -->
                 <div class="bg-surface-mid p-6 rounded-theme-2xl border border-surface-high relative overflow-hidden group">
                    <div class="absolute inset-0 bg-brand/5 animate-pulse"></div>
                    <div class="relative z-10">
                        <div class="flex justify-between items-end mb-4">
                            <div class="space-y-1">
                                <span class="text-[10px] text-brand font-bold uppercase tracking-widest block">
                                    {{ forceFull() ? 'Full Scan' : 'Incremental Scan' }}
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
                 
                 @if (error()) {
                    <div class="p-4 bg-danger/10 border border-danger/30 rounded-theme-xl text-danger text-sm">
                        {{ error() }}
                    </div>
                    <button (click)="close.emit()" class="w-full py-4 bg-surface-mid hover:bg-surface-high border border-surface-high text-white rounded-theme-2xl text-base font-bold transition-all">
                        Close
                    </button>
                 } @else if (isComplete()) {
                    <div class="w-full py-4 flex items-center justify-center bg-success/20 border border-success/50 text-success rounded-theme-2xl text-base font-black italic uppercase tracking-wider shadow-[0_0_15px_rgba(var(--color-success),0.2)]">
                        Scanning Complete...
                    </div>
                 }
            </div>
        </div>
    </div>
  `
})
export class DatasetSingleRescanModalComponent implements OnInit, OnDestroy {
    datasetName = input.required<string>();
    forceFull = input.required<boolean>();

    close = output<void>();
    completed = output<void>();

    private datasetService = inject(DatasetService);
    private wsService = inject(WebSocketService);
    private subs: Subscription[] = [];

    datasetProgress = signal({ current: 0, total: 0, file: '', status: 'Initializing scan...' });
    isComplete = signal(false);
    error = signal<string | null>(null);

    ngOnInit() {
        this.subscribeToEvents();
        this.startScan();
    }

    startScan() {
        this.datasetService.scanDataset(this.datasetName(), this.forceFull()).subscribe({
            next: () => {
                // The scan finished successfully
                this.isComplete.set(true);
                this.datasetProgress.update(p => ({ ...p, status: 'Scan finalized.', current: Math.max(p.total, p.current) }));
                this.completed.emit();
                setTimeout(() => this.close.emit(), 1000); // Auto-close after 1 second
            },
            error: (err) => {
                console.error('Scan failed', err);
                this.error.set(err.message || 'Scan failed to complete.');
            }
        });
    }

    subscribeToEvents() {
        this.subs.push(this.wsService.on<any>('scan_progress').subscribe(payload => {
            // Only listen for OUR dataset
            if (payload.dataset === this.datasetName()) {
                this.datasetProgress.set({
                    current: payload.current,
                    total: payload.total,
                    file: payload.file,
                    status: payload.status
                });
            }
        }));
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
