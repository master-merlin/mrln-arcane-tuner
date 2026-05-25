import { Component, input, output, inject, signal, OnInit } from '@angular/core';
import { DatasetService } from '../../../../services/dataset';
import { ToastService } from '../../../../services/toast';
import { DecimalPipe } from '@angular/common';

interface CacheVariant {
    name: string;
    selected: boolean;
}

interface CacheType {
    name: string;
    selected: boolean;
    variants: CacheVariant[];
}

interface CacheVersion {
    version: string;
    selected: boolean;
    types: CacheType[];
    sizeBytes: number;
}

interface CacheModel {
    name: string;
    selected: boolean;
    expanded: boolean;
    versions: CacheVersion[];
    totalSize: number;
}

@Component({
    selector: 'app-viewer-cache-admin-modal',
    standalone: true,
    imports: [DecimalPipe],
    template: `
        <div class="fixed inset-0 z-[110] flex items-center justify-center p-6 backdrop-blur-sm bg-base/70 bubble-in">
            <div class="bg-surface-low border border-surface-high w-full max-w-2xl rounded-theme-2xl shadow-2xl overflow-hidden border-shine">
                <!-- Header -->
                <div class="p-6 border-b border-surface-high flex items-center justify-between bg-surface-mid/50">
                    <div>
                        <h2 class="text-xl font-bold text-white">Cache Administration</h2>
                        <p class="text-xs text-text-subtle font-medium tracking-wide uppercase mt-1">{{ datasetName() }} — Training Cache</p>
                    </div>
                    <button (click)="close.emit()" class="p-2 hover:bg-surface-high rounded-theme-lg transition-colors text-text-muted hover:text-white">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                    </button>
                </div>

                <div class="p-6 space-y-4 max-h-[65vh] overflow-y-auto">
                    @if (isLoading()) {
                        <div class="flex items-center justify-center py-12 text-text-subtle">
                            <svg class="w-6 h-6 animate-spin mr-3" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>
                            Loading cache info...
                        </div>
                    } @else if (models().length === 0) {
                        <div class="text-center py-12">
                            <svg class="w-12 h-12 text-text-disabled mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-2.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"></path></svg>
                            <p class="text-text-subtle font-medium">No cached data found</p>
                            <p class="text-text-disabled text-xs mt-1">Cache will be created during training pre-processing</p>
                        </div>
                    } @else {
                        <!-- Total Size Banner -->
                        <div class="flex items-center justify-between bg-surface-mid/50 rounded-theme-lg px-4 py-3 border border-surface-high/50">
                            <span class="text-xs text-text-muted font-bold uppercase tracking-wider">Total Cache Size</span>
                            <span class="text-sm font-bold text-white font-mono">{{ totalCacheSize() / 1048576 | number:'1.1-1' }} MB</span>
                        </div>

                        <!-- Model Tree -->
                        @for (model of models(); track model.name) {
                            <div class="bg-surface-mid/30 rounded-theme-lg border border-surface-high/30 overflow-hidden">
                                <!-- Model Header -->
                                <div class="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-surface-mid/50 transition-colors"
                                     (click)="toggleExpand(model)">
                                    <!-- Expand Arrow -->
                                    <svg class="w-4 h-4 text-text-subtle transition-transform" [class.rotate-90]="model.expanded" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                                    </svg>
                                    <!-- Checkbox -->
                                    <label class="flex items-center gap-2 cursor-pointer" (click)="$event.stopPropagation()">
                                        <input type="checkbox" [checked]="model.selected" (change)="toggleModel(model)" class="accent-brand w-4 h-4 cursor-pointer">
                                    </label>
                                    <span class="text-white font-semibold text-sm flex-1">{{ model.name }}</span>
                                    <span class="text-xs text-text-subtle font-mono">{{ model.totalSize / 1048576 | number:'1.1-1' }} MB</span>
                                </div>

                                <!-- Expanded Details -->
                                @if (model.expanded) {
                                    <div class="border-t border-surface-high/20">
                                        @for (ver of model.versions; track ver.version) {
                                            <div class="px-4 py-2 ml-6">
                                                <div class="flex items-center gap-2 mb-2">
                                                    <label class="flex items-center gap-2 cursor-pointer" (click)="$event.stopPropagation()">
                                                        <input type="checkbox" [checked]="ver.selected" (change)="toggleVersion(model, ver)" class="accent-brand w-3.5 h-3.5 cursor-pointer">
                                                    </label>
                                                    <span class="text-[10px] text-brand font-bold uppercase tracking-widest">v{{ ver.version }}</span>
                                                    <span class="text-[10px] text-text-disabled font-mono">{{ ver.sizeBytes / 1048576 | number:'1.1-1' }} MB</span>
                                                </div>
                                                @for (type of ver.types; track type.name) {
                                                    <div class="ml-4 mb-2">
                                                        <div class="flex items-center gap-2 mb-1">
                                                            <label class="flex items-center gap-2 cursor-pointer">
                                                                <input type="checkbox" [checked]="type.selected" (change)="toggleType(model, type)" class="accent-brand w-3.5 h-3.5 cursor-pointer">
                                                                <span class="text-xs text-text-secondary font-medium capitalize">{{ type.name }}</span>
                                                            </label>
                                                        </div>
                                                        <!-- Variants (hide _flat pseudo-variants) -->
                                                        @for (variant of type.variants; track variant.name) {
                                                            @if (variant.name !== '_flat') {
                                                            <div class="ml-6 flex items-center gap-2 py-0.5">
                                                                <label class="flex items-center gap-2 cursor-pointer">
                                                                    <input type="checkbox" [checked]="variant.selected" (change)="toggleVariant(model, type, variant)" class="accent-brand w-3 h-3 cursor-pointer">
                                                                    <span class="text-[11px] text-text-muted">{{ variant.name }}</span>
                                                                </label>
                                                            </div>
                                                            }
                                                        }
                                                    </div>
                                                }
                                            </div>
                                        }
                                    </div>
                                }
                            </div>
                        }
                    }
                </div>

                <!-- Footer Actions -->
                @if (models().length > 0) {
                    <div class="p-4 border-t border-surface-high flex items-center justify-between bg-surface-mid/30">
                        <div class="flex items-center gap-2">
                            <button (click)="selectAll()"
                                class="px-4 py-2 text-xs font-medium rounded-theme-lg bg-surface-mid/60 hover:bg-surface-mid text-text-muted hover:text-white transition-all border border-surface-high/30">
                                {{ allSelected() ? 'Deselect All' : 'Select All' }}
                            </button>
                        </div>
                        <div class="flex items-center gap-2">
                            <button (click)="close.emit()" class="px-4 py-2 text-xs text-text-muted hover:text-white transition-colors font-medium">
                                Cancel
                            </button>
                            @if (hasOutdated()) {
                                <button (click)="selectOutdated(); confirmPurge()" [disabled]="isPurging()"
                                    class="px-4 py-2 bg-amber-600/20 text-amber-400 hover:bg-amber-600/30 rounded-theme-lg transition-colors text-xs font-bold border border-amber-600/30 flex items-center gap-1.5 disabled:opacity-40">
                                    <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                                    Purge Outdated
                                </button>
                            }
                            <button (click)="confirmPurge()"
                                    [disabled]="isPurging() || !hasSelection()"
                                    class="px-5 py-2 bg-danger/20 text-danger hover:bg-danger/30 rounded-theme-lg transition-colors text-xs font-bold border border-danger/30 flex items-center gap-1.5 disabled:opacity-40 disabled:cursor-not-allowed">
                                @if (isPurging()) {
                                    <svg class="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>
                                    Purging...
                                } @else {
                                    <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                                    Purge Selected
                                }
                            </button>
                        </div>
                    </div>
                }
            </div>
        </div>
    `,
    styles: []
})
export class ViewerCacheAdminModalComponent implements OnInit {

    datasetName = input.required<string>();
    close = output<void>();

    private datasetService = inject(DatasetService);
    private toast = inject(ToastService);

    isLoading = signal(true);
    isPurging = signal(false);
    models = signal<CacheModel[]>([]);

    totalCacheSize = signal(0);

    ngOnInit() {
        this.loadCacheTree();
    }

    private loadCacheTree() {
        this.isLoading.set(true);
        this.datasetService.listCache(this.datasetName()).subscribe({
            next: (res: any) => {
                const tree = res.cache || {};
                const modelList: CacheModel[] = [];
                let total = 0;

                for (const [modelName, versions] of Object.entries<any>(tree)) {
                    const versionList: CacheVersion[] = [];
                    let modelTotal = 0;

                    for (const [ver, data] of Object.entries<any>(versions)) {
                        const types: CacheType[] = [];

                        // Dynamically discover all cache types
                        const typesObj = data.types || {};
                        for (const [typeName, variantsObj] of Object.entries<any>(typesObj)) {
                            if (variantsObj && Object.keys(variantsObj).length > 0) {
                                types.push({
                                    name: typeName,
                                    selected: false,
                                    variants: Object.keys(variantsObj).map(v => ({
                                        name: v,
                                        selected: false
                                    }))
                                });
                            }
                        }

                        const sz = data.size_bytes || 0;
                        modelTotal += sz;
                        versionList.push({ version: ver, types, sizeBytes: sz, selected: false });
                    }

                    total += modelTotal;
                    modelList.push({
                        name: modelName,
                        selected: false,
                        expanded: false,
                        versions: versionList,
                        totalSize: modelTotal
                    });
                }

                this.models.set(modelList);
                this.totalCacheSize.set(total);
                this.isLoading.set(false);
            },
            error: (err) => {
                this.toast.error('Failed to load cache: ' + (err.error?.detail || err.message));
                this.isLoading.set(false);
            }
        });
    }

    toggleExpand(model: CacheModel) {
        model.expanded = !model.expanded;
        this.models.update(m => [...m]);
    }

    toggleModel(model: CacheModel) {
        model.selected = !model.selected;
        for (const ver of model.versions) {
            ver.selected = model.selected;
            for (const type of ver.types) {
                type.selected = model.selected;
                for (const v of type.variants) {
                    v.selected = model.selected;
                }
            }
        }
        this.models.update(m => [...m]);
    }

    toggleVersion(model: CacheModel, ver: CacheVersion) {
        ver.selected = !ver.selected;
        for (const type of ver.types) {
            type.selected = ver.selected;
            for (const v of type.variants) {
                v.selected = ver.selected;
            }
        }
        model.selected = model.versions.every(v => v.selected);
        this.models.update(m => [...m]);
    }

    toggleType(model: CacheModel, type: CacheType) {
        type.selected = !type.selected;
        for (const v of type.variants) {
            v.selected = type.selected;
        }
        // Cascade up: type → version → model
        for (const ver of model.versions) {
            ver.selected = ver.types.every(t => t.selected);
        }
        model.selected = model.versions.every(ver => ver.selected);
        this.models.update(m => [...m]);
    }

    toggleVariant(model: CacheModel, type: CacheType, variant: CacheVariant) {
        variant.selected = !variant.selected;
        type.selected = type.variants.every(v => v.selected);
        for (const ver of model.versions) {
            ver.selected = ver.types.every(t => t.selected);
        }
        model.selected = model.versions.every(ver => ver.selected);
        this.models.update(m => [...m]);
    }

    hasSelection(): boolean {
        return this.models().some(m =>
            m.versions.some(v =>
                v.types.some(t =>
                    t.variants.some(va => va.selected)
                )
            )
        );
    }

    allSelected(): boolean {
        return this.models().length > 0 && this.models().every(m => m.selected);
    }

    selectAll() {
        const newState = !this.allSelected();
        for (const model of this.models()) {
            model.selected = newState;
            for (const ver of model.versions) {
                ver.selected = newState;
                for (const type of ver.types) {
                    type.selected = newState;
                    for (const v of type.variants) {
                        v.selected = newState;
                    }
                }
            }
        }
        this.models.update(m => [...m]);
    }

    hasOutdated(): boolean {
        return this.models().some(m => m.versions.length > 1);
    }

    selectOutdated() {
        // For each model, find the latest version and select all older ones
        for (const model of this.models()) {
            if (model.versions.length <= 1) continue;

            // Sort versions descending by semantic version
            const sorted = [...model.versions].sort((a, b) => {
                const pa = a.version.split('.').map(Number);
                const pb = b.version.split('.').map(Number);
                for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
                    const diff = (pb[i] || 0) - (pa[i] || 0);
                    if (diff !== 0) return diff;
                }
                return 0;
            });

            const latestVersion = sorted[0].version;
            for (const ver of model.versions) {
                const isOutdated = ver.version !== latestVersion;
                ver.selected = isOutdated;
                for (const type of ver.types) {
                    type.selected = isOutdated;
                    for (const v of type.variants) {
                        v.selected = isOutdated;
                    }
                }
            }
            model.selected = model.versions.every(v => v.selected);
        }
        this.models.update(m => [...m]);
    }

    confirmPurge() {
        if (!confirm('Are you sure you want to purge the selected cache items? This cannot be undone.')) return;

        // Build purge request from selections
        const selectedModels: string[] = [];
        const selectedTypes = new Set<string>();
        const selectedVariants = new Set<string>();

        for (const model of this.models()) {
            const hasSelected = model.versions.some(v =>
                v.types.some(t => t.variants.some(va => va.selected))
            );
            if (!hasSelected) continue;

            selectedModels.push(model.name);
            for (const ver of model.versions) {
                for (const type of ver.types) {
                    if (type.variants.some(va => va.selected)) {
                        selectedTypes.add(type.name);
                        for (const v of type.variants) {
                            if (v.selected) selectedVariants.add(v.name);
                        }
                    }
                }
            }
        }

        this.isPurging.set(true);
        this.datasetService.purgeCache(this.datasetName(), {
            models: selectedModels.length > 0 ? selectedModels : undefined,
            types: selectedTypes.size > 0 ? Array.from(selectedTypes) : undefined,
            variants: selectedVariants.size > 0 ? Array.from(selectedVariants) : undefined,
        }).subscribe({
            next: (res: any) => {
                this.isPurging.set(false);
                const freedMB = ((res.freed_bytes || 0) / 1048576).toFixed(1);
                this.toast.success(`Purged ${res.deleted} cache entries (${freedMB} MB freed)`);
                this.loadCacheTree();
            },
            error: (err) => {
                this.isPurging.set(false);
                this.toast.error('Purge failed: ' + (err.error?.detail || err.message));
            }
        });
    }
}
