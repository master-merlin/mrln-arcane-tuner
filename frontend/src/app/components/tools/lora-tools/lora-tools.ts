import { Component, ChangeDetectionStrategy, inject, input, signal, OnDestroy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import uPlot from 'uplot';

export type ToolTab = 'inspect' | 'resize';

/** One LoRA layer's weight-delta stats (`inspect_lora().layer_details[]`). */
export interface LoraLayerDetail {
  module: string;
  component: string;
  norm_delta: number;
  strength: number;
  /** Bar width %, computed client-side in sortedLayerDetails(). */
  _barPct?: number;
}

/** Aggregate norm stats across all layers. */
export interface LoraNormSummary {
  mean_norm: number;
  std_norm: number;
  max_norm: number;
  max_norm_layer: string;
  min_norm: number;
  min_norm_layer: string;
}

/** Layer-relevance / speed-training analysis. Fields are required because the
 *  template reads them via non-null assertion inside an `@if` that gates the
 *  whole section on its presence (Angular evaluates that guard at runtime). */
export interface LoraLayerRelevance {
  essential_count: number;
  total_layers: number;
  essential_params_pct: number;
  speed_gain_pct: number;
  target_module_patterns: string[];
  essential_modules: string[];
  tier_map: Record<string, string>;
}

/**
 * `GET /tools/lora/inspect` result. The backend returns a free-form dict; these
 * are the fields this component actually reads (all optional — older/partial
 * inspections may omit sections).
 */
export interface LoraInspectResult {
  format?: string;
  rank?: number;
  alpha?: number;
  lora_modules?: number;
  dtype?: string;
  file_size_mb?: number;
  path?: string;
  layer_details?: LoraLayerDetail[];
  // Required (not optional): the template reads `.norm_summary.x` /
  // `.layer_relevance.y` via `!` inside `@if`-guarded sections — Angular still
  // evaluates the guard on the real (possibly-absent) runtime value.
  norm_summary: LoraNormSummary;
  layer_relevance: LoraLayerRelevance;
  module_list?: string[];
  training_params?: Record<string, unknown>;
  tag_frequency?: Record<string, Record<string, number>>;
  weight_stats?: Record<string, { avg_magnitude?: number; avg_strength?: number }>;
}

/** `POST /tools/lora/resize` result. */
export interface LoraResizeResult {
  old_rank?: number;
  new_rank?: number;
  modules_resized?: number;
  output_size_mb?: number;
}

@Component({
    selector: 'app-lora-tools',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [FormsModule],
    template: `
    <div class="space-y-4 animate-in fade-in duration-300">

        <!-- ═══════════════ INSPECT TAB ═══════════════ -->
        @if (tab() === 'inspect') {
            <div class="card animate-in fade-in duration-200" style="padding: 14px;">

                <!-- File Path Input -->
                <div class="flex gap-2" [class.mb-4]="inspectResult() || inspectError()">
                    <input type="text" [(ngModel)]="inspectPath"
                        placeholder="Path to .safetensors file…"
                        class="input mono" style="flex: 1;" />
                    <button (click)="inspectLora()" [disabled]="isInspecting() || !inspectPath"
                        class="btn primary">
                        @if (isInspecting()) {
                            <span class="flex items-center gap-2">
                                <span class="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                                Analyzing…
                            </span>
                        } @else {
                            Inspect
                        }
                    </button>
                </div>

                @if (inspectError()) {
                    <div class="bg-red-900/30 border border-red-800 text-red-300 text-sm rounded-theme-md p-3 mb-4">{{ inspectError() }}</div>
                }

                <!-- Results -->
                @if (inspectResult()) {
                    <div class="space-y-4 animate-in fade-in duration-300">

                        <!-- Quick Stats Row -->
                        <div class="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
                            @for (stat of quickStats(); track stat.label) {
                                <div class="kpi compact">
                                    <div class="kpi-label">{{ stat.label }}</div>
                                    <div class="kpi-value">{{ stat.value }}</div>
                                </div>
                            }
                        </div>


                        <!-- Layer Weight Analysis (lora-inspector style) -->
                        @if (inspectResult()?.layer_details?.length) {
                            <div class="bg-base/40 border border-border-default rounded-theme-md p-4">
                                <button (click)="showLayerAnalysis.set(!showLayerAnalysis())"
                                    class="w-full flex items-center justify-between text-xs font-bold uppercase tracking-widest text-text-subtle hover:text-text-secondary transition-colors">
                                    <span>Layer Weight Analysis ({{ inspectResult()?.layer_details?.length }} layers)</span>
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                                        [class.rotate-180]="showLayerAnalysis()" class="transition-transform">
                                        <polyline points="6 9 12 15 18 9"></polyline>
                                    </svg>
                                </button>
                                @if (showLayerAnalysis()) {
                                    @if (inspectResult()?.norm_summary) {
                                        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3 mb-4">
                                            <div class="bg-base/50 rounded-theme-sm p-2 text-center">
                                                <div class="text-[10px] text-text-subtle font-bold uppercase">Mean Norm</div>
                                                <div class="text-sm font-bold text-white">{{ inspectResult()!.norm_summary.mean_norm }}</div>
                                            </div>
                                            <div class="bg-base/50 rounded-theme-sm p-2 text-center">
                                                <div class="text-[10px] text-text-subtle font-bold uppercase">Std Dev</div>
                                                <div class="text-sm font-bold text-white">{{ inspectResult()!.norm_summary.std_norm }}</div>
                                            </div>
                                            <div class="bg-base/50 rounded-theme-sm p-2 text-center">
                                                <div class="text-[10px] text-text-subtle font-bold uppercase">🔥 Hottest</div>
                                                <div class="text-xs font-bold text-amber-400 truncate" [title]="inspectResult()!.norm_summary.max_norm_layer">
                                                    {{ inspectResult()!.norm_summary.max_norm }} <span class="text-text-subtle font-normal">{{ shortLayer(inspectResult()!.norm_summary.max_norm_layer) }}</span>
                                                </div>
                                            </div>
                                            <div class="bg-base/50 rounded-theme-sm p-2 text-center">
                                                <div class="text-[10px] text-text-subtle font-bold uppercase">🧊 Coldest</div>
                                                <div class="text-xs font-bold text-blue-400 truncate" [title]="inspectResult()!.norm_summary.min_norm_layer">
                                                    {{ inspectResult()!.norm_summary.min_norm }} <span class="text-text-subtle font-normal">{{ shortLayer(inspectResult()!.norm_summary.min_norm_layer) }}</span>
                                                </div>
                                            </div>
                                        </div>
                                    }

                                    <!-- Speed Training Recommendation -->
                                    @if (inspectResult()?.layer_relevance?.essential_count) {
                                        <div class="bg-gradient-to-r from-amber-900/20 to-amber-800/10 border border-amber-700/30 rounded-theme-md p-4 mb-3">
                                            <div class="flex items-start justify-between gap-4">
                                                <div>
                                                    <h4 class="text-xs font-bold text-amber-400 uppercase tracking-widest mb-1">&#9889; Speed Training Suggestion</h4>
                                                    <p class="text-sm text-white font-medium">
                                                        Train <span class="text-amber-400 font-bold">{{ inspectResult()!.layer_relevance.essential_count }}</span>
                                                        / {{ inspectResult()!.layer_relevance.total_layers }} layers for ~90% of learned effect
                                                    </p>
                                                    <p class="text-[10px] text-text-muted mt-1">
                                                        {{ inspectResult()!.layer_relevance.essential_params_pct }}% of parameters
                                                        @if (inspectResult()!.layer_relevance.speed_gain_pct > 0) {
                                                            &middot; est. <span class="text-success font-bold">{{ inspectResult()!.layer_relevance.speed_gain_pct }}% faster</span>
                                                        }
                                                    </p>
                                                </div>
                                                @if (inspectResult()!.layer_relevance.target_module_patterns.length) {
                                                    <button (click)="copyTargetModules()"
                                                        class="shrink-0 text-[10px] font-bold uppercase px-3 py-1.5 rounded-theme-md bg-amber-600/20 text-amber-400 border border-amber-600/30 hover:bg-amber-600/30 transition-colors">
                                                        {{ copiedModules() ? '&#10003; Copied' : 'Copy Modules' }}
                                                    </button>
                                                }
                                            </div>
                                            @if (inspectResult()!.layer_relevance.target_module_patterns.length) {
                                                <div class="mt-2 flex flex-wrap gap-1">
                                                    @for (mod of inspectResult()!.layer_relevance.target_module_patterns; track mod) {
                                                        <span class="text-[10px] font-mono bg-amber-900/30 text-amber-300/80 px-2 py-0.5 rounded-full border border-amber-700/20">{{ mod }}</span>
                                                    }
                                                </div>
                                            }
                                        </div>
                                    }

                                    <!-- Graphs (collapsible, one per block type) -->
                                    <div class="bg-base/30 border border-border-default/50 rounded-theme-md p-3 mb-3">
                                        <button (click)="toggleGraphs()"
                                            class="w-full flex items-center justify-between text-[10px] font-bold uppercase tracking-widest text-text-subtle hover:text-text-secondary transition-colors">
                                            <span>Graphs</span>
                                            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                                                [class.rotate-180]="showGraphs()" class="transition-transform">
                                                <polyline points="6 9 12 15 18 9"></polyline>
                                            </svg>
                                        </button>
                                        @if (showGraphs()) {
                                            <div class="mt-3 space-y-4">
                                                @for (block of blockTypes(); track block) {
                                                    <div class="bg-base/40 rounded-theme-sm p-3">
                                                        <h5 class="text-[10px] font-bold text-text-subtle uppercase tracking-widest mb-2">{{ block }}</h5>
                                                        <div class="norm-chart" [attr.data-block]="block"></div>
                                                    </div>
                                                }
                                            </div>
                                        }
                                    </div>

                                    <div class="flex items-center gap-2 mb-2">
                                        <span class="text-[10px] text-text-subtle font-bold uppercase">Sort by:</span>
                                        <button (click)="layerSortBy.set('norm')" [class.text-brand]="layerSortBy() === 'norm'"
                                            class="text-[10px] font-bold px-2 py-0.5 rounded hover:text-white transition-colors"
                                            [class.text-text-subtle]="layerSortBy() !== 'norm'">Norm</button>
                                        <button (click)="layerSortBy.set('strength')" [class.text-brand]="layerSortBy() === 'strength'"
                                            class="text-[10px] font-bold px-2 py-0.5 rounded hover:text-white transition-colors"
                                            [class.text-text-subtle]="layerSortBy() !== 'strength'">Strength</button>
                                        <button (click)="layerSortBy.set('module')" [class.text-brand]="layerSortBy() === 'module'"
                                            class="text-[10px] font-bold px-2 py-0.5 rounded hover:text-white transition-colors"
                                            [class.text-text-subtle]="layerSortBy() !== 'module'">Name</button>
                                    </div>
                                    <div class="max-h-96 overflow-y-auto space-y-1">
                                        @for (layer of sortedLayerDetails(); track layer.module) {
                                            <div class="group flex items-center gap-2 py-1 px-2 rounded hover:bg-surface-mid/50 transition-colors">
                                                <div class="w-5 text-center text-[9px]" [title]="getLayerTier(layer.module)">
                                                    {{ getLayerTierIcon(layer.module) }}
                                                </div>
                                                <div class="w-32 sm:w-44 text-[10px] font-mono text-text-muted truncate" [title]="layer.module">
                                                    {{ shortLayer(layer.module) }}
                                                </div>
                                                <div class="flex-1 h-3 bg-base/80 rounded-full overflow-hidden">
                                                    <div class="h-full rounded-full transition-all duration-300"
                                                        [style.width.%]="layer._barPct"
                                                        [style.background]="getLayerBarColor(layer.module, layer._barPct)">
                                                    </div>
                                                </div>
                                                <div class="w-16 text-right text-[10px] font-mono text-text-secondary">{{ layer.norm_delta }}</div>
                                                <div class="w-16 text-right text-[10px] font-mono text-text-subtle">{{ layer.strength }}</div>
                                            </div>
                                        }
                                    </div>
                                }
                            </div>
                        }

                        <!-- Module List (collapsible) -->
                        <div class="bg-base/40 border border-border-default rounded-theme-md p-4">
                            <button (click)="showModules.set(!showModules())"
                                class="w-full flex items-center justify-between text-xs font-bold uppercase tracking-widest text-text-subtle hover:text-text-secondary transition-colors">
                                <span>LoRA Modules ({{ inspectResult()?.lora_modules }})</span>
                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                                    [class.rotate-180]="showModules()" class="transition-transform">
                                    <polyline points="6 9 12 15 18 9"></polyline>
                                </svg>
                            </button>
                            @if (showModules()) {
                                <div class="mt-3 max-h-64 overflow-y-auto font-mono text-xs text-text-muted space-y-0.5">
                                    @for (mod of inspectResult()?.module_list || []; track mod) {
                                        <div class="py-0.5 px-2 rounded hover:bg-surface-mid/50 truncate">{{ mod }}</div>
                                    }
                                </div>
                            }
                        </div>

                        <!-- Training Params (collapsible) -->
                        @if (inspectResult()?.training_params && hasKeys(inspectResult()?.training_params)) {
                            <div class="bg-base/40 border border-border-default rounded-theme-md p-4">
                                <button (click)="showTrainingParams.set(!showTrainingParams())"
                                    class="w-full flex items-center justify-between text-xs font-bold uppercase tracking-widest text-text-subtle hover:text-text-secondary transition-colors">
                                    <span>Training Parameters</span>
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                                        [class.rotate-180]="showTrainingParams()" class="transition-transform">
                                        <polyline points="6 9 12 15 18 9"></polyline>
                                    </svg>
                                </button>
                                @if (showTrainingParams()) {
                                    <div class="mt-3 max-h-64 overflow-y-auto">
                                        <table class="w-full text-xs">
                                            @for (entry of trainingParamEntries(); track entry.key) {
                                                <tr class="border-b border-border-default/50">
                                                    <td class="py-1.5 pr-4 text-text-subtle font-mono whitespace-nowrap">{{ entry.key }}</td>
                                                    <td class="py-1.5 text-text-secondary font-mono break-all">{{ entry.value }}</td>
                                                </tr>
                                            }
                                        </table>
                                    </div>
                                }
                            </div>
                        }

                        <!-- Tag Frequency (collapsible) -->
                        @if (inspectResult()?.tag_frequency && hasKeys(inspectResult()?.tag_frequency)) {
                            <div class="bg-base/40 border border-border-default rounded-theme-md p-4">
                                <button (click)="showTags.set(!showTags())"
                                    class="w-full flex items-center justify-between text-xs font-bold uppercase tracking-widest text-text-subtle hover:text-text-secondary transition-colors">
                                    <span>Tag Frequency</span>
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                                        [class.rotate-180]="showTags()" class="transition-transform">
                                        <polyline points="6 9 12 15 18 9"></polyline>
                                    </svg>
                                </button>
                                @if (showTags()) {
                                    <div class="mt-3 max-h-64 overflow-y-auto">
                                        @for (group of tagGroups(); track group.name) {
                                            <div class="mb-3">
                                                <div class="text-[10px] text-text-subtle font-bold uppercase mb-1">{{ group.name }}</div>
                                                <div class="flex flex-wrap gap-1.5">
                                                    @for (tag of group.tags; track tag.name) {
                                                        <span class="inline-flex items-center gap-1 bg-surface-mid text-text-secondary text-[10px] px-2 py-0.5 rounded-full">
                                                            {{ tag.name }}
                                                            <span class="text-brand-light font-bold">{{ tag.count }}</span>
                                                        </span>
                                                    }
                                                </div>
                                            </div>
                                        }
                                    </div>
                                }
                            </div>
                        }
                    </div>
                }
            </div>
        }

        <!-- ═══════════════ RESIZE TAB ═══════════════ -->
        @if (tab() === 'resize') {
            <div class="card animate-in fade-in duration-200" style="padding: 14px;">

                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
                    <div>
                        <label class="field-label">Input Path</label>
                        <input type="text" [(ngModel)]="resizeInputPath"
                            placeholder="Path to source .safetensors…"
                            class="input mono" />
                    </div>
                    <div>
                        <label class="field-label">Output Path</label>
                        <input type="text" [(ngModel)]="resizeOutputPath"
                            placeholder="Path for resized output…"
                            class="input mono" />
                    </div>
                    <div>
                        <label class="field-label">New Rank</label>
                        <input type="number" [(ngModel)]="resizeNewRank" [min]="1" [max]="256"
                            class="input mono" />
                    </div>
                    <div>
                        <label class="field-label">New Alpha <span class="text-text-disabled">(optional)</span></label>
                        <input type="number" [(ngModel)]="resizeNewAlpha" step="0.1"
                            placeholder="Auto-scaled if empty"
                            class="input mono" />
                    </div>
                    <div>
                        <label class="field-label">Save Dtype <span class="text-text-disabled">(optional)</span></label>
                        <select [(ngModel)]="resizeDtype"
                            class="select">
                            <option value="">Preserve Original</option>
                            <option value="fp16">FP16</option>
                            <option value="bf16">BF16</option>
                            <option value="fp32">FP32</option>
                        </select>
                    </div>
                </div>

                <div class="flex items-center gap-4">
                    <button (click)="resizeLora()" [disabled]="isResizing() || !resizeInputPath || !resizeOutputPath || !resizeNewRank"
                        class="btn primary">
                        @if (isResizing()) {
                            <span class="flex items-center gap-2">
                                <span class="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                                Resizing…
                            </span>
                        } @else {
                            Resize via SVD
                        }
                    </button>

                    @if (resizeError()) {
                        <div class="text-sm text-red-400">{{ resizeError() }}</div>
                    }
                </div>

                @if (resizeResult()) {
                    <div class="mt-4 bg-green-900/20 border border-green-800/50 rounded-theme-md p-4 animate-in fade-in duration-300">
                        <div class="text-sm text-green-400 font-medium mb-2">✓ Resize Complete</div>
                        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
                            <div class="text-center">
                                <div class="text-[10px] text-text-subtle font-bold uppercase">Old Rank</div>
                                <div class="text-sm font-bold text-white">{{ resizeResult()?.old_rank }}</div>
                            </div>
                            <div class="text-center">
                                <div class="text-[10px] text-text-subtle font-bold uppercase">New Rank</div>
                                <div class="text-sm font-bold text-brand-light">{{ resizeResult()?.new_rank }}</div>
                            </div>
                            <div class="text-center">
                                <div class="text-[10px] text-text-subtle font-bold uppercase">Modules</div>
                                <div class="text-sm font-bold text-white">{{ resizeResult()?.modules_resized }}</div>
                            </div>
                            <div class="text-center">
                                <div class="text-[10px] text-text-subtle font-bold uppercase">Size</div>
                                <div class="text-sm font-bold text-white">{{ resizeResult()?.output_size_mb }} MB</div>
                            </div>
                        </div>
                    </div>
                }
            </div>
        }
    </div>
  `
})
export class LoraToolsComponent implements OnDestroy {
    private http = inject(HttpClient);
    private rtc = inject(RuntimeConfigService);

    /** Which flow to show — driven by the Tools screen's outer tabs. */
    readonly tab = input<ToolTab>('inspect');

    // ── Inspect State ──
    inspectPath = '';
    isInspecting = signal<boolean>(false);
    inspectResult = signal<LoraInspectResult | null>(null);
    inspectError = signal<string | null>(null);
    showModules = signal<boolean>(false);
    showTrainingParams = signal<boolean>(false);
    showTags = signal<boolean>(false);
    showLayerAnalysis = signal<boolean>(false);
    layerSortBy = signal<'norm' | 'strength' | 'module'>('norm');
    showGraphs = signal<boolean>(false);
    copiedModules = signal<boolean>(false);
    private normPlots: uPlot[] = [];

    // ── Resize State ──
    resizeInputPath = '';
    resizeOutputPath = '';
    resizeNewRank = 16;
    resizeNewAlpha: number | null = null;
    resizeDtype = '';
    isResizing = signal<boolean>(false);
    resizeResult = signal<LoraResizeResult | null>(null);
    resizeError = signal<string | null>(null);

    quickStats = signal<{ label: string; value: string }[]>([]);
    weightStatEntries = signal<{ component: string; avg_magnitude: string; avg_strength: string }[]>([]);
    trainingParamEntries = signal<{ key: string; value: string }[]>([]);
    tagGroups = signal<{ name: string; tags: { name: string; count: number }[] }[]>([]);

    inspectLora() {
        if (!this.inspectPath) return;
        this.inspectPath = this.inspectPath.replace(/^"|"$/g, '').trim();
        this.isInspecting.set(true);
        this.inspectError.set(null);
        this.inspectResult.set(null);

        this.http.get<LoraInspectResult>(`${this.rtc.apiUrl}/tools/lora/inspect`, {
            params: { path: this.inspectPath }
        }).subscribe({
            next: (result) => {
                this.inspectResult.set(result);
                this.isInspecting.set(false);
                this.buildQuickStats(result);
                this.buildWeightStats(result);
                this.buildTrainingParams(result);
                this.buildTagGroups(result);

                // Pre-fill resize input from inspect
                this.resizeInputPath = result.path || this.inspectPath;
                // Suggest output path
                const dotIdx = this.resizeInputPath.lastIndexOf('.');
                this.resizeOutputPath = dotIdx > 0
                    ? this.resizeInputPath.substring(0, dotIdx) + '_resized.safetensors'
                    : this.resizeInputPath + '_resized';
            },
            error: (err) => {
                this.isInspecting.set(false);
                this.inspectError.set(err.error?.detail || err.message || 'Inspection failed');
            }
        });
    }

    resizeLora() {
        if (!this.resizeInputPath || !this.resizeOutputPath || !this.resizeNewRank) return;
        this.resizeInputPath = this.resizeInputPath.replace(/^"|"$/g, '').trim();
        this.resizeOutputPath = this.resizeOutputPath.replace(/^"|"$/g, '').trim();
        this.isResizing.set(true);
        this.resizeError.set(null);
        this.resizeResult.set(null);

        const body: Record<string, unknown> = {
            input_path: this.resizeInputPath,
            output_path: this.resizeOutputPath,
            new_rank: this.resizeNewRank,
        };
        if (this.resizeNewAlpha != null) body['new_alpha'] = this.resizeNewAlpha;
        if (this.resizeDtype) body['save_dtype'] = this.resizeDtype;

        this.http.post<LoraResizeResult>(`${this.rtc.apiUrl}/tools/lora/resize`, body).subscribe({
            next: (result) => {
                this.resizeResult.set(result);
                this.isResizing.set(false);
            },
            error: (err) => {
                this.isResizing.set(false);
                this.resizeError.set(err.error?.detail || err.message || 'Resize failed');
            }
        });
    }

    hasKeys(obj: Record<string, unknown> | undefined): boolean {
        return !!obj && Object.keys(obj).length > 0;
    }

    private buildQuickStats(r: LoraInspectResult) {
        this.quickStats.set([
            { label: 'Format', value: r.format || '—' },
            { label: 'Rank', value: r.rank != null ? String(r.rank) : '—' },
            { label: 'Alpha', value: r.alpha != null ? String(r.alpha) : '—' },
            { label: 'Modules', value: String(r.lora_modules || 0) },
            { label: 'Dtype', value: (r.dtype || '—').replace('torch.', '') },
            { label: 'Size', value: r.file_size_mb + ' MB' },
        ]);
    }

    private buildWeightStats(r: LoraInspectResult) {
        if (!r.weight_stats) { this.weightStatEntries.set([]); return; }
        const entries = Object.entries(r.weight_stats).map(([component, stats]) => ({
            component: component.replace(/_/g, ' '),
            avg_magnitude: stats.avg_magnitude != null ? stats.avg_magnitude.toFixed(6) : '—',
            avg_strength: stats.avg_strength != null ? stats.avg_strength.toFixed(6) : '—',
        }));
        this.weightStatEntries.set(entries);
    }

    private buildTrainingParams(r: LoraInspectResult) {
        if (!r.training_params) { this.trainingParamEntries.set([]); return; }
        const entries = Object.entries(r.training_params)
            .filter(([_, v]) => v != null && v !== '')
            .map(([key, value]) => ({
                key: key.replace(/^ss_/, ''),
                value: typeof value === 'object' ? JSON.stringify(value) : String(value),
            }));
        this.trainingParamEntries.set(entries);
    }

    private buildTagGroups(r: LoraInspectResult) {
        if (!r.tag_frequency) { this.tagGroups.set([]); return; }
        const groups = Object.entries(r.tag_frequency).map(([name, tags]) => ({
            name,
            tags: Object.entries(tags)
                .map(([tagName, count]) => ({ name: tagName, count: count as number }))
                .sort((a, b) => b.count - a.count),
        }));
        this.tagGroups.set(groups);
    }

    /** Shorten a full module path to its last 2-3 segments for display. */
    shortLayer(name: string): string {
        if (!name) return '';
        const parts = name.split('.');
        return parts.length > 3 ? '…' + parts.slice(-3).join('.') : name;
    }

    /** Get layer details sorted by current sort mode, with bar percentages pre-computed. */
    sortedLayerDetails(): LoraLayerDetail[] {
        const details = this.inspectResult()?.layer_details;
        if (!details?.length) return [];

        const sorted = [...details];
        const sortBy = this.layerSortBy();
        if (sortBy === 'norm') {
            sorted.sort((a, b) => b.norm_delta - a.norm_delta);
        } else if (sortBy === 'strength') {
            sorted.sort((a, b) => b.strength - a.strength);
        } else {
            sorted.sort((a, b) => a.module.localeCompare(b.module));
        }

        // Compute bar percentages relative to the max value
        const maxNorm = Math.max(...sorted.map((l) => l.norm_delta || 0), 1e-10);
        for (const layer of sorted) {
            layer._barPct = Math.round(((layer.norm_delta || 0) / maxNorm) * 100);
        }
        return sorted;
    }

    /** Unique block types (components) from layer_details. */
    blockTypes(): string[] {
        const details = this.inspectResult()?.layer_details;
        if (!details?.length) return [];
        return [...new Set<string>(details.map((l) => l.component))] as string[];
    }

    /** Toggle graphs and build charts when opening. */
    toggleGraphs() {
        const next = !this.showGraphs();
        this.showGraphs.set(next);
        if (next) {
            setTimeout(() => this.buildNormCharts(), 50);
        } else {
            this.destroyCharts();
        }
    }

    ngOnDestroy() {
        this.destroyCharts();
    }

    private destroyCharts() {
        this.normPlots.forEach(p => p.destroy());
        this.normPlots = [];
    }

    /** Build one uPlot chart per block type showing layer norms. */
    private buildNormCharts() {
        this.destroyCharts();

        const details = this.inspectResult()?.layer_details;
        if (!details?.length) return;

        const blocks = this.blockTypes();

        for (const block of blocks) {
            const container = document.querySelector(`.norm-chart[data-block="${block}"]`) as HTMLElement;
            if (!container) continue;
            container.innerHTML = '';

            // Filter layers for this block, sorted by module name for consistent ordering
            const layers = details
                .filter((l) => l.component === block)
                .sort((a, b) => a.module.localeCompare(b.module));
            if (!layers.length) continue;

            const indices = new Float64Array(layers.map((_, i) => i));
            const norms = new Float64Array(layers.map((l) => l.norm_delta || 0));
            const labels: string[] = layers.map((l) => this.shortLayer(l.module));

            const width = container.clientWidth || 500;
            const brandRGB = '255, 51, 102'; // brand red (matches harmonization chart)

            const splineFn = (uPlot as any).paths?.spline;
            const splineBuilder = typeof splineFn === 'function' ? splineFn() : undefined;

            const seriesConfig: uPlot.Series = {
                label: 'Norm',
                stroke: `rgba(${brandRGB}, 0.9)`,
                width: 2,
                fill: (u: uPlot) => {
                    const top = u.bbox.top / devicePixelRatio;
                    const bot = (u.bbox.top + u.bbox.height) / devicePixelRatio;
                    if (!isFinite(top) || !isFinite(bot) || top === bot) {
                        return `rgba(${brandRGB}, 0.15)`;
                    }
                    const gradient = u.ctx.createLinearGradient(0, top, 0, bot);
                    gradient.addColorStop(0, `rgba(${brandRGB}, 0.35)`);
                    gradient.addColorStop(1, `rgba(${brandRGB}, 0.02)`);
                    return gradient;
                },
                points: {
                    show: layers.length <= 50,
                    size: 4,
                    fill: `rgba(${brandRGB}, 1)`,
                    stroke: `rgba(${brandRGB}, 1)`,
                },
            };
            if (splineBuilder) seriesConfig.paths = splineBuilder;

            const opts: uPlot.Options = {
                width,
                height: 120,
                cursor: { show: true, drag: { x: false, y: false } },
                legend: { show: false },
                scales: {
                    x: { time: false },
                    y: { auto: true, range: (_u: uPlot, _min: number, max: number) => [0, Math.max(max * 1.25, 0.001)] as [number, number] },
                },
                axes: [
                    {
                        stroke: '#4b5563',
                        grid: { show: false },
                        ticks: { show: false },
                        font: '9px Inter, sans-serif',
                        gap: 4,
                        values: (_u: uPlot, vals: number[]) => vals.map(v => {
                            const idx = Math.round(v);
                            return labels[idx] || '';
                        }),
                    },
                    {
                        stroke: '#4b5563',
                        grid: { stroke: 'rgba(75, 85, 99, 0.15)', show: true },
                        ticks: { show: false },
                        font: '9px Inter, sans-serif',
                        size: 45,
                        gap: 4,
                        values: (_u: uPlot, vals: number[]) => vals.map(v =>
                            v < 0.001 ? v.toExponential(1) : v.toFixed(3)
                        ),
                    },
                ],
                series: [
                    { label: 'Layer' },
                    seriesConfig,
                ],
            };

            try {
                const plot = new uPlot(opts, [indices, norms], container);
                this.normPlots.push(plot);
            } catch (e) {
                console.error('[CHART] norm chart error:', e);
            }
        }
    }

    /** Copy essential layer paths to clipboard for targeted layer import. */
    copyTargetModules() {
        // Use full module paths (per-instance) so targeted layers can select
        // exact block instances (e.g. 60/112) instead of all instances of
        // each module type (320/448).
        const modules = this.inspectResult()?.layer_relevance?.essential_modules;
        if (!modules?.length) return;
        const text = JSON.stringify(modules);
        navigator.clipboard.writeText(text).then(() => {
            this.copiedModules.set(true);
            setTimeout(() => this.copiedModules.set(false), 2000);
        });
    }

    /** Get tier name for a module. */
    getLayerTier(module: string): string {
        const tier = this.inspectResult()?.layer_relevance?.tier_map?.[module];
        if (tier === 'essential') return 'Essential (≤90% energy)';
        if (tier === 'contributing') return 'Contributing (90–97%)';
        if (tier === 'negligible') return 'Negligible (<3%)';
        return '';
    }

    /** Get tier icon emoji for a module. */
    getLayerTierIcon(module: string): string {
        const tier = this.inspectResult()?.layer_relevance?.tier_map?.[module];
        if (tier === 'essential') return '🔥';
        if (tier === 'contributing') return '⚡';
        if (tier === 'negligible') return '🧊';
        return '·';
    }

    /** Get bar color based on tier and percentage. */
    getLayerBarColor(module: string, barPct?: number): string {
        const tier = this.inspectResult()?.layer_relevance?.tier_map?.[module];
        if (tier === 'essential') return '#f59e0b';  // amber
        if (tier === 'contributing') return '#8b5cf6'; // purple
        return '#3b82f6'; // blue (negligible)
    }
}
