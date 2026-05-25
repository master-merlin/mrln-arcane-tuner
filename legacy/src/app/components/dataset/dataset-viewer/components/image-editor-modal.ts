import { Component, ViewChild, HostListener, ElementRef, OnInit, OnDestroy, computed, signal, inject, input, output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { DatasetService, CurvePoint, CurvesConfig, ImageAdjustments, HistogramData, PipelineBlock } from '../../../../services/dataset';
import { ToastService } from '../../../../services/toast';
import { RuntimeConfigService } from '../../../../services/runtime-config.service';
import { OverlayStore } from '../../../../state/overlay.store';
import { CurvesEditorComponent } from './curves-editor';
import { HistogramDisplayComponent } from './histogram-display';
import { HSLPanelComponent, HSLConfig } from './hsl-panel';

type ChannelKey = 'master' | 'r' | 'g' | 'b';

interface LoadedLut {
    id: number;
    name: string;
    content: string;
    strength: number;
    parsed: { size: number; table: Float32Array };
}

const IDENTITY_CURVE: CurvePoint[] = [{ x: 0, y: 0 }, { x: 255, y: 255 }];

@Component({
    selector: 'app-image-editor-modal',
    standalone: true,
    imports: [FormsModule, CurvesEditorComponent, HistogramDisplayComponent, HSLPanelComponent],
    template: `
    <div class="fixed inset-0 z-50 flex flex-col bg-base/98 backdrop-blur-sm" data-testid="image-editor-modal">
        <!-- Toolbar -->
        <div class="h-12 flex items-center justify-between px-6 border-b border-surface-mid bg-surface-low/50">
            <div class="flex items-center gap-3">
                <button (click)="close.emit()" class="text-text-muted hover:text-text p-2 rounded-theme-lg hover:bg-surface-mid/50 transition-all" data-testid="editor-close">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                </button>
                <h2 class="text-sm font-semibold text-text">Image Adjustments</h2>
                @if (isDirty()) {
                    <span class="text-[10px] px-2 py-0.5 rounded-full bg-warning/20 text-warning font-medium">Modified</span>
                }
            </div>
            <div class="flex items-center gap-2">
                <button (click)="resetAll()" [disabled]="!isDirty()"
                    class="px-3 py-1.5 text-xs rounded-theme-lg bg-surface-mid/50 hover:bg-surface-mid text-text-muted hover:text-text transition-all disabled:opacity-30"
                    data-testid="editor-reset">
                    ⟲ Reset
                </button>
                @if (hasOverlay()) {
                    <button (click)="revertOverlay()" class="px-3 py-1.5 text-xs rounded-theme-lg bg-red-600/60 hover:bg-red-600 text-white transition-all" data-testid="editor-revert-overlay">
                        ↩ Revert Overlay
                    </button>
                    <button (click)="showCommitConfirm.set(true)" class="px-3 py-1.5 text-xs rounded-theme-lg bg-amber-600/60 hover:bg-amber-600 text-white transition-all" data-testid="editor-commit-overlay">
                        ⚠ Commit to Original
                    </button>
                }
                <button (click)="renderOverlay()" [disabled]="!isDirty() || isRendering()"
                    class="px-4 py-1.5 text-xs font-semibold rounded-theme-lg bg-brand hover:bg-brand-hover text-white transition-all disabled:opacity-30 flex items-center gap-2"
                    data-testid="editor-render-overlay">
                    @if (isRendering()) {
                        <svg class="animate-spin h-3 w-3" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
                        Rendering...
                    } @else {
                        Save as Overlay
                    }
                </button>
            </div>
        </div>

        <!-- Main Content -->
        <div class="flex-1 flex min-h-0 overflow-hidden">
            <!-- Left: Controls Panel -->
            <div class="w-80 flex-shrink-0 border-r border-surface-mid bg-surface-mid overflow-y-auto p-4 flex flex-col gap-1">

                <!-- Curves Section (collapsible) -->
                <button (click)="curvesOpen.set(!curvesOpen())" class="flex items-center justify-between w-full py-2 text-xs font-bold text-text-subtle uppercase tracking-widest hover:text-text transition-all" data-testid="editor-section-curves">
                    <span class="flex items-center gap-2">Curves
                        @if (curvesDirty()) { <span (click)="resetCurves(); $event.stopPropagation()" class="text-[10px] text-text-muted/60 hover:text-warning cursor-pointer" title="Reset Curves">↻</span> }
                    </span>
                    <svg [class]="'w-3.5 h-3.5 transition-transform ' + (curvesOpen() ? 'rotate-180' : '')" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                </button>
                @if (curvesOpen()) {
                    <div class="pb-3">
                        <app-curves-editor
                            [masterCurve]="masterCurve()"
                            [rCurve]="rCurve()"
                            [gCurve]="gCurve()"
                            [bCurve]="bCurve()"
                            [histogramData]="liveHistogram()"
                            (curveChanged)="onCurveChanged($event)"
                            data-testid="editor-curves">
                        </app-curves-editor>
                    </div>
                }

                <!-- Pipeline Overview (drag-reorder) -->
                <button (click)="pipelineOverviewOpen.set(!pipelineOverviewOpen())" class="flex items-center justify-between w-full py-2 text-xs font-bold text-brand uppercase tracking-widest hover:text-brand-hover transition-all border-t border-surface-high/20" data-testid="editor-section-pipeline">
                    <span>⚡ Pipeline Order</span>
                    <svg [class]="'w-3.5 h-3.5 transition-transform ' + (pipelineOverviewOpen() ? 'rotate-180' : '')" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                </button>
                @if (pipelineOverviewOpen()) {
                    <div class="pb-3 flex flex-col gap-0.5" data-testid="pipeline-block-list">
                        @for (blockType of pipelineOrder(); track blockType; let i = $index) {
                            <div class="flex items-center gap-1.5 px-2 py-1 rounded-theme-sm text-[10px] transition-all cursor-grab active:cursor-grabbing"
                                [class]="blockEnabled()[blockType] !== false ? 'bg-surface-low/60 hover:bg-surface-low text-text' : 'bg-surface-low/20 text-text-muted/50'"
                                draggable="true"
                                (dragstart)="onBlockDragStart($event, i)"
                                (dragover)="onBlockDragOver($event, i)"
                                (drop)="onBlockDrop($event, i)"
                                [attr.data-testid]="'pipeline-block-' + blockType">
                                <span class="text-text-muted/40 select-none">☰</span>
                                <button (click)="toggleBlockEnabled(blockType); $event.stopPropagation()"
                                    class="w-3.5 h-3.5 flex items-center justify-center rounded-sm border transition-all"
                                    [class]="blockEnabled()[blockType] !== false ? 'border-brand/60 bg-brand/20 text-brand' : 'border-surface-high/30 text-transparent'"
                                    [attr.data-testid]="'pipeline-toggle-' + blockType">
                                    <svg class="w-2.5 h-2.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
                                </button>
                                <span class="flex-1 font-medium truncate" [class.line-through]="blockEnabled()[blockType] === false">
                                    {{ BLOCK_LABELS[blockType] || blockType }}
                                </span>
                                <span class="font-mono text-[9px]"
                                    [class]="getBlockSummary(blockType) !== 'off' ? 'text-brand/80' : 'text-text-muted/30'">
                                    {{ getBlockSummary(blockType) }}
                                </span>
                            </div>
                        }
                    </div>
                }

                <!-- CUBE LUT Section (collapsible) -->
                <button (click)="lutOpen.set(!lutOpen())" class="flex items-center justify-between w-full py-2 text-xs font-bold text-text-subtle uppercase tracking-widest hover:text-text transition-all border-t border-surface-high/20" data-testid="editor-section-lut">
                    <span class="flex items-center gap-2">CUBE LUT
                        @if (lutDirty()) { <span (click)="resetLut(); $event.stopPropagation()" class="text-[10px] text-text-muted/60 hover:text-warning cursor-pointer" title="Reset LUT">↻</span> }
                    </span>
                    <svg [class]="'w-3.5 h-3.5 transition-transform ' + (lutOpen() ? 'rotate-180' : '')" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                </button>
                @if (lutOpen()) {
                    <div class="pb-3 flex flex-col gap-2">
                        @for (lut of lutStack(); track lut.id) {
                            <div class="flex flex-col gap-1.5 p-2 rounded-theme-lg bg-brand/5 border border-brand/15">
                                <div class="flex items-center justify-between">
                                    <span class="text-[11px] text-brand font-medium truncate" [title]="lut.name">{{ lut.name }}</span>
                                    <button (click)="removeLut(lut.id)" class="text-[10px] text-danger hover:text-danger-hover flex-shrink-0 ml-2" [attr.data-testid]="'editor-remove-lut-' + lut.id">✕</button>
                                </div>
                                <div class="flex items-center gap-2">
                                    <input type="range" min="0" max="1" step="0.01" [ngModel]="lut.strength" (ngModelChange)="updateLutStrength(lut.id, +$event)"
                                        class="flex-1 accent-brand" [attr.data-testid]="'editor-lut-strength-' + lut.id">
                                    <span class="text-[10px] font-mono text-text-muted w-8 text-right">{{ (lut.strength * 100).toFixed(0) }}%</span>
                                </div>
                            </div>
                        }
                        <div class="flex gap-1">
                            <label class="flex-1 px-2 py-1.5 text-[10px] text-center rounded-theme-sm bg-surface-mid/40 hover:bg-surface-mid/80 text-text-muted hover:text-text cursor-pointer transition-all"
                                data-testid="editor-import-cube">
                                + Import .cube
                                <input type="file" accept=".cube" class="hidden" (change)="onCubeFileSelected($event)">
                            </label>
                            <button (click)="exportCubeLut()" class="flex-1 px-2 py-1.5 text-[10px] rounded-theme-sm bg-surface-mid/40 hover:bg-surface-mid/80 text-text-muted hover:text-text transition-all"
                                data-testid="editor-export-cube">
                                Export .cube
                            </button>
                        </div>
                    </div>
                }

                <!-- Color & Tone (collapsible) -->
                <button (click)="colorOpen.set(!colorOpen())" class="flex items-center justify-between w-full py-2 text-xs font-bold text-text-subtle uppercase tracking-widest hover:text-text transition-all border-t border-surface-high/20" data-testid="editor-section-color">
                    <span class="flex items-center gap-2">Color & Tone
                        @if (colorDirty()) { <span (click)="resetColor(); $event.stopPropagation()" class="text-[10px] text-text-muted/60 hover:text-warning cursor-pointer" title="Reset Color">↻</span> }
                    </span>
                    <svg [class]="'w-3.5 h-3.5 transition-transform ' + (colorOpen() ? 'rotate-180' : '')" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                </button>
                @if (colorOpen()) {
                    <div class="pb-3 flex flex-col gap-3">
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-[11px] text-text-muted">
                                <span>Hue</span><span class="font-mono">{{ hueShift() }}°</span>
                            </div>
                            <input type="range" min="-180" max="180" step="1" [ngModel]="hueShift()" (ngModelChange)="hueShift.set($event)"
                                class="w-full accent-brand" data-testid="editor-hue">
                        </div>

                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-[11px] text-text-muted">
                                <span>Saturation</span><span class="font-mono">{{ saturation().toFixed(2) }}</span>
                            </div>
                            <input type="range" min="0" max="3" step="0.01" [ngModel]="saturation()" (ngModelChange)="saturation.set(+$event)"
                                class="w-full accent-brand" data-testid="editor-saturation">
                        </div>

                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-[11px] text-text-muted">
                                <span>Contrast</span><span class="font-mono">{{ contrast().toFixed(2) }}</span>
                            </div>
                            <input type="range" min="0" max="3" step="0.01" [ngModel]="contrast()" (ngModelChange)="contrast.set(+$event)"
                                class="w-full accent-brand" data-testid="editor-contrast">
                        </div>
                    </div>
                }

                <!-- Sharpening (collapsible) -->
                <button (click)="sharpenOpen.set(!sharpenOpen())" class="flex items-center justify-between w-full py-2 text-xs font-bold text-text-subtle uppercase tracking-widest hover:text-text transition-all border-t border-surface-high/20" data-testid="editor-section-sharpen">
                    <span class="flex items-center gap-2">Sharpening
                        @if (sharpenDirty()) { <span (click)="resetSharpen(); $event.stopPropagation()" class="text-[10px] text-text-muted/60 hover:text-warning cursor-pointer" title="Reset Sharpening">↻</span> }
                    </span>
                    <svg [class]="'w-3.5 h-3.5 transition-transform ' + (sharpenOpen() ? 'rotate-180' : '')" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                </button>
                @if (sharpenOpen()) {
                    <div class="pb-3 flex flex-col gap-3">
                        <select [ngModel]="sharpenMethod()" (ngModelChange)="sharpenMethod.set($event)"
                            class="w-full bg-surface-low border border-surface-high/30 rounded-theme-lg px-2 py-1.5 text-xs text-text"
                            data-testid="editor-sharpen-method">
                            <option value="none">None</option>
                            <option value="unsharp_mask">Unsharp Mask</option>
                            <option value="kernel">Kernel</option>
                            <option value="high_pass">High Pass</option>
                        </select>

                        @if (sharpenMethod() === 'unsharp_mask') {
                            <div class="flex flex-col gap-2">
                                <div class="flex flex-col gap-1">
                                    <div class="flex justify-between text-[11px] text-text-muted"><span>Radius</span><span class="font-mono">{{ sharpenRadius().toFixed(1) }}</span></div>
                                    <input type="range" min="0.1" max="10" step="0.1" [ngModel]="sharpenRadius()" (ngModelChange)="sharpenRadius.set(+$event)" class="w-full accent-brand" data-testid="editor-sharpen-radius">
                                </div>
                                <div class="flex flex-col gap-1">
                                    <div class="flex justify-between text-[11px] text-text-muted"><span>Amount</span><span class="font-mono">{{ sharpenPercent() }}%</span></div>
                                    <input type="range" min="0" max="500" step="1" [ngModel]="sharpenPercent()" (ngModelChange)="sharpenPercent.set(+$event)" class="w-full accent-brand" data-testid="editor-sharpen-percent">
                                </div>
                                <div class="flex flex-col gap-1">
                                    <div class="flex justify-between text-[11px] text-text-muted"><span>Threshold</span><span class="font-mono">{{ sharpenThreshold() }}</span></div>
                                    <input type="range" min="0" max="20" step="1" [ngModel]="sharpenThreshold()" (ngModelChange)="sharpenThreshold.set(+$event)" class="w-full accent-brand" data-testid="editor-sharpen-threshold">
                                </div>
                            </div>
                        }
                        @if (sharpenMethod() === 'kernel') {
                            <div class="flex flex-col gap-1">
                                <div class="flex justify-between text-[11px] text-text-muted"><span>Strength</span><span class="font-mono">{{ sharpenStrength().toFixed(2) }}</span></div>
                                <input type="range" min="0" max="2" step="0.01" [ngModel]="sharpenStrength()" (ngModelChange)="sharpenStrength.set(+$event)" class="w-full accent-brand" data-testid="editor-sharpen-strength">
                            </div>
                        }
                        @if (sharpenMethod() === 'high_pass') {
                            <div class="flex flex-col gap-2">
                                <div class="flex flex-col gap-1">
                                    <div class="flex justify-between text-[11px] text-text-muted"><span>Radius</span><span class="font-mono">{{ sharpenRadius().toFixed(1) }}</span></div>
                                    <input type="range" min="0.5" max="20" step="0.5" [ngModel]="sharpenRadius()" (ngModelChange)="sharpenRadius.set(+$event)" class="w-full accent-brand" data-testid="editor-hp-radius">
                                </div>
                                <div class="flex flex-col gap-1">
                                    <div class="flex justify-between text-[11px] text-text-muted"><span>Strength</span><span class="font-mono">{{ sharpenStrength().toFixed(2) }}</span></div>
                                    <input type="range" min="0" max="2" step="0.01" [ngModel]="sharpenStrength()" (ngModelChange)="sharpenStrength.set(+$event)" class="w-full accent-brand" data-testid="editor-hp-strength">
                                </div>
                            </div>
                        }
                    </div>
                }
                <!-- White Balance (collapsible) -->
                <button (click)="wbOpen.set(!wbOpen())" class="flex items-center justify-between w-full py-2 text-xs font-bold text-text-subtle uppercase tracking-widest hover:text-text transition-all border-t border-surface-high/20" data-testid="editor-section-wb">
                    <span class="flex items-center gap-2">White Balance
                        @if (wbDirty()) { <span (click)="resetWb(); $event.stopPropagation()" class="text-[10px] text-text-muted/60 hover:text-warning cursor-pointer" title="Reset WB">↻</span> }
                    </span>
                    <svg [class]="'w-3.5 h-3.5 transition-transform ' + (wbOpen() ? 'rotate-180' : '')" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                </button>
                @if (wbOpen()) {
                    <div class="pb-3 flex flex-col gap-3">
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-[11px] text-text-muted">
                                <span>Temperature</span><span class="font-mono">{{ wbTemperature() }}K</span>
                            </div>
                            <input type="range" min="2000" max="12000" step="100" [ngModel]="wbTemperature()" (ngModelChange)="wbTemperature.set(+$event)"
                                class="w-full accent-brand" data-testid="editor-wb-temp">
                        </div>
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-[11px] text-text-muted">
                                <span>Tint</span><span class="font-mono">{{ wbTint() }}</span>
                            </div>
                            <input type="range" min="-100" max="100" step="1" [ngModel]="wbTint()" (ngModelChange)="wbTint.set(+$event)"
                                class="w-full accent-brand" data-testid="editor-wb-tint">
                        </div>
                    </div>
                }

                <!-- HSL / Selective Color (collapsible) -->
                <button (click)="hslOpen.set(!hslOpen())" class="flex items-center justify-between w-full py-2 text-xs font-bold text-text-subtle uppercase tracking-widest hover:text-text transition-all border-t border-surface-high/20" data-testid="editor-section-hsl">
                    <span class="flex items-center gap-2">HSL / Selective Color
                        @if (hslDirty()) { <span (click)="resetHsl(); $event.stopPropagation()" class="text-[10px] text-text-muted/60 hover:text-warning cursor-pointer" title="Reset HSL">↻</span> }
                    </span>
                    <svg [class]="'w-3.5 h-3.5 transition-transform ' + (hslOpen() ? 'rotate-180' : '')" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                </button>
                @if (hslOpen()) {
                    <div class="pb-3">
                        <app-hsl-panel
                            [hslConfig]="hslConfig()"
                            (hslChanged)="hslConfig.set($event)"
                            data-testid="editor-hsl-panel">
                        </app-hsl-panel>
                    </div>
                }

                <!-- Vignette (collapsible) -->
                <button (click)="vignetteOpen.set(!vignetteOpen())" class="flex items-center justify-between w-full py-2 text-xs font-bold text-text-subtle uppercase tracking-widest hover:text-text transition-all border-t border-surface-high/20" data-testid="editor-section-vignette">
                    <span class="flex items-center gap-2">Vignette
                        @if (vignetteDirty()) { <span (click)="resetVignette(); $event.stopPropagation()" class="text-[10px] text-text-muted/60 hover:text-warning cursor-pointer" title="Reset Vignette">↻</span> }
                    </span>
                    <svg [class]="'w-3.5 h-3.5 transition-transform ' + (vignetteOpen() ? 'rotate-180' : '')" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                </button>
                @if (vignetteOpen()) {
                    <div class="pb-3 flex flex-col gap-3">
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-[11px] text-text-muted">
                                <span>Amount</span><span class="font-mono">{{ (vignetteAmount() * 100).toFixed(0) }}</span>
                            </div>
                            <input type="range" min="-1" max="1" step="0.01" [ngModel]="vignetteAmount()" (ngModelChange)="vignetteAmount.set(+$event)"
                                class="w-full accent-brand" data-testid="editor-vig-amount">
                        </div>
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-[11px] text-text-muted">
                                <span>Midpoint</span><span class="font-mono">{{ (vignetteMidpoint() * 100).toFixed(0) }}%</span>
                            </div>
                            <input type="range" min="0" max="1" step="0.01" [ngModel]="vignetteMidpoint()" (ngModelChange)="vignetteMidpoint.set(+$event)"
                                class="w-full accent-brand" data-testid="editor-vig-mid">
                        </div>
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-[11px] text-text-muted">
                                <span>Feather</span><span class="font-mono">{{ (vignetteFeather() * 100).toFixed(0) }}%</span>
                            </div>
                            <input type="range" min="0.01" max="1" step="0.01" [ngModel]="vignetteFeather()" (ngModelChange)="vignetteFeather.set(+$event)"
                                class="w-full accent-brand" data-testid="editor-vig-feather">
                        </div>
                    </div>
                }

                <!-- Lens Correction (collapsible) -->
                <button (click)="lensOpen.set(!lensOpen())" class="flex items-center justify-between w-full py-2 text-xs font-bold text-text-subtle uppercase tracking-widest hover:text-text transition-all border-t border-surface-high/20" data-testid="editor-section-lens">
                    <span class="flex items-center gap-2">Lens Correction
                        @if (lensDirty()) { <span (click)="resetLens(); $event.stopPropagation()" class="text-[10px] text-text-muted/60 hover:text-warning cursor-pointer" title="Reset Lens">↻</span> }
                    </span>
                    <svg [class]="'w-3.5 h-3.5 transition-transform ' + (lensOpen() ? 'rotate-180' : '')" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                </button>
                @if (lensOpen()) {
                    <div class="pb-3 flex flex-col gap-3">
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-[11px] text-text-muted">
                                <span>Barrel / Pincushion</span><span class="font-mono">{{ lensBarrel().toFixed(2) }}</span>
                            </div>
                            <input type="range" min="-1" max="1" step="0.01" [ngModel]="lensBarrel()" (ngModelChange)="lensBarrel.set(+$event)"
                                class="w-full accent-brand" data-testid="editor-lens-barrel">
                        </div>
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-[11px] text-text-muted">
                                <span>V. Keystone</span><span class="font-mono">{{ lensVKeystone().toFixed(1) }}°</span>
                            </div>
                            <input type="range" min="-45" max="45" step="0.5" [ngModel]="lensVKeystone()" (ngModelChange)="lensVKeystone.set(+$event)"
                                class="w-full accent-brand" data-testid="editor-lens-vk">
                        </div>
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-[11px] text-text-muted">
                                <span>H. Keystone</span><span class="font-mono">{{ lensHKeystone().toFixed(1) }}°</span>
                            </div>
                            <input type="range" min="-45" max="45" step="0.5" [ngModel]="lensHKeystone()" (ngModelChange)="lensHKeystone.set(+$event)"
                                class="w-full accent-brand" data-testid="editor-lens-hk">
                        </div>
                    </div>
                }
            </div>
            <div class="flex-1 flex items-center justify-center bg-base min-w-0 relative overflow-hidden p-4">
                <!-- Preview container — sized by canvas via fit-content -->
                <div class="relative flex items-center justify-center" style="max-width: 100%; max-height: 100%;">
                    <!-- Edited (bottom layer) -->
                    <canvas #previewCanvas class="block max-w-full max-h-[calc(100vh-8rem)] object-contain rounded-theme-lg shadow-2xl" data-testid="editor-preview"></canvas>

                    <!-- Comparison overlay -->
                    @if (comparisonMode() && trueOriginalUrl()) {
                        <!-- Original image clipped to left of divider -->
                        <img [src]="trueOriginalUrl()"
                            class="absolute inset-0 w-full h-full object-contain rounded-theme-lg pointer-events-none"
                            [style.clip-path]="'inset(0 ' + ((1 - comparePosition()) * 100) + '% 0 0)'"
                            alt="Original">

                        <!-- Divider line -->
                        <div class="absolute top-0 bottom-0 w-0.5 bg-white/80 shadow-lg cursor-ew-resize z-10"
                            [style.left.%]="comparePosition() * 100"
                            (mousedown)="onCompareSliderDown($event)"
                            data-testid="editor-compare-slider">
                            <!-- Handle -->
                            <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-white/90 shadow-xl flex items-center justify-center cursor-ew-resize">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#333" stroke-width="2.5"><polyline points="9 6 4 12 9 18"/><polyline points="15 6 20 12 15 18"/></svg>
                            </div>
                        </div>

                        <!-- Labels -->
                        <span class="absolute top-2 left-3 text-[10px] font-semibold text-white/70 bg-black/40 px-1.5 py-0.5 rounded pointer-events-none">Before</span>
                        <span class="absolute top-2 right-3 text-[10px] font-semibold text-white/70 bg-black/40 px-1.5 py-0.5 rounded pointer-events-none">After</span>
                    }

                    <!-- A/B Toggle (overlaid within image bounds) -->
                    <div class="absolute top-2 left-1/2 -translate-x-1/2 z-20">
                        <button (click)="comparisonMode.set(!comparisonMode())"
                            [class]="'flex items-center gap-1.5 px-3 py-1.5 text-[11px] rounded-theme-lg transition-all backdrop-blur-sm shadow-lg ' +
                                (comparisonMode() ? 'bg-brand/80 text-white border border-brand' : 'bg-black/30 text-white/70 hover:bg-black/50 hover:text-white border border-white/10')"
                            data-testid="editor-compare-toggle">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="12" y1="3" x2="12" y2="21"/></svg>
                            A/B
                        </button>
                    </div>
                </div>
            </div>

            <!-- Right: Histogram Panel -->
            <div class="w-80 flex-shrink-0 border-l border-surface-mid bg-surface-mid overflow-y-auto p-4 flex flex-col gap-1">

                <!-- Histogram (collapsible) -->
                <button (click)="histogramOpen.set(!histogramOpen())" class="flex items-center justify-between w-full py-2 text-xs font-bold text-text-subtle uppercase tracking-widest hover:text-text transition-all" data-testid="editor-section-histogram">
                    <span>Histogram</span>
                    <svg [class]="'w-3.5 h-3.5 transition-transform ' + (histogramOpen() ? 'rotate-180' : '')" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                </button>
                @if (histogramOpen()) {
                    <div class="pb-3">
                        <app-histogram-display
                            [data]="liveHistogram()"
                            data-testid="editor-histogram">
                        </app-histogram-display>
                    </div>
                }

                <!-- Image Info -->
                @if (currentPair(); as pair) {
                    <div class="flex flex-col gap-2 p-3 rounded-theme-lg bg-surface-mid/20 border border-surface-high/20">
                        <span class="text-xs font-medium text-text truncate" [title]="pair.media_file">{{ getFilename(pair.media_file) }}</span>
                        @if (pair.metadata) {
                            <div class="flex gap-3 text-[11px] text-text-muted font-mono">
                                <span>{{ pair.metadata.width }} × {{ pair.metadata.height }}</span>
                                <span>{{ (pair.metadata.size_bytes / (1024 * 1024)).toFixed(2) }} MB</span>
                            </div>
                        }
                    </div>
                }

                <!-- Color Match (collapsible) -->
                <button (click)="colorMatchOpen.set(!colorMatchOpen())" class="flex items-center justify-between w-full py-2 text-xs font-bold text-text-subtle uppercase tracking-widest hover:text-text transition-all border-t border-surface-high/20" data-testid="editor-section-colormatch">
                    <span>Color Match</span>
                    <svg [class]="'w-3.5 h-3.5 transition-transform ' + (colorMatchOpen() ? 'rotate-180' : '')" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                </button>
                @if (colorMatchOpen()) {
                    <div class="pb-3 flex flex-col gap-3">
                        <div class="flex flex-col gap-1">
                            <span class="text-[10px] text-text-muted uppercase tracking-wider">Reference Image</span>
                            @if (colorMatchRef()) {
                                <div class="flex items-center gap-2 p-1.5 rounded-theme-sm bg-brand/10 border border-brand/20">
                                    <img [src]="getStableMediaUrl(colorMatchRef()!)" class="w-10 h-10 rounded object-cover" alt="ref">
                                    <span class="text-[10px] text-text truncate flex-1">{{ getFilename(colorMatchRef()!) }}</span>
                                    <button (click)="colorMatchRef.set(null)" class="text-[10px] text-danger" data-testid="cm-clear-ref">✕</button>
                                </div>
                            } @else {
                                <button (click)="showRefPicker.set(true)" class="w-full px-2 py-2 text-[10px] text-center border border-dashed border-surface-high/40 rounded-theme-sm text-text-muted hover:bg-surface-mid/30 transition-all" data-testid="cm-pick-ref">
                                    + Select reference image
                                </button>
                            }
                        </div>
                        <div class="flex flex-col gap-1">
                            <span class="text-[10px] text-text-muted uppercase tracking-wider">Method</span>
                            <select [ngModel]="colorMatchMethod()" (ngModelChange)="colorMatchMethod.set($event)"
                                class="w-full bg-surface-low border border-surface-high/30 rounded-theme-lg px-2 py-1.5 text-xs text-text" data-testid="cm-method">
                                <option value="cdf">CDF (Histogram)</option>
                                <option value="wavelet">Wavelet</option>
                            </select>
                        </div>
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-[11px] text-text-muted">
                                <span>Strength</span><span class="font-mono">{{ (colorMatchStrength() * 100).toFixed(0) }}%</span>
                            </div>
                            <input type="range" min="0" max="1" step="0.01" [ngModel]="colorMatchStrength()" (ngModelChange)="colorMatchStrength.set(+$event)"
                                class="w-full accent-brand" data-testid="cm-strength">
                        </div>
                        <button (click)="applyColorMatch()" [disabled]="!colorMatchRef() || isApplying()"
                            class="w-full px-3 py-1.5 text-[11px] font-medium rounded-theme-lg transition-all bg-brand/80 hover:bg-brand text-white disabled:opacity-40 disabled:cursor-not-allowed"
                            data-testid="cm-apply">
                            {{ isApplying() ? 'Matching...' : 'Apply Color Match' }}
                        </button>
                    </div>
                }

                <!-- Batch Apply (collapsible) -->
                <button (click)="batchOpen.set(!batchOpen())" class="flex items-center justify-between w-full py-2 text-xs font-bold text-text-subtle uppercase tracking-widest hover:text-text transition-all border-t border-surface-high/20" data-testid="editor-section-batch">
                    <span>Batch Apply</span>
                    <svg [class]="'w-3.5 h-3.5 transition-transform ' + (batchOpen() ? 'rotate-180' : '')" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                </button>
                @if (batchOpen()) {
                    <div class="pb-3 flex flex-col gap-3">
                        <p class="text-[10px] text-text-muted">Apply current adjustments to multiple images in the dataset.</p>
                        <div class="flex gap-1">
                            <button (click)="applyBatchAll()" [disabled]="!isDirty() || isBatchApplying()"
                                class="flex-1 px-2 py-1.5 text-[10px] font-medium rounded-theme-sm bg-brand/60 hover:bg-brand/80 text-white disabled:opacity-40 transition-all"
                                data-testid="batch-apply-all">
                                Apply to All ({{ allPairs().length }})
                            </button>
                            <button (click)="showBatchSelector.set(true)" [disabled]="!isDirty() || isBatchApplying()"
                                class="flex-1 px-2 py-1.5 text-[10px] font-medium rounded-theme-sm bg-surface-mid/60 hover:bg-surface-mid/80 text-text disabled:opacity-40 transition-all"
                                data-testid="batch-select">
                                Select...
                            </button>
                        </div>
                        @if (batchProgress()) {
                            <div class="flex flex-col gap-1">
                                <div class="flex justify-between text-[10px] text-text-muted">
                                    <span>{{ batchProgress()!.current }} / {{ batchProgress()!.total }}</span>
                                    <span>{{ batchProgress()!.status }}</span>
                                </div>
                                <div class="w-full h-1.5 bg-surface-mid/30 rounded-full overflow-hidden">
                                    <div class="h-full bg-brand transition-all rounded-full" [style.width.%]="(batchProgress()!.current / batchProgress()!.total) * 100"></div>
                                </div>
                            </div>
                        }
                    </div>
                }

                <!-- Restore / Denoise (collapsible) -->
                <button (click)="restoreOpen.set(!restoreOpen())" class="flex items-center justify-between w-full py-2 text-xs font-bold text-text-subtle uppercase tracking-widest hover:text-text transition-all border-t border-surface-high/20" data-testid="editor-section-restore">
                    <span class="flex items-center gap-2">Restoration
                        @if (selectedRestoreModel()) { <span (click)="resetRestore(); $event.stopPropagation()" class="text-[10px] text-text-muted/60 hover:text-warning cursor-pointer" title="Reset Restoration">↻</span> }
                    </span>
                    <svg [class]="'w-3.5 h-3.5 transition-transform ' + (restoreOpen() ? 'rotate-180' : '')" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                </button>
                @if (restoreOpen()) {
                    <div class="pb-3 flex flex-col gap-3">
                        <div class="flex flex-col gap-1">
                            <span class="text-[10px] text-text-muted uppercase tracking-wider">Models Folder</span>
                            <div class="flex gap-1">
                                <input type="text" [ngModel]="restoreFolder()" (ngModelChange)="restoreFolder.set($event)"
                                    placeholder="models\\restore" class="flex-1 bg-surface-low border border-surface-high/30 rounded-theme-sm px-2 py-1 text-[10px] text-text" data-testid="restore-folder">
                                <button (click)="scanRestoreModels()" class="px-2 py-1 text-[10px] bg-brand/60 hover:bg-brand/80 text-white rounded-theme-sm transition-all" data-testid="restore-scan">
                                    Scan
                                </button>
                            </div>
                        </div>
                        @if (restoreModels().length > 0) {
                            <div class="flex flex-col gap-1">
                                <span class="text-[10px] text-text-muted uppercase tracking-wider">Model</span>
                                <select [ngModel]="selectedRestoreModel()" (ngModelChange)="selectedRestoreModel.set($event)"
                                    class="w-full bg-surface-low border border-surface-high/30 rounded-theme-lg px-2 py-1.5 text-xs text-text" data-testid="restore-model">
                                    @for (m of restoreModels(); track m.path) {
                                        <option [value]="m.path">{{ m.name }} ({{ m.size_mb }} MB)</option>
                                    }
                                </select>
                            </div>
                            <div class="flex flex-col gap-1">
                                <div class="flex justify-between text-[11px] text-text-muted">
                                    <span>Strength</span><span class="font-mono">{{ (restoreStrength() * 100).toFixed(0) }}%</span>
                                </div>
                                <input type="range" min="0" max="1" step="0.05" [ngModel]="restoreStrength()" (ngModelChange)="restoreStrength.set(+$event)"
                                    class="w-full accent-brand" data-testid="restore-strength">
                            </div>
                            <div class="flex flex-col gap-1">
                                <div class="flex justify-between text-[11px] text-text-muted">
                                    <span>Tile Size</span><span class="font-mono">{{ restoreTileSize() }}px</span>
                                </div>
                                <input type="range" min="128" max="1024" step="64" [ngModel]="restoreTileSize()" (ngModelChange)="restoreTileSize.set(+$event)"
                                    class="w-full accent-brand" data-testid="restore-tile-size">
                            </div>
                            <button (click)="applyRestore()" [disabled]="!selectedRestoreModel() || isRestoring()"
                                class="w-full px-3 py-1.5 text-[11px] font-medium rounded-theme-lg transition-all bg-brand/80 hover:bg-brand text-white disabled:opacity-40 disabled:cursor-not-allowed"
                                data-testid="restore-apply">
                                {{ isRestoring() ? 'Restoring...' : 'Apply Restoration' }}
                            </button>
                        } @else {
                            <p class="text-[10px] text-text-muted italic">Enter a folder path and click Scan to find restore models (e.g. SCUNet, NAFNet, RestoreFormer).</p>
                        }
                        <!-- Download from registry -->
                        <button (click)="toggleRegistry('restore')" class="flex items-center justify-between w-full px-2 py-1.5 text-[10px] bg-surface-low/60 hover:bg-surface-low border border-surface-high/20 rounded-theme-sm text-text-muted hover:text-text transition-all" data-testid="restore-download-btn">
                            <span class="flex items-center gap-1.5"><span>📦</span><span>Download Models</span></span>
                            <svg [class]="'w-3 h-3 transition-transform ' + (restoreRegistryOpen() ? 'rotate-180' : '')" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                        </button>
                        @if (restoreRegistryOpen()) {
                            <div class="flex flex-col gap-1 p-2 rounded-theme-sm bg-surface-low/40 border border-surface-high/15">
                                <span class="text-[9px] text-text-muted uppercase tracking-wider mb-0.5">Curated Restore Models</span>
                                @for (m of restoreRegistry(); track m.filename) {
                                    <div class="flex items-center gap-1.5 text-[10px]">
                                        @if (m.downloaded) {
                                            <span class="text-emerald-400 flex-shrink-0">✓</span>
                                        } @else {
                                            <button (click)="downloadRegistryModel('restore', m.filename)"
                                                [disabled]="restoreDownloading() !== null"
                                                class="flex-shrink-0 px-1.5 py-0.5 text-[9px] bg-brand/50 hover:bg-brand/70 disabled:opacity-40 text-white rounded-sm transition-all"
                                                [attr.data-testid]="'restore-dl-' + m.filename">
                                                @if (restoreDownloading() === m.filename) { ⏳ } @else { ⬇ }
                                            </button>
                                        }
                                        <span class="flex-1 truncate" [title]="m.description">{{ m.filename }}</span>
                                        <span class="text-[9px] text-text-muted/50 font-mono">{{ m.size_mb }}MB</span>
                                    </div>
                                }
                            </div>
                        }
                    </div>
                }

                <!-- Upscale (collapsible) -->
                <button (click)="upscaleOpen.set(!upscaleOpen())" class="flex items-center justify-between w-full py-2 text-xs font-bold text-text-subtle uppercase tracking-widest hover:text-text transition-all border-t border-surface-high/20" data-testid="editor-section-upscale">
                    <span class="flex items-center gap-2">Upscale
                        @if (selectedUpscaleModel()) { <span (click)="resetUpscale(); $event.stopPropagation()" class="text-[10px] text-text-muted/60 hover:text-warning cursor-pointer" title="Reset Upscale">↻</span> }
                    </span>
                    <svg [class]="'w-3.5 h-3.5 transition-transform ' + (upscaleOpen() ? 'rotate-180' : '')" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                </button>
                @if (upscaleOpen()) {
                    <div class="pb-3 flex flex-col gap-3">
                        <div class="flex flex-col gap-1">
                            <span class="text-[10px] text-text-muted uppercase tracking-wider">Models Folder</span>
                            <div class="flex gap-1">
                                <input type="text" [ngModel]="upscaleFolder()" (ngModelChange)="upscaleFolder.set($event)"
                                    placeholder="C:\\models\\upscale" class="flex-1 bg-surface-low border border-surface-high/30 rounded-theme-sm px-2 py-1 text-[10px] text-text" data-testid="upscale-folder">
                                <button (click)="scanModels()" class="px-2 py-1 text-[10px] bg-brand/60 hover:bg-brand/80 text-white rounded-theme-sm transition-all" data-testid="upscale-scan">
                                    Scan
                                </button>
                            </div>
                        </div>
                        @if (upscaleModels().length > 0) {
                            <div class="flex flex-col gap-1">
                                <span class="text-[10px] text-text-muted uppercase tracking-wider">Model</span>
                                <select [ngModel]="selectedUpscaleModel()" (ngModelChange)="selectedUpscaleModel.set($event)"
                                    class="w-full bg-surface-low border border-surface-high/30 rounded-theme-lg px-2 py-1.5 text-xs text-text" data-testid="upscale-model">
                                    @for (m of upscaleModels(); track m.path) {
                                        <option [value]="m.path">{{ m.name }} ({{ m.size_mb }} MB)</option>
                                    }
                                </select>
                            </div>
                            <div class="flex flex-col gap-1">
                                <div class="flex justify-between text-[11px] text-text-muted">
                                    <span>Tile Size</span><span class="font-mono">{{ upscaleTileSize() }}px</span>
                                </div>
                                <input type="range" min="128" max="1024" step="64" [ngModel]="upscaleTileSize()" (ngModelChange)="upscaleTileSize.set(+$event)"
                                    class="w-full accent-brand" data-testid="upscale-tile-size">
                            </div>
                            <div class="flex flex-col gap-1">
                                <span class="text-[10px] text-text-muted uppercase tracking-wider">Target Scale</span>
                                <select [ngModel]="upscaleTargetScale()" (ngModelChange)="upscaleTargetScale.set(+$event)"
                                    class="w-full bg-surface-low border border-surface-high/30 rounded-theme-lg px-2 py-1.5 text-xs text-text" data-testid="upscale-target-scale">
                                    <option [value]="0">Native (model default)</option>
                                    <option [value]="1">1× (no upscale, enhance only)</option>
                                    <option [value]="2">2×</option>
                                    <option [value]="4">4×</option>
                                    <option [value]="8">8×</option>
                                </select>
                            </div>
                            <div class="flex flex-col gap-1">
                                <span class="text-[10px] text-text-muted uppercase tracking-wider">Resize Method</span>
                                <select [ngModel]="upscaleResizeMethod()" (ngModelChange)="upscaleResizeMethod.set($event)"
                                    class="w-full bg-surface-low border border-surface-high/30 rounded-theme-lg px-2 py-1.5 text-xs text-text" data-testid="upscale-resize-method">
                                    <option value="lanczos">Lanczos (sharpest)</option>
                                    <option value="bicubic">Bicubic (smooth)</option>
                                    <option value="bilinear">Bilinear (fast)</option>
                                    <option value="nearest">Nearest (pixelated)</option>
                                </select>
                            </div>
                            <button (click)="applyUpscale()" [disabled]="!selectedUpscaleModel() || isUpscaling()"
                                class="w-full px-3 py-1.5 text-[11px] font-medium rounded-theme-lg transition-all bg-brand/80 hover:bg-brand text-white disabled:opacity-40 disabled:cursor-not-allowed"
                                data-testid="upscale-apply">
                                {{ isUpscaling() ? 'Upscaling...' : 'Apply Upscale' }}
                            </button>
                        } @else {
                            <p class="text-[10px] text-text-muted italic">Enter a folder path and click Scan to find models.</p>
                        }
                        <!-- Download from registry -->
                        <button (click)="toggleRegistry('upscale')" class="flex items-center justify-between w-full px-2 py-1.5 text-[10px] bg-surface-low/60 hover:bg-surface-low border border-surface-high/20 rounded-theme-sm text-text-muted hover:text-text transition-all" data-testid="upscale-download-btn">
                            <span class="flex items-center gap-1.5"><span>📦</span><span>Download Models</span></span>
                            <svg [class]="'w-3 h-3 transition-transform ' + (upscaleRegistryOpen() ? 'rotate-180' : '')" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                        </button>
                        @if (upscaleRegistryOpen()) {
                            <div class="flex flex-col gap-1 p-2 rounded-theme-sm bg-surface-low/40 border border-surface-high/15">
                                <span class="text-[9px] text-text-muted uppercase tracking-wider mb-0.5">Curated Upscale Models</span>
                                @for (m of upscaleRegistry(); track m.filename) {
                                    <div class="flex items-center gap-1.5 text-[10px]">
                                        @if (m.downloaded) {
                                            <span class="text-emerald-400 flex-shrink-0">✓</span>
                                        } @else {
                                            <button (click)="downloadRegistryModel('upscale', m.filename)"
                                                [disabled]="upscaleDownloading() !== null"
                                                class="flex-shrink-0 px-1.5 py-0.5 text-[9px] bg-brand/50 hover:bg-brand/70 disabled:opacity-40 text-white rounded-sm transition-all"
                                                [attr.data-testid]="'upscale-dl-' + m.filename">
                                                @if (upscaleDownloading() === m.filename) { ⏳ } @else { ⬇ }
                                            </button>
                                        }
                                        <span class="flex-1 truncate" [title]="m.description">{{ m.filename }}</span>
                                        <span class="text-[9px] text-text-muted/50 font-mono">{{ m.size_mb }}MB</span>
                                    </div>
                                }
                            </div>
                        }
                    </div>
                }
            </div>
        </div>

        <!-- Status Bar -->
        <div class="h-12 flex items-center justify-center px-6 border-t border-surface-mid bg-surface-low/50 text-xs text-text-subtle">
            ESC to close · Preview is an approximation — final processing uses PIL on the backend
        </div>

        <!-- Reference Image Picker Modal -->
        @if (showRefPicker()) {
            <div class="fixed inset-0 z-[60] bg-black/60 flex items-center justify-center" (click)="showRefPicker.set(false)" data-testid="ref-picker-modal">
                <div class="bg-surface-low rounded-theme-xl shadow-2xl max-w-3xl w-full max-h-[80vh] overflow-hidden border border-surface-high/30" (click)="$event.stopPropagation()">
                    <div class="flex items-center justify-between p-4 border-b border-surface-high/30">
                        <span class="text-sm font-semibold text-text">Select Reference Image</span>
                        <button (click)="showRefPicker.set(false)" class="text-text-muted hover:text-text">✕</button>
                    </div>
                    <div class="p-4 overflow-y-auto max-h-[60vh] grid grid-cols-5 gap-2">
                        @for (pair of allPairs(); track pair.media_file) {
                            <button (click)="colorMatchRef.set(pair.media_file); showRefPicker.set(false)"
                                class="aspect-square rounded-theme-sm overflow-hidden border-2 transition-all hover:border-brand/60 bg-surface-mid"
                                [class.border-brand]="colorMatchRef() === pair.media_file"
                                [class.border-transparent]="colorMatchRef() !== pair.media_file">
                                <img [src]="getStableMediaUrl(pair.media_file)" class="w-full h-full object-cover" loading="lazy">
                            </button>
                        }
                    </div>
                </div>
            </div>
        }

        <!-- Batch Selection Modal -->
        @if (showBatchSelector()) {
            <div class="fixed inset-0 z-[60] bg-black/60 flex items-center justify-center" (click)="showBatchSelector.set(false)" data-testid="batch-selector-modal">
                <div class="bg-surface-low rounded-theme-xl shadow-2xl max-w-3xl w-full max-h-[80vh] overflow-hidden border border-surface-high/30" (click)="$event.stopPropagation()">
                    <div class="flex items-center justify-between p-4 border-b border-surface-high/30">
                        <span class="text-sm font-semibold text-text">Select Images ({{ batchSelected().size }} selected)</span>
                        <div class="flex items-center gap-2">
                            <button (click)="toggleBatchAll()" class="text-[10px] text-brand hover:text-brand-hover" data-testid="batch-toggle-all">
                                {{ batchSelected().size === allPairs().length ? 'Deselect All' : 'Select All' }}
                            </button>
                            <button (click)="showBatchSelector.set(false)" class="text-text-muted hover:text-text">✕</button>
                        </div>
                    </div>
                    <div class="p-4 overflow-y-auto max-h-[55vh] grid grid-cols-5 gap-2">
                        @for (pair of allPairs(); track pair.media_file) {
                            <button (click)="toggleBatchItem(pair.media_file)"
                                class="aspect-square rounded-theme-sm overflow-hidden border-2 transition-all relative"
                                [class.border-brand]="batchSelected().has(pair.media_file)"
                                [class.border-transparent]="!batchSelected().has(pair.media_file)">
                                <img [src]="getStableMediaUrl(pair.media_file)" class="w-full h-full object-cover" loading="lazy">
                                @if (batchSelected().has(pair.media_file)) {
                                    <div class="absolute inset-0 bg-brand/20 flex items-center justify-center">
                                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
                                    </div>
                                }
                            </button>
                        }
                    </div>
                    <div class="flex justify-end p-4 border-t border-surface-high/30">
                        <button (click)="applyBatchSelected()" [disabled]="batchSelected().size === 0 || isBatchApplying()"
                            class="px-4 py-2 text-xs font-medium rounded-theme-lg bg-brand hover:bg-brand-hover text-white disabled:opacity-40 transition-all"
                            data-testid="batch-apply-selected">
                            Apply to {{ batchSelected().size }} images
                        </button>
                    </div>
                </div>
            </div>
        }


        <!-- Overlay Commit Confirmation Modal -->
        @if (showCommitConfirm()) {
            <div class="fixed inset-0 z-[70] bg-black/70 flex items-center justify-center" data-testid="commit-confirm-modal">
                <div class="bg-surface rounded-theme-xl shadow-2xl max-w-md w-full overflow-hidden">
                    <div class="p-6 flex flex-col gap-4">
                        <div class="flex items-center gap-3">
                            <div class="w-10 h-10 rounded-full bg-amber-500/20 flex items-center justify-center flex-shrink-0">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                            </div>
                            <div>
                                <h3 class="text-sm font-semibold text-text">Commit Overlay</h3>
                                <p class="text-[11px] text-text-muted mt-1">This will <strong class="text-amber-400">permanently overwrite</strong> the original image with the current overlay. The overlay recipe will be deleted. This cannot be undone.</p>
                            </div>
                        </div>
                        <div class="flex justify-end gap-2">
                            <button (click)="showCommitConfirm.set(false)" class="px-4 py-2 text-xs font-medium rounded-theme-lg bg-surface-mid/60 hover:bg-surface-mid text-text transition-all" data-testid="commit-cancel">
                                Cancel
                            </button>
                            <button (click)="commitOverlay()" class="px-4 py-2 text-xs font-medium rounded-theme-lg bg-amber-600 hover:bg-amber-500 text-white transition-all" data-testid="commit-confirm">
                                Commit Anyway
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        }

        <!-- Discard Changes Confirmation Modal -->
        @if (showDiscardConfirm()) {
            <div class="fixed inset-0 z-[70] bg-black/70 flex items-center justify-center" data-testid="discard-confirm-modal">
                <div class="bg-surface rounded-theme-xl shadow-2xl max-w-md w-full overflow-hidden">
                    <div class="p-6 flex flex-col gap-4">
                        <div class="flex items-center gap-3">
                            <div class="w-10 h-10 rounded-full bg-amber-500/20 flex items-center justify-center flex-shrink-0">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                            </div>
                            <div>
                                <h3 class="text-sm font-semibold text-text">Unsaved Changes</h3>
                                <p class="text-[11px] text-text-muted mt-1">You have unapplied adjustments. Do you want to <strong class="text-amber-400">apply & save</strong> or discard them?</p>
                            </div>
                        </div>
                        <div class="flex justify-end gap-2">
                            <button (click)="discardAndClose()" class="px-4 py-2 text-xs font-medium rounded-theme-lg bg-surface-mid/60 hover:bg-surface-mid text-text transition-all" data-testid="discard-changes-btn">
                                Discard
                            </button>
                            <button (click)="applyAndClose()" class="px-4 py-2 text-xs font-medium rounded-theme-lg bg-brand hover:bg-brand/90 text-white transition-all" data-testid="apply-and-close-btn">
                                Apply & Save
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        }
    </div>
    `,
    styles: []
})
export class ImageEditorModalComponent implements OnInit, OnDestroy {
    @ViewChild('previewCanvas') previewCanvasRef!: ElementRef<HTMLCanvasElement>;

    private datasetService = inject(DatasetService);
    private toast = inject(ToastService);
    private rtc = inject(RuntimeConfigService);
    private overlayStore = inject(OverlayStore);

    currentPair = input.required<any>();
    datasetName = input.required<string>();
    mediaBaseUrl = input.required<string>();
    allPairs = input<any[]>([]);

    close = output<void>();
    applied = output<void>();

    // Section collapse state
    curvesOpen = signal(true);
    lutOpen = signal(false);
    colorOpen = signal(false);
    sharpenOpen = signal(false);
    histogramOpen = signal(true);
    wbOpen = signal(false);
    hslOpen = signal(false);
    vignetteOpen = signal(false);
    lensOpen = signal(false);

    // Curves state
    masterCurve = signal<CurvePoint[]>([...IDENTITY_CURVE]);
    rCurve = signal<CurvePoint[]>([...IDENTITY_CURVE]);
    gCurve = signal<CurvePoint[]>([...IDENTITY_CURVE]);
    bCurve = signal<CurvePoint[]>([...IDENTITY_CURVE]);

    // Color / Tone
    hueShift = signal(0);
    saturation = signal(1.0);
    contrast = signal(1.0);

    // Sharpening
    sharpenMethod = signal('none');
    sharpenRadius = signal(2.0);
    sharpenPercent = signal(150);
    sharpenThreshold = signal(3);
    sharpenStrength = signal(1.0);

    // CUBE LUT Stack
    lutStack = signal<LoadedLut[]>([]);
    private lutIdCounter = 0;
    hasLuts = computed(() => this.lutStack().length > 0);

    // White Balance
    wbTemperature = signal(6500);
    wbTint = signal(0);

    // HSL Selective
    hslConfig = signal<HSLConfig>({});

    // Vignette
    vignetteAmount = signal(0.0);
    vignetteMidpoint = signal(0.5);
    vignetteFeather = signal(0.5);

    // Lens Correction
    lensBarrel = signal(0.0);
    lensVKeystone = signal(0.0);
    lensHKeystone = signal(0.0);

    // UI state
    isApplying = signal(false);
    liveHistogram = signal<HistogramData | null>(null);
    showDiscardConfirm = signal(false);

    // A/B Comparison
    comparisonMode = signal(false);
    comparePosition = signal(0.5);
    originalImageUrl = signal<string | null>(null);
    trueOriginalUrl = signal<string | null>(null);
    private compareDragging = false;
    private boundCompareMove = (e: MouseEvent) => this.onCompareSliderMove(e);
    private boundCompareUp = () => this.onCompareSliderUp();

    // Per-section dirty computed signals
    curvesDirty = computed(() => {
        return [this.masterCurve(), this.rCurve(), this.gCurve(), this.bCurve()].some(c =>
            c.length !== 2 || c[0].x !== 0 || c[0].y !== 0 || c[1].x !== 255 || c[1].y !== 255
        );
    });
    lutDirty = computed(() => this.lutStack().length > 0);
    colorDirty = computed(() => this.hueShift() !== 0 || this.saturation() !== 1.0 || this.contrast() !== 1.0);
    sharpenDirty = computed(() => this.sharpenMethod() !== 'none');
    wbDirty = computed(() => this.wbTemperature() !== 6500 || this.wbTint() !== 0);
    hslDirty = computed(() => Object.values(this.hslConfig()).some((r: any) =>
        Math.abs(r.hue_shift) > 0.001 || Math.abs(r.saturation) > 0.001 || Math.abs(r.luminance) > 0.001
    ));
    vignetteDirty = computed(() => this.vignetteAmount() !== 0);
    lensDirty = computed(() => this.lensBarrel() !== 0 || this.lensVKeystone() !== 0 || this.lensHKeystone() !== 0);

    // Color Match
    colorMatchOpen = signal(false);
    colorMatchRef = signal<string | null>(null);
    colorMatchMethod = signal('cdf');
    colorMatchStrength = signal(1.0);
    colorMatchApplied = signal(false);
    showRefPicker = signal(false);

    // Batch Apply
    batchOpen = signal(false);
    showBatchSelector = signal(false);
    batchSelected = signal<Set<string>>(new Set());
    isBatchApplying = signal(false);
    batchProgress = signal<{ current: number; total: number; status: string } | null>(null);

    // Upscale
    upscaleOpen = signal(false);
    upscaleFolder = signal('models\\upscale');
    upscaleModels = signal<{ name: string; path: string; size_mb: number }[]>([]);
    selectedUpscaleModel = signal<string | null>(null);
    upscaleTileSize = signal(512);
    isUpscaling = signal(false);
    showUpscaleConfirm = signal(false);
    upscaleTargetScale = signal(0);
    upscaleResizeMethod = signal('lanczos');

    // Restore (Denoise / Face Restore / Deartifact / Dehaze)
    restoreOpen = signal(false);
    restoreFolder = signal('models\\restore');
    restoreModels = signal<{ name: string; path: string; size_mb: number }[]>([]);
    selectedRestoreModel = signal<string | null>(null);
    restoreStrength = signal(1.0);
    restoreTileSize = signal(512);
    isRestoring = signal(false);

    // Model registry & download
    restoreRegistry = signal<any[]>([]);
    restoreRegistryOpen = signal(false);
    restoreDownloading = signal<string | null>(null);
    upscaleRegistry = signal<any[]>([]);
    upscaleRegistryOpen = signal(false);
    upscaleDownloading = signal<string | null>(null);

    // Overlay state
    hasOverlay = signal(false);
    isRendering = signal(false);
    showCommitConfirm = signal(false);

    // Pipeline block ordering
    static readonly DEFAULT_BLOCK_ORDER: string[] = [
        'denoise', 'face_restore', 'white_balance', 'curves', 'cube_lut',
        'hsl_selective', 'hue_saturation', 'contrast', 'vignette',
        'lens_correction', 'sharpening', 'upscale',
    ];
    static readonly BLOCK_LABELS: Record<string, string> = {
        denoise: 'Denoise', face_restore: 'Face Restore', white_balance: 'White Balance',
        curves: 'Curves', cube_lut: 'CUBE LUT', hsl_selective: 'HSL Selective',
        hue_saturation: 'Hue / Saturation', contrast: 'Contrast', vignette: 'Vignette',
        lens_correction: 'Lens Correction', sharpening: 'Sharpening', upscale: 'Upscale',
    };
    pipelineOrder = signal<string[]>([...ImageEditorModalComponent.DEFAULT_BLOCK_ORDER]);
    blockEnabled = signal<Record<string, boolean>>(
        Object.fromEntries(ImageEditorModalComponent.DEFAULT_BLOCK_ORDER.map(t => [t, true]))
    );
    dragSourceIndex = signal<number | null>(null);
    pipelineOverviewOpen = signal(false);
    readonly BLOCK_LABELS = ImageEditorModalComponent.BLOCK_LABELS;

    isDirty = computed(() => {
        const mc = this.masterCurve(), rc = this.rCurve(), gc = this.gCurve(), bc = this.bCurve();
        const curvesChanged = [mc, rc, gc, bc].some(c =>
            c.length !== 2 || c[0].x !== 0 || c[0].y !== 0 || c[1].x !== 255 || c[1].y !== 255
        );
        const hslChanged = Object.values(this.hslConfig()).some(r =>
            Math.abs(r.hue_shift) > 0.001 || Math.abs(r.saturation) > 0.001 || Math.abs(r.luminance) > 0.001
        );
        return curvesChanged ||
            this.colorMatchApplied() ||
            this.hueShift() !== 0 ||
            this.saturation() !== 1.0 ||
            this.contrast() !== 1.0 ||
            this.sharpenMethod() !== 'none' ||
            this.hasLuts() ||
            this.wbTemperature() !== 6500 ||
            this.wbTint() !== 0 ||
            hslChanged ||
            this.vignetteAmount() !== 0 ||
            this.lensBarrel() !== 0 ||
            this.lensVKeystone() !== 0 ||
            this.lensHKeystone() !== 0;
    });

    private originalImageData: ImageData | null = null;
    private previewDebounce: ReturnType<typeof setTimeout> | null = null;
    private checkInterval: ReturnType<typeof setInterval> | null = null;

    private static readonly TENSION = 0.4;
    private static readonly MAX_PREVIEW_SIZE = 2048;

    @HostListener('document:keydown.escape', ['$event'])
    onEsc(event: Event): void {
        event.stopPropagation();
        event.preventDefault();
        if (this.showDiscardConfirm()) {
            this.showDiscardConfirm.set(false);
            return;
        }
        if (this.isDirty()) {
            this.showDiscardConfirm.set(true);
            return;
        }
        this.close.emit();
    }

    discardAndClose(): void {
        this.showDiscardConfirm.set(false);
        this.close.emit();
    }

    applyAndClose(): void {
        this.showDiscardConfirm.set(false);
        this.applyChanges();
    }

    ngOnInit(): void {
        this.loadImage();
    }

    ngOnDestroy(): void {
        if (this.previewDebounce) clearTimeout(this.previewDebounce);
        if (this.checkInterval) clearInterval(this.checkInterval);
    }

    // ── Image Loading ───────────────────────────────────────────────────

    private loadImage(): void {
        // Clear any existing watcher to prevent interval stacking
        if (this.checkInterval) { clearInterval(this.checkInterval); this.checkInterval = null as any; }
        if (this.previewDebounce) { clearTimeout(this.previewDebounce); this.previewDebounce = null as any; }
        const pair = this.currentPair();
        if (!pair) return;

        const originalUrl = this.getMediaUrl(pair.media_file);
        const displayUrl = this.hasOverlay() ? this.getOverlayUrl(pair.media_file) : originalUrl;
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            const canvas = this.previewCanvasRef?.nativeElement;
            if (!canvas) return;

            // Downscale for preview performance
            const maxDim = ImageEditorModalComponent.MAX_PREVIEW_SIZE;
            let pw = img.naturalWidth, ph = img.naturalHeight;
            if (pw > maxDim || ph > maxDim) {
                const scale = maxDim / Math.max(pw, ph);
                pw = Math.round(pw * scale);
                ph = Math.round(ph * scale);
            }

            canvas.width = pw;
            canvas.height = ph;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;
            ctx.drawImage(img, 0, 0, pw, ph);
            this.originalImageData = ctx.getImageData(0, 0, pw, ph);

            // Store current URL for canvas reference
            this.originalImageUrl.set(displayUrl);
            // Preserve the true original URL for A/B comparison (set only once per editor session)
            if (!this.trueOriginalUrl()) {
                this.trueOriginalUrl.set(originalUrl);
            }

            // Compute initial histogram
            this.computeHistogramFromCanvas(ctx, pw, ph);

            // Load server-side histogram for the curves editor background
            this.datasetService.getHistogram(this.datasetName(), pair.media_file).subscribe({
                next: (hist) => this.liveHistogram.set(hist),
            });
        };
        img.src = displayUrl;

        // Load overlay recipe if exists
        this.loadOverlayRecipe();

        // Watch for changes and schedule preview updates
        this.watchChanges();
    }

    /** Reload canvas from overlay URL — called after overlay is created or discovered, without re-triggering recipe load */
    private reloadCanvasForOverlay(): void {
        const pair = this.currentPair();
        if (!pair) return;
        const overlayUrl = this.getOverlayUrl(pair.media_file);
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            const canvas = this.previewCanvasRef?.nativeElement;
            if (!canvas) return;
            const maxDim = ImageEditorModalComponent.MAX_PREVIEW_SIZE;
            let pw = img.naturalWidth, ph = img.naturalHeight;
            if (pw > maxDim || ph > maxDim) {
                const scale = maxDim / Math.max(pw, ph);
                pw = Math.round(pw * scale);
                ph = Math.round(ph * scale);
            }
            canvas.width = pw;
            canvas.height = ph;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;
            ctx.drawImage(img, 0, 0, pw, ph);
            this.originalImageData = ctx.getImageData(0, 0, pw, ph);
            this.originalImageUrl.set(overlayUrl);
            this.computeHistogramFromCanvas(ctx, pw, ph);
        };
        img.src = overlayUrl;
    }

    // ── Overlay Recipe Loading ────────────────────────────────────────


    private loadOverlayRecipe(): void {
        const pair = this.currentPair();
        if (!pair) return;
        // Seed OverlayStore so the commit/delete mutations have a current
        // row to optimistically remove. The component still drives its
        // own recipe-to-signal mapping (the store doesn't model the
        // editor's curves/HSL/etc. state).
        void this.overlayStore.loadFor(this.datasetName(), pair.media_file);
        this.datasetService.getOverlayRecipe(this.datasetName(), pair.media_file).subscribe({
            next: (res: any) => {
                this.hasOverlay.set(true);
                if (res?.recipe?.operations) {
                    this.applyRecipeToSignals(res.recipe.operations);
                }
                // Reload canvas to show the overlay image instead of the original
                this.reloadCanvasForOverlay();
            },
            error: () => this.hasOverlay.set(false),
        });
    }

    private applyRecipeToSignals(blocks: any[]): void {
        // Rebuild pipeline order from recipe block order
        const types = blocks.map((b: any) => b.type as string);
        // Merge with defaults: recipe blocks first, then any missing defaults
        const merged = [...types];
        for (const t of ImageEditorModalComponent.DEFAULT_BLOCK_ORDER) {
            if (!merged.includes(t)) merged.push(t);
        }
        this.pipelineOrder.set(merged);

        // Apply each block's params to the corresponding signal
        for (const block of blocks) {
            const p = block.params || {};
            switch (block.type) {
                case 'denoise':
                case 'face_restore':
                case 'deartifact':
                case 'dehaze':
                    if (p.model_path) this.selectedRestoreModel.set(p.model_path);
                    if (p.strength != null) this.restoreStrength.set(p.strength);
                    if (p.tile_size) this.restoreTileSize.set(p.tile_size);
                    break;
                case 'white_balance':
                    if (p.temperature != null) this.wbTemperature.set(p.temperature);
                    if (p.tint != null) this.wbTint.set(p.tint);
                    break;
                case 'curves':
                    if (p.master) this.masterCurve.set(p.master);
                    if (p.red) this.rCurve.set(p.red);
                    if (p.green) this.gCurve.set(p.green);
                    if (p.blue) this.bCurve.set(p.blue);
                    break;
                case 'hsl_selective':
                    if (p.config) this.hslConfig.set(p.config);
                    break;
                case 'hue_saturation':
                    if (p.hue_shift != null) this.hueShift.set(p.hue_shift);
                    if (p.saturation != null) this.saturation.set(p.saturation);
                    break;
                case 'contrast':
                    if (p.factor != null) this.contrast.set(p.factor);
                    break;
                case 'vignette':
                    if (p.amount != null) this.vignetteAmount.set(p.amount);
                    if (p.midpoint != null) this.vignetteMidpoint.set(p.midpoint);
                    if (p.feather != null) this.vignetteFeather.set(p.feather);
                    break;
                case 'lens_correction':
                    if (p.barrel != null) this.lensBarrel.set(p.barrel);
                    if (p.v_keystone != null) this.lensVKeystone.set(p.v_keystone);
                    if (p.h_keystone != null) this.lensHKeystone.set(p.h_keystone);
                    break;
                case 'sharpening':
                    if (p.method) this.sharpenMethod.set(p.method);
                    if (p.amount != null) this.sharpenPercent.set(p.amount);
                    if (p.radius != null) this.sharpenRadius.set(p.radius);
                    if (p.threshold != null) this.sharpenThreshold.set(p.threshold);
                    break;
                case 'upscale':
                    if (p.model_path) this.selectedUpscaleModel.set(p.model_path);
                    if (p.tile_size) this.upscaleTileSize.set(p.tile_size);
                    if (p.target_scale != null) this.upscaleTargetScale.set(p.target_scale);
                    break;
            }
        }
    }

    // ── Drag-Reorder Handlers ─────────────────────────────────────────

    onBlockDragStart(event: DragEvent, index: number): void {
        this.dragSourceIndex.set(index);
        if (event.dataTransfer) {
            event.dataTransfer.effectAllowed = 'move';
            event.dataTransfer.setData('text/plain', String(index));
        }
    }

    onBlockDragOver(event: DragEvent, index: number): void {
        event.preventDefault();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
    }

    onBlockDrop(event: DragEvent, targetIndex: number): void {
        event.preventDefault();
        const sourceIndex = this.dragSourceIndex();
        if (sourceIndex === null || sourceIndex === targetIndex) return;

        const order = [...this.pipelineOrder()];
        const [moved] = order.splice(sourceIndex, 1);
        order.splice(targetIndex, 0, moved);
        this.pipelineOrder.set(order);
        this.dragSourceIndex.set(null);
    }

    toggleBlockEnabled(blockType: string): void {
        const current = { ...this.blockEnabled() };
        current[blockType] = !current[blockType];
        this.blockEnabled.set(current);
    }

    getBlockSummary(blockType: string): string {
        switch (blockType) {
            case 'denoise': return this.selectedRestoreModel() ? `${(this.restoreStrength() * 100).toFixed(0)}%` : 'off';
            case 'white_balance': return this.wbTemperature() !== 6500 || this.wbTint() !== 0 ? `${this.wbTemperature()}K` : 'off';
            case 'curves': {
                const changed = [this.masterCurve(), this.rCurve(), this.gCurve(), this.bCurve()].some(c =>
                    c.length !== 2 || c[0].x !== 0 || c[0].y !== 0 || c[1].x !== 255 || c[1].y !== 255
                );
                return changed ? 'active' : 'off';
            }
            case 'cube_lut': return this.hasLuts() ? `${this.lutStack().length} LUT(s)` : 'off';
            case 'hsl_selective': {
                const changed = Object.values(this.hslConfig()).some((r: any) =>
                    Math.abs(r.hue_shift) > 0.001 || Math.abs(r.saturation) > 0.001 || Math.abs(r.luminance) > 0.001
                );
                return changed ? 'active' : 'off';
            }
            case 'hue_saturation': return this.hueShift() !== 0 || this.saturation() !== 1.0 ? 'active' : 'off';
            case 'contrast': return this.contrast() !== 1.0 ? `${this.contrast().toFixed(2)}` : 'off';
            case 'vignette': return this.vignetteAmount() !== 0 ? `${this.vignetteAmount().toFixed(2)}` : 'off';
            case 'lens_correction': return (this.lensBarrel() !== 0 || this.lensVKeystone() !== 0 || this.lensHKeystone() !== 0) ? 'active' : 'off';
            case 'sharpening': return this.sharpenMethod() !== 'none' ? this.sharpenMethod() : 'off';
            case 'upscale': return this.selectedUpscaleModel() ? 'ready' : 'off';
            default: return 'off';
        }
    }

    private watchChanges(): void {
        const schedulePreview = () => {
            if (this.previewDebounce) clearTimeout(this.previewDebounce);
            this.previewDebounce = setTimeout(() => this.updatePreview(), 80);
        };

        let lastState = '';
        this.checkInterval = setInterval(() => {
            const state = JSON.stringify({
                mc: this.masterCurve(), rc: this.rCurve(), gc: this.gCurve(), bc: this.bCurve(),
                hue: this.hueShift(), sat: this.saturation(), con: this.contrast(),
                sm: this.sharpenMethod(), sr: this.sharpenRadius(), sp: this.sharpenPercent(),
                st: this.sharpenThreshold(), ss: this.sharpenStrength(),
                luts: this.lutStack().map(l => ({ id: l.id, s: l.strength })),
                wbt: this.wbTemperature(), wbi: this.wbTint(),
                hsl: this.hslConfig(),
                va: this.vignetteAmount(), vm: this.vignetteMidpoint(), vf: this.vignetteFeather(),
                lb: this.lensBarrel(), lv: this.lensVKeystone(), lh: this.lensHKeystone(),
            });
            if (state !== lastState) {
                lastState = state;
                schedulePreview();
            }
        }, 50);
    }

    // ── Canvas Preview Pipeline ─────────────────────────────────────────

    private updatePreview(): void {
        if (!this.originalImageData) return;
        const canvas = this.previewCanvasRef?.nativeElement;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const src = this.originalImageData;
        const out = ctx.createImageData(src.width, src.height);
        const pixels = new Uint8ClampedArray(src.data);

        // White Balance precompute
        const wbTemp = this.wbTemperature();
        const wbTint = this.wbTint();
        const { wbR, wbG, wbB } = this.computeWBFactors(wbTemp, wbTint);

        // 0. CUBE LUT Stack (applied first, sequentially)
        const activeLuts = this.lutStack().filter(l => l.strength > 0);

        // 1. Curves LUT
        const mLut = this.buildChannelLUT(this.masterCurve());
        const rLut = this.buildChannelLUT(this.rCurve());
        const gLut = this.buildChannelLUT(this.gCurve());
        const bLut = this.buildChannelLUT(this.bCurve());

        // HSL selective precompute
        const hslConfig = this.hslConfig();
        const hslEntries = Object.entries(hslConfig).filter(([_, adj]) =>
            Math.abs(adj.hue_shift) > 0.001 || Math.abs(adj.saturation) > 0.001 || Math.abs(adj.luminance) > 0.001
        );
        const hslHasChanges = hslEntries.length > 0;
        const hslRanges: Record<string, [number, number]> = {
            reds: [0, 30], oranges: [30, 30], yellows: [60, 30], greens: [120, 40],
            cyans: [180, 30], blues: [240, 40], purples: [285, 30], magentas: [330, 30],
        };

        for (let i = 0; i < pixels.length; i += 4) {
            let r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];

            // White Balance (RGB multipliers)
            if (wbTemp !== 6500 || wbTint !== 0) {
                r = Math.max(0, Math.min(255, Math.round(r * wbR)));
                g = Math.max(0, Math.min(255, Math.round(g * wbG)));
                b = Math.max(0, Math.min(255, Math.round(b * wbB)));
            }

            // Apply each LUT in stack order
            for (const lut of activeLuts) {
                const [lr, lg, lb] = this.trilinearInterp(lut.parsed, r, g, b);
                r = Math.round(r + (lr - r) * lut.strength);
                g = Math.round(g + (lg - g) * lut.strength);
                b = Math.round(b + (lb - b) * lut.strength);
            }

            // Curves: Master first, then per-channel
            r = rLut[mLut[r]];
            g = gLut[mLut[g]];
            b = bLut[mLut[b]];

            // Hue/Saturation (in HSL space)
            const hShift = this.hueShift();
            const sFactor = this.saturation();
            if (hShift !== 0 || sFactor !== 1.0 || hslHasChanges) {
                let [h, s, l] = this.rgbToHsl(r, g, b);

                // HSL Selective per-range adjustments
                if (hslHasChanges && s > 0.01) {
                    const hueDeg = h * 360;
                    for (const [rangeName, adj] of hslEntries) {
                        const range = hslRanges[rangeName];
                        if (!range) continue;
                        const [center, width] = range;
                        let d = Math.abs(hueDeg - center);
                        d = Math.min(d, 360 - d);
                        if (d > width * 1.5) continue;
                        const falloff = Math.max(0, Math.min(1, 1 - (d - width) / (width * 0.5 + 0.001)));
                        const weight = 0.5 * (1 + Math.cos(Math.PI * (1 - falloff)));
                        h = ((h + (adj.hue_shift / 360) * weight) % 1 + 1) % 1;
                        s = Math.max(0, Math.min(1, s + (adj.saturation / 100) * weight));
                        l = Math.max(0, Math.min(1, l + (adj.luminance / 200) * weight));
                    }
                }

                const newH = ((h + hShift / 360) % 1 + 1) % 1;
                const newS = Math.max(0, Math.min(1, s * sFactor));
                [r, g, b] = this.hslToRgb(newH, newS, l);
            }

            // Contrast
            const cFactor = this.contrast();
            if (cFactor !== 1.0) {
                r = Math.max(0, Math.min(255, Math.round(cFactor * (r - 128) + 128)));
                g = Math.max(0, Math.min(255, Math.round(cFactor * (g - 128) + 128)));
                b = Math.max(0, Math.min(255, Math.round(cFactor * (b - 128) + 128)));
            }

            out.data[i] = r;
            out.data[i + 1] = g;
            out.data[i + 2] = b;
            out.data[i + 3] = pixels[i + 3];
        }

        ctx.putImageData(out, 0, 0);

        // Vignette (post-process — needs x,y coordinates)
        const vigAmount = this.vignetteAmount();
        if (vigAmount !== 0) {
            this.applyVignetteToCanvas(ctx, canvas.width, canvas.height,
                vigAmount, this.vignetteMidpoint(), this.vignetteFeather());
        }

        // Lens Correction (post-process — spatial warp)
        const barrel = this.lensBarrel();
        const vk = this.lensVKeystone();
        const hk = this.lensHKeystone();
        if (barrel !== 0 || vk !== 0 || hk !== 0) {
            this.applyLensCorrectionToCanvas(ctx, canvas.width, canvas.height, barrel, vk, hk);
        }

        // 4. Sharpening (post-process — needs spatial neighbors)
        if (this.sharpenMethod() !== 'none') {
            this.applySharpeningToCanvas(ctx, canvas.width, canvas.height);
        }

        this.computeHistogramFromCanvas(ctx, canvas.width, canvas.height);
    }

    // ── Client-side Sharpening ───────────────────────────────────────────

    private applySharpeningToCanvas(ctx: CanvasRenderingContext2D, w: number, h: number): void {
        const method = this.sharpenMethod();

        if (method === 'unsharp_mask') {
            this.applyUnsharpMask(ctx, w, h, this.sharpenRadius(), this.sharpenPercent() / 100, this.sharpenThreshold());
        } else if (method === 'kernel') {
            this.applyKernelSharpen(ctx, w, h, this.sharpenStrength());
        } else if (method === 'high_pass') {
            this.applyHighPassSharpen(ctx, w, h, this.sharpenRadius(), this.sharpenStrength());
        }
    }

    private applyUnsharpMask(ctx: CanvasRenderingContext2D, w: number, h: number, radius: number, amount: number, threshold: number): void {
        const src = ctx.getImageData(0, 0, w, h);
        const blurred = this.boxBlur(src, Math.max(1, Math.round(radius)));
        const out = ctx.createImageData(w, h);

        for (let i = 0; i < src.data.length; i += 4) {
            for (let ch = 0; ch < 3; ch++) {
                const orig = src.data[i + ch];
                const blur = blurred.data[i + ch];
                const diff = orig - blur;
                if (Math.abs(diff) >= threshold) {
                    out.data[i + ch] = Math.max(0, Math.min(255, Math.round(orig + diff * amount)));
                } else {
                    out.data[i + ch] = orig;
                }
            }
            out.data[i + 3] = src.data[i + 3];
        }
        ctx.putImageData(out, 0, 0);
    }

    private applyKernelSharpen(ctx: CanvasRenderingContext2D, w: number, h: number, strength: number): void {
        const src = ctx.getImageData(0, 0, w, h);
        const out = ctx.createImageData(w, h);
        // Sharpen kernel: center = 1 + 4*s, edges = -s
        const s = strength;
        const center = 1 + 4 * s;

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = (y * w + x) * 4;
                for (let ch = 0; ch < 3; ch++) {
                    const c = src.data[idx + ch] * center;
                    const t = y > 0 ? src.data[((y - 1) * w + x) * 4 + ch] : src.data[idx + ch];
                    const b = y < h - 1 ? src.data[((y + 1) * w + x) * 4 + ch] : src.data[idx + ch];
                    const l = x > 0 ? src.data[(y * w + x - 1) * 4 + ch] : src.data[idx + ch];
                    const r = x < w - 1 ? src.data[(y * w + x + 1) * 4 + ch] : src.data[idx + ch];
                    out.data[idx + ch] = Math.max(0, Math.min(255, Math.round(c - s * (t + b + l + r))));
                }
                out.data[idx + 3] = src.data[idx + 3];
            }
        }
        ctx.putImageData(out, 0, 0);
    }

    private applyHighPassSharpen(ctx: CanvasRenderingContext2D, w: number, h: number, radius: number, strength: number): void {
        const src = ctx.getImageData(0, 0, w, h);
        const blurred = this.boxBlur(src, Math.max(1, Math.round(radius)));
        const out = ctx.createImageData(w, h);

        for (let i = 0; i < src.data.length; i += 4) {
            for (let ch = 0; ch < 3; ch++) {
                const orig = src.data[i + ch];
                // High-pass = original - blur, then overlay blend
                const hp = orig - blurred.data[i + ch] + 128;
                // Overlay blend: if base < 128 -> 2*a*b/255, else 255 - 2*(255-a)*(255-b)/255
                let blended: number;
                if (orig < 128) {
                    blended = (2 * orig * hp) / 255;
                } else {
                    blended = 255 - (2 * (255 - orig) * (255 - hp)) / 255;
                }
                // Mix with strength
                out.data[i + ch] = Math.max(0, Math.min(255, Math.round(orig + (blended - orig) * strength)));
            }
            out.data[i + 3] = src.data[i + 3];
        }
        ctx.putImageData(out, 0, 0);
    }

    /** Fast box blur (2-pass separable) used as Gaussian approximation */
    private boxBlur(src: ImageData, radius: number): ImageData {
        const w = src.width, h = src.height;
        const temp = new Uint8ClampedArray(src.data);
        const out = new Uint8ClampedArray(src.data);
        const size = radius * 2 + 1;

        // Horizontal pass
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                let rSum = 0, gSum = 0, bSum = 0;
                for (let k = -radius; k <= radius; k++) {
                    const sx = Math.max(0, Math.min(w - 1, x + k));
                    const idx = (y * w + sx) * 4;
                    rSum += src.data[idx];
                    gSum += src.data[idx + 1];
                    bSum += src.data[idx + 2];
                }
                const idx = (y * w + x) * 4;
                temp[idx] = rSum / size;
                temp[idx + 1] = gSum / size;
                temp[idx + 2] = bSum / size;
                temp[idx + 3] = src.data[idx + 3];
            }
        }

        // Vertical pass
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                let rSum = 0, gSum = 0, bSum = 0;
                for (let k = -radius; k <= radius; k++) {
                    const sy = Math.max(0, Math.min(h - 1, y + k));
                    const idx = (sy * w + x) * 4;
                    rSum += temp[idx];
                    gSum += temp[idx + 1];
                    bSum += temp[idx + 2];
                }
                const idx = (y * w + x) * 4;
                out[idx] = rSum / size;
                out[idx + 1] = gSum / size;
                out[idx + 2] = bSum / size;
                out[idx + 3] = temp[idx + 3];
            }
        }

        const result = new ImageData(w, h);
        result.data.set(out);
        return result;
    }

    // ── Trilinear interpolation for 3D LUT ──────────────────────────────

    private trilinearInterp(lut: { size: number; table: Float32Array }, r: number, g: number, b: number): [number, number, number] {
        const s = lut.size;
        const scale = (s - 1) / 255;

        const rf = r * scale, gf = g * scale, bf = b * scale;
        const r0 = Math.floor(rf), g0 = Math.floor(gf), b0 = Math.floor(bf);
        const r1 = Math.min(r0 + 1, s - 1), g1 = Math.min(g0 + 1, s - 1), b1 = Math.min(b0 + 1, s - 1);
        const dr = rf - r0, dg = gf - g0, db = bf - b0;

        const idx = (ri: number, gi: number, bi: number) => (bi * s * s + gi * s + ri) * 3;

        // 8 corners
        const c000 = idx(r0, g0, b0), c100 = idx(r1, g0, b0);
        const c010 = idx(r0, g1, b0), c110 = idx(r1, g1, b0);
        const c001 = idx(r0, g0, b1), c101 = idx(r1, g0, b1);
        const c011 = idx(r0, g1, b1), c111 = idx(r1, g1, b1);

        const t = lut.table;
        const result: [number, number, number] = [0, 0, 0];

        for (let ch = 0; ch < 3; ch++) {
            const v000 = t[c000 + ch], v100 = t[c100 + ch];
            const v010 = t[c010 + ch], v110 = t[c110 + ch];
            const v001 = t[c001 + ch], v101 = t[c101 + ch];
            const v011 = t[c011 + ch], v111 = t[c111 + ch];

            const c00 = v000 * (1 - dr) + v100 * dr;
            const c10 = v010 * (1 - dr) + v110 * dr;
            const c01 = v001 * (1 - dr) + v101 * dr;
            const c11 = v011 * (1 - dr) + v111 * dr;

            const c0 = c00 * (1 - dg) + c10 * dg;
            const c1 = c01 * (1 - dg) + c11 * dg;

            result[ch] = Math.max(0, Math.min(255, Math.round((c0 * (1 - db) + c1 * db) * 255)));
        }

        return result;
    }

    // ── Parse CUBE file client-side ─────────────────────────────────────

    private parseCubeString(content: string): { size: number; table: Float32Array } | null {
        const lines = content.split(/\r?\n/);
        let size = 0;
        const entries: number[] = [];

        for (const rawLine of lines) {
            const line = rawLine.trim();
            if (!line || line.startsWith('#') || line.startsWith('TITLE') ||
                line.startsWith('DOMAIN_MIN') || line.startsWith('DOMAIN_MAX')) continue;

            if (line.startsWith('LUT_3D_SIZE')) {
                size = parseInt(line.split(/\s+/)[1], 10);
                continue;
            }
            if (line.startsWith('LUT_1D_SIZE')) return null; // 1D not supported

            const parts = line.split(/\s+/).map(Number);
            if (parts.length >= 3 && !isNaN(parts[0])) {
                entries.push(parts[0], parts[1], parts[2]);
            }
        }

        if (size === 0 || entries.length !== size * size * size * 3) return null;
        return { size, table: new Float32Array(entries) };
    }

    private buildChannelLUT(points: CurvePoint[]): Uint8Array {
        const lut = new Uint8Array(256);
        if (points.length < 2) {
            for (let i = 0; i < 256; i++) lut[i] = i;
            return lut;
        }

        const sorted = [...points].sort((a, b) => a.x - b.x);
        const xs = sorted.map(p => p.x);
        const ys = sorted.map(p => p.y);
        const tau = 1 - ImageEditorModalComponent.TENSION;

        for (let i = 0; i < 256; i++) {
            if (i <= xs[0]) { lut[i] = ys[0]; continue; }
            if (i >= xs[xs.length - 1]) { lut[i] = ys[ys.length - 1]; continue; }

            let seg = 0;
            while (seg < xs.length - 2 && xs[seg + 1] < i) seg++;

            const x0 = xs[seg], x1 = xs[seg + 1];
            const y0 = ys[seg], y1 = ys[seg + 1];
            const t = (i - x0) / (x1 - x0);

            const ym1 = seg > 0 ? ys[seg - 1] : 2 * y0 - y1;
            const y2 = seg < xs.length - 2 ? ys[seg + 2] : 2 * y1 - y0;
            const t2 = t * t, t3 = t2 * t;

            const val = tau * 0.5 * (
                (2 * y0) + (-ym1 + y1) * t +
                (2 * ym1 - 5 * y0 + 4 * y1 - y2) * t2 +
                (-ym1 + 3 * y0 - 3 * y1 + y2) * t3
            ) + (1 - tau) * (y0 + (y1 - y0) * t);

            lut[i] = Math.max(0, Math.min(255, Math.round(val)));
        }
        return lut;
    }

    private computeHistogramFromCanvas(ctx: CanvasRenderingContext2D, w: number, h: number): void {
        const data = ctx.getImageData(0, 0, w, h).data;
        const rH = new Array(256).fill(0);
        const gH = new Array(256).fill(0);
        const bH = new Array(256).fill(0);
        const lH = new Array(256).fill(0);

        for (let i = 0; i < data.length; i += 4) {
            rH[data[i]]++;
            gH[data[i + 1]]++;
            bH[data[i + 2]]++;
            const lum = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
            lH[Math.min(255, lum)]++;
        }
        this.liveHistogram.set({ r: rH, g: gH, b: bH, luminance: lH });
    }

    // ── White Balance Preview ────────────────────────────────────────────

    private computeWBFactors(temperature: number, tint: number): { wbR: number; wbG: number; wbB: number } {
        const kelvinToRgb = (temp: number): [number, number, number] => {
            const t = Math.max(1000, Math.min(40000, temp)) / 100;
            const red = t <= 66 ? 1.0 : Math.min(1, Math.max(0, 329.698727446 * Math.pow(t - 60, -0.1332047592) / 255));
            const green = t <= 66
                ? Math.min(1, Math.max(0, (99.4708025861 * Math.log(t) - 161.1195681661) / 255))
                : Math.min(1, Math.max(0, 288.1221695283 * Math.pow(t - 60, -0.0755148492) / 255));
            const blue = t >= 66 ? 1.0 : t <= 19 ? 0.0 : Math.min(1, Math.max(0, (138.5177312231 * Math.log(t - 10) - 305.0447927307) / 255));
            return [red, green, blue];
        };

        const [tR, tG, tB] = kelvinToRgb(temperature);
        const [nR, nG, nB] = kelvinToRgb(6500);
        let rScale = nR / Math.max(tR, 0.001);
        let gScale = nG / Math.max(tG, 0.001);
        let bScale = nB / Math.max(tB, 0.001);

        const tf = tint / 100;
        gScale *= (1 + tf * 0.3);
        rScale *= (1 - tf * 0.1);
        bScale *= (1 - tf * 0.1);

        return { wbR: rScale, wbG: gScale, wbB: bScale };
    }

    // ── Vignette Preview ────────────────────────────────────────────────

    private applyVignetteToCanvas(ctx: CanvasRenderingContext2D, w: number, h: number,
        amount: number, midpoint: number, feather: number): void {
        const imgData = ctx.getImageData(0, 0, w, h);
        const data = imgData.data;
        const cx = w / 2, cy = h / 2;
        const maxR = Math.sqrt(cx * cx + cy * cy);
        const featherVal = Math.max(feather, 0.01);

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const dx = (x - cx) / cx;
                const dy = (y - cy) / cy;
                const radius = Math.sqrt(dx * dx + dy * dy) / Math.SQRT2;
                const mask = Math.max(0, Math.min(1, (radius - midpoint) / featherVal));
                const mult = amount > 0 ? (1 - amount * mask) : (1 + Math.abs(amount) * mask);

                const idx = (y * w + x) * 4;
                data[idx] = Math.max(0, Math.min(255, Math.round(data[idx] * mult)));
                data[idx + 1] = Math.max(0, Math.min(255, Math.round(data[idx + 1] * mult)));
                data[idx + 2] = Math.max(0, Math.min(255, Math.round(data[idx + 2] * mult)));
            }
        }
        ctx.putImageData(imgData, 0, 0);
    }

    // ── Lens Correction Preview ──────────────────────────────────────────

    private applyLensCorrectionToCanvas(ctx: CanvasRenderingContext2D, w: number, h: number,
        barrel: number, vKeystone: number, hKeystone: number): void {
        const srcData = ctx.getImageData(0, 0, w, h);
        const src = srcData.data;
        const outData = ctx.createImageData(w, h);
        const out = outData.data;

        const cx = w / 2, cy = h / 2;
        const k = barrel * 0.5; // Scale to reasonable range matching backend

        // Precompute perspective coefficients if needed
        const hasKeystone = vKeystone !== 0 || hKeystone !== 0;
        let perspCoeffs: number[] | null = null;
        if (hasKeystone) {
            const vk = Math.tan(vKeystone * Math.PI / 360); // radians of half-angle
            const hk = Math.tan(hKeystone * Math.PI / 360);
            const x0 = hk * w * 0.5;
            const y0 = vk * h * 0.5;
            // Compute inverse perspective coefficients
            perspCoeffs = this.computePerspectiveCoeffs(
                [0, 0, w, 0, w, h, 0, h],                                           // destination (canvas rect)
                [x0, y0, w - x0, -y0, w + x0, h + y0, -x0, h - y0]                  // source (warped corners)
            );
        }

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                let sx = x, sy = y;

                // Barrel/pincushion distortion (reverse mapping)
                if (barrel !== 0) {
                    const xn = (sx - cx) / cx;
                    const yn = (sy - cy) / cy;
                    const r = Math.sqrt(xn * xn + yn * yn);
                    if (r > 0) {
                        const rNew = r * (1 + k * r * r);
                        const factor = rNew / r;
                        sx = cx + (sx - cx) * factor;
                        sy = cy + (sy - cy) * factor;
                    }
                }

                // Perspective keystone (reverse mapping)
                if (perspCoeffs) {
                    const c = perspCoeffs;
                    const denom = c[6] * sx + c[7] * sy + 1;
                    if (Math.abs(denom) > 1e-10) {
                        const px = (c[0] * sx + c[1] * sy + c[2]) / denom;
                        const py = (c[3] * sx + c[4] * sy + c[5]) / denom;
                        sx = px;
                        sy = py;
                    }
                }

                // Bilinear interpolation from source
                const outIdx = (y * w + x) * 4;
                if (sx < 0 || sx >= w - 1 || sy < 0 || sy >= h - 1) {
                    out[outIdx] = 0;
                    out[outIdx + 1] = 0;
                    out[outIdx + 2] = 0;
                    out[outIdx + 3] = 255;
                } else {
                    const x0 = Math.floor(sx), y0 = Math.floor(sy);
                    const fx = sx - x0, fy = sy - y0;
                    const i00 = (y0 * w + x0) * 4;
                    const i10 = i00 + 4;
                    const i01 = ((y0 + 1) * w + x0) * 4;
                    const i11 = i01 + 4;
                    for (let c = 0; c < 3; c++) {
                        out[outIdx + c] = Math.round(
                            src[i00 + c] * (1 - fx) * (1 - fy) +
                            src[i10 + c] * fx * (1 - fy) +
                            src[i01 + c] * (1 - fx) * fy +
                            src[i11 + c] * fx * fy
                        );
                    }
                    out[outIdx + 3] = 255;
                }
            }
        }
        ctx.putImageData(outData, 0, 0);
    }

    private computePerspectiveCoeffs(dst: number[], src: number[]): number[] {
        // Solve 8-coefficient perspective transform: src = M * dst
        // Using least squares for the 8 unknowns
        const A: number[][] = [];
        const B: number[] = [];
        for (let i = 0; i < 4; i++) {
            const dx = dst[i * 2], dy = dst[i * 2 + 1];
            const sx = src[i * 2], sy = src[i * 2 + 1];
            A.push([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy]);
            A.push([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy]);
            B.push(sx);
            B.push(sy);
        }
        // Solve 8x8 system via Gaussian elimination
        const n = 8;
        const M = A.map((row, i) => [...row, B[i]]);
        for (let col = 0; col < n; col++) {
            let maxRow = col;
            for (let row = col + 1; row < n; row++) {
                if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
            }
            [M[col], M[maxRow]] = [M[maxRow], M[col]];
            const pivot = M[col][col];
            if (Math.abs(pivot) < 1e-12) continue;
            for (let j = col; j <= n; j++) M[col][j] /= pivot;
            for (let row = 0; row < n; row++) {
                if (row === col) continue;
                const factor = M[row][col];
                for (let j = col; j <= n; j++) M[row][j] -= factor * M[col][j];
            }
        }
        return M.map(row => row[n]);
    }

    // ── Color Space Conversion ──────────────────────────────────────────

    private rgbToHsl(r: number, g: number, b: number): [number, number, number] {
        r /= 255; g /= 255; b /= 255;
        const max = Math.max(r, g, b), min = Math.min(r, g, b);
        const l = (max + min) / 2;
        if (max === min) return [0, 0, l];
        const d = max - min;
        const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        let h = 0;
        if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
        else if (max === g) h = ((b - r) / d + 2) / 6;
        else h = ((r - g) / d + 4) / 6;
        return [h, s, l];
    }

    private hslToRgb(h: number, s: number, l: number): [number, number, number] {
        if (s === 0) { const v = Math.round(l * 255); return [v, v, v]; }
        const hue2rgb = (p: number, q: number, t: number) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
        };
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        return [
            Math.round(hue2rgb(p, q, h + 1 / 3) * 255),
            Math.round(hue2rgb(p, q, h) * 255),
            Math.round(hue2rgb(p, q, h - 1 / 3) * 255),
        ];
    }

    // ── Event Handlers ──────────────────────────────────────────────────

    onCurveChanged(event: { channel: ChannelKey; points: CurvePoint[] }): void {
        switch (event.channel) {
            case 'master': this.masterCurve.set(event.points); break;
            case 'r': this.rCurve.set(event.points); break;
            case 'g': this.gCurve.set(event.points); break;
            case 'b': this.bCurve.set(event.points); break;
        }
    }

    resetAll(): void {
        this.resetCurves();
        this.resetLut();
        this.resetColor();
        this.resetSharpen();
        this.resetWb();
        this.resetHsl();
        this.resetVignette();
        this.resetLens();
        this.resetRestore();
        this.resetUpscale();
        // Color match
        this.colorMatchApplied.set(false);
        this.colorMatchRef.set(null);
        // Reload original image data from disk (safe — clears intervals first)
        this.loadImage();
    }

    // ── Per-Section Reset Methods ─────────────────────────────────────────

    resetCurves(): void {
        this.masterCurve.set([...IDENTITY_CURVE]);
        this.rCurve.set([...IDENTITY_CURVE]);
        this.gCurve.set([...IDENTITY_CURVE]);
        this.bCurve.set([...IDENTITY_CURVE]);
    }

    resetLut(): void {
        this.lutStack.set([]);
    }

    resetColor(): void {
        this.hueShift.set(0);
        this.saturation.set(1.0);
        this.contrast.set(1.0);
    }

    resetSharpen(): void {
        this.sharpenMethod.set('none');
        this.sharpenRadius.set(2.0);
        this.sharpenPercent.set(150);
        this.sharpenThreshold.set(3);
        this.sharpenStrength.set(1.0);
    }

    resetWb(): void {
        this.wbTemperature.set(6500);
        this.wbTint.set(0);
    }

    resetHsl(): void {
        this.hslConfig.set({});
    }

    resetVignette(): void {
        this.vignetteAmount.set(0.0);
        this.vignetteMidpoint.set(0.5);
        this.vignetteFeather.set(0.5);
    }

    resetLens(): void {
        this.lensBarrel.set(0.0);
        this.lensVKeystone.set(0.0);
        this.lensHKeystone.set(0.0);
    }

    resetRestore(): void {
        this.selectedRestoreModel.set(null);
        this.restoreStrength.set(1.0);
        this.restoreTileSize.set(512);
    }

    resetUpscale(): void {
        this.selectedUpscaleModel.set(null);
        this.upscaleTileSize.set(512);
        this.upscaleTargetScale.set(0);
        this.upscaleResizeMethod.set('lanczos');
    }

    // ── Color Match ──────────────────────────────────────────────────────

    applyColorMatch(): void {
        const pair = this.currentPair();
        const ref = this.colorMatchRef();
        if (!pair || !ref) return;

        this.isApplying.set(true);
        this.datasetService.colorMatch(
            this.datasetName(), pair.media_file, ref,
            this.colorMatchMethod(), this.colorMatchStrength(),
        ).subscribe({
            next: (blob: Blob) => {
                this.isApplying.set(false);
                this.colorMatchApplied.set(true);
                this.toast.success('Color match preview loaded');

                // Load blob as the new canvas base image
                const url = URL.createObjectURL(blob);
                const img = new window.Image();
                img.onload = () => {
                    const canvas = this.previewCanvasRef?.nativeElement;
                    if (!canvas) { URL.revokeObjectURL(url); return; }

                    const maxDim = ImageEditorModalComponent.MAX_PREVIEW_SIZE;
                    let pw = img.naturalWidth, ph = img.naturalHeight;
                    if (pw > maxDim || ph > maxDim) {
                        const scale = maxDim / Math.max(pw, ph);
                        pw = Math.round(pw * scale);
                        ph = Math.round(ph * scale);
                    }

                    canvas.width = pw;
                    canvas.height = ph;
                    const ctx = canvas.getContext('2d');
                    if (!ctx) { URL.revokeObjectURL(url); return; }
                    ctx.drawImage(img, 0, 0, pw, ph);
                    this.originalImageData = ctx.getImageData(0, 0, pw, ph);
                    URL.revokeObjectURL(url);

                    // Re-apply current filters on top of the color-matched base
                    this.updatePreview();
                    this.computeHistogramFromCanvas(ctx, pw, ph);
                };
                img.src = url;
            },
            error: (err) => {
                this.isApplying.set(false);
                this.toast.error(`Color match failed: ${err?.error?.detail || err.message}`);
            },
        });
    }

    // ── Batch Apply ──────────────────────────────────────────────────────

    toggleBatchItem(file: string): void {
        const current = new Set(this.batchSelected());
        if (current.has(file)) current.delete(file);
        else current.add(file);
        this.batchSelected.set(current);
    }

    toggleBatchAll(): void {
        if (this.batchSelected().size === this.allPairs().length) {
            this.batchSelected.set(new Set());
        } else {
            this.batchSelected.set(new Set(this.allPairs().map((p: any) => p.media_file)));
        }
    }

    applyBatchAll(): void {
        const paths = this.allPairs().map((p: any) => p.media_file);
        this.executeBatch(paths);
    }

    applyBatchSelected(): void {
        this.showBatchSelector.set(false);
        const paths = [...this.batchSelected()];
        this.executeBatch(paths);
    }

    private buildCurrentAdjustments(): Record<string, any> {
        const adjustments: Record<string, any> = {};

        // Color match (applied first in backend pipeline)
        if (this.colorMatchApplied() && this.colorMatchRef()) {
            adjustments['color_match'] = {
                reference_path: this.colorMatchRef(),
                method: this.colorMatchMethod(),
                strength: this.colorMatchStrength(),
            };
        }

        const luts = this.lutStack().filter(l => l.strength > 0);
        if (luts.length > 0) {
            adjustments['cube_lut'] = luts[0].content;
            adjustments['cube_lut_strength'] = luts[0].strength;
        }

        const mc = this.masterCurve(), rc = this.rCurve(), gc = this.gCurve(), bc = this.bCurve();
        const hasCurves = [mc, rc, gc, bc].some(c => c.length !== 2 || c[0].y !== 0 || c[1].y !== 255);
        if (hasCurves) adjustments['curves'] = { master: mc, r: rc, g: gc, b: bc };

        if (this.hueShift() !== 0) adjustments['hue_shift'] = this.hueShift();
        if (this.saturation() !== 1.0) adjustments['saturation'] = this.saturation();
        if (this.contrast() !== 1.0) adjustments['contrast'] = this.contrast();

        if (this.sharpenMethod() !== 'none') {
            const method = this.sharpenMethod();
            const params: Record<string, number> = {};
            if (method === 'unsharp_mask') {
                params['radius'] = this.sharpenRadius();
                params['percent'] = this.sharpenPercent();
                params['threshold'] = this.sharpenThreshold();
            } else if (method === 'kernel') {
                params['strength'] = this.sharpenStrength();
            } else if (method === 'high_pass') {
                params['radius'] = this.sharpenRadius();
                params['strength'] = this.sharpenStrength();
            }
            adjustments['sharpening'] = { method, params };
        }

        if (this.wbTemperature() !== 6500 || this.wbTint() !== 0) {
            adjustments['white_balance'] = { temperature: this.wbTemperature(), tint: this.wbTint() };
        }
        if (this.vignetteAmount() !== 0) {
            adjustments['vignette'] = {
                amount: this.vignetteAmount(), midpoint: this.vignetteMidpoint(), feather: this.vignetteFeather(),
            };
        }
        if (this.lensBarrel() !== 0 || this.lensVKeystone() !== 0 || this.lensHKeystone() !== 0) {
            adjustments['lens_correction'] = {
                barrel: this.lensBarrel(), vertical_keystone: this.lensVKeystone(), horizontal_keystone: this.lensHKeystone(),
            };
        }
        const hsl = this.hslConfig();
        const hslActive = Object.values(hsl).some(r =>
            Math.abs(r.hue_shift) > 0.001 || Math.abs(r.saturation) > 0.001 || Math.abs(r.luminance) > 0.001
        );
        if (hslActive) adjustments['hsl_selective'] = hsl;

        return adjustments;
    }

    private executeBatch(paths: string[]): void {
        if (paths.length === 0) return;
        this.isBatchApplying.set(true);
        this.batchProgress.set({ current: 0, total: paths.length, status: 'Starting...' });

        const adjustments = this.buildCurrentAdjustments();
        const url = this.datasetService.getBatchAdjustUrl(this.datasetName());
        const body = JSON.stringify({ paths, ...adjustments });

        fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body,
        }).then(response => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const reader = response.body?.getReader();
            if (!reader) throw new Error('No stream');

            const decoder = new TextDecoder();
            let buffer = '';

            const pump = (): Promise<void> => reader.read().then(({ done, value }) => {
                if (done) {
                    this.isBatchApplying.set(false);
                    this.batchProgress.set(null);
                    this.toast.success(`Batch applied to ${paths.length} images`);
                    this.applied.emit();
                    this.loadImage();
                    return;
                }
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const event = JSON.parse(line.slice(6));
                        if (event.done) {
                            this.batchProgress.set({ current: event.total, total: event.total, status: 'Done' });
                        } else if (event.error) {
                            this.batchProgress.update(p => p ? { ...p, current: event.index + 1, status: `Error: ${event.file}` } : p);
                        } else {
                            this.batchProgress.set({ current: event.index + 1, total: paths.length, status: event.file });
                        }
                    } catch { /* ignore parse errors */ }
                }
                return pump();
            });
            return pump();
        }).catch(err => {
            this.isBatchApplying.set(false);
            this.batchProgress.set(null);
            this.toast.error(`Batch apply failed: ${err.message}`);
        });
    }

    // ── CUBE LUT ────────────────────────────────────────────────────────

    // ── Upscale ──────────────────────────────────────────────────────────

    scanModels(): void {
        const folder = this.upscaleFolder();
        if (!folder) {
            this.toast.error('Enter a models folder path');
            return;
        }
        this.datasetService.listUpscaleModels(folder).subscribe({
            next: (res: any) => {
                this.upscaleModels.set(res.models || []);
                if (res.models?.length) {
                    this.selectedUpscaleModel.set(res.models[0].path);
                    this.toast.success(`Found ${res.models.length} model(s)`);
                } else {
                    this.toast.error('No models found in folder');
                }
            },
            error: (err) => this.toast.error(`Scan failed: ${err?.error?.detail || err.message}`),
        });
    }

    applyUpscale(): void {
        const pair = this.currentPair();
        const model = this.selectedUpscaleModel();
        if (!pair || !model) return;

        this.isUpscaling.set(true);
        const blocks: PipelineBlock[] = [{
            type: 'upscale',
            enabled: true,
            params: {
                model_path: model,
                tile_size: this.upscaleTileSize(),
                tile_pad: 32,
                target_scale: this.upscaleTargetScale(),
                resize_method: this.upscaleResizeMethod(),
            },
        }];
        // OverlayStore handles the optimistic upsert + rollback toast on
        // failure; the OptimisticResult exposes the response payload so
        // the success toast can include the rendered dimensions.
        void this.overlayStore.renderPipeline(
            this.datasetName(), pair.media_file, blocks,
        ).then(result => {
            this.isUpscaling.set(false);
            if (result.ok) {
                this.hasOverlay.set(true);
                const dims = result.value.dimensions;
                this.toast.success(`Upscale applied as overlay (${dims?.[0]}×${dims?.[1]})`);
                this.applied.emit();
                this.loadImage();
            }
        });
    }

    // ── Restoration ──────────────────────────────────────────────────────

    scanRestoreModels(): void {
        const folder = this.restoreFolder();
        if (!folder) {
            this.toast.error('Enter a models folder path');
            return;
        }
        this.datasetService.listRestoreModels(folder).subscribe({
            next: (res: any) => {
                this.restoreModels.set(res.models || []);
                if (res.models?.length) {
                    this.selectedRestoreModel.set(res.models[0].path);
                    this.toast.success(`Found ${res.models.length} restore model(s)`);
                } else {
                    this.toast.error('No restore models found in folder');
                }
            },
            error: (err) => this.toast.error(`Scan failed: ${err?.error?.detail || err.message}`),
        });
    }

    loadRegistry(category: 'restore' | 'upscale'): void {
        this.datasetService.getModelRegistry(category).subscribe({
            next: (res: any) => {
                if (category === 'restore') {
                    this.restoreRegistry.set(res.models || []);
                    this.restoreRegistryOpen.set(true);
                } else {
                    this.upscaleRegistry.set(res.models || []);
                    this.upscaleRegistryOpen.set(true);
                }
            },
            error: (err) => this.toast.error(`Registry load failed: ${err?.error?.detail || err.message}`),
        });
    }

    toggleRegistry(category: 'restore' | 'upscale'): void {
        const openSignal = category === 'restore' ? this.restoreRegistryOpen : this.upscaleRegistryOpen;
        const dataSignal = category === 'restore' ? this.restoreRegistry : this.upscaleRegistry;
        if (openSignal()) {
            openSignal.set(false);
        } else if (dataSignal().length > 0) {
            // Already loaded, just toggle open
            openSignal.set(true);
        } else {
            // First open — fetch from API
            this.loadRegistry(category);
        }
    }

    applyRestore(): void {
        const pair = this.currentPair();
        const model = this.selectedRestoreModel();
        if (!pair || !model) return;

        this.isRestoring.set(true);
        const blocks: PipelineBlock[] = [{
            type: 'denoise',
            enabled: true,
            params: {
                model_path: model,
                strength: this.restoreStrength(),
                tile_size: this.restoreTileSize(),
                tile_pad: 32,
            },
        }];
        // OverlayStore handles the optimistic upsert + rollback toast on
        // failure; see applyUpscale for the pattern.
        void this.overlayStore.renderPipeline(
            this.datasetName(), pair.media_file, blocks,
        ).then(result => {
            this.isRestoring.set(false);
            if (result.ok) {
                this.hasOverlay.set(true);
                const dims = result.value.dimensions;
                this.toast.success(`Restoration applied (${dims?.[0]}×${dims?.[1]})`);
                this.applied.emit();
                this.loadImage();
            }
        });
    }

    downloadRegistryModel(category: 'restore' | 'upscale', filename: string): void {
        const trackingSignal = category === 'restore' ? this.restoreDownloading : this.upscaleDownloading;
        trackingSignal.set(filename);
        this.toast.info(`Downloading ${filename}…`);
        this.datasetService.downloadModel(category, filename).subscribe({
            next: (res: any) => {
                trackingSignal.set(null);
                this.toast.success(`Downloaded ${res.filename} (${res.size_mb} MB)`);
                // Refresh registry status
                this.loadRegistry(category);
                // Re-scan the folder so the model appears in the dropdown
                if (category === 'restore') {
                    this.scanRestoreModels();
                } else {
                    this.scanModels();
                }
            },
            error: (err) => {
                trackingSignal.set(null);
                this.toast.error(`Download failed: ${err?.error?.detail || err.message}`);
            },
        });
    }

    // ── Non-Destructive Overlay Pipeline ──────────────────────────────────

    buildPipelineBlocks(): PipelineBlock[] {
        const blocks: PipelineBlock[] = [];

        // Restoration (denoise) — always first in pipeline
        const restoreModel = this.selectedRestoreModel();
        if (restoreModel) {
            blocks.push({
                type: 'denoise',
                enabled: true,
                params: {
                    model_path: restoreModel,
                    strength: this.restoreStrength(),
                    tile_size: this.restoreTileSize(),
                    tile_pad: 32,
                },
            });
        }

        // White Balance
        if (this.wbTemperature() !== 6500 || this.wbTint() !== 0) {
            blocks.push({
                type: 'white_balance',
                enabled: true,
                params: { temperature: this.wbTemperature(), tint: this.wbTint() },
            });
        }

        // Curves
        const mc = this.masterCurve(), rc = this.rCurve(), gc = this.gCurve(), bc = this.bCurve();
        const hasCurves = [mc, rc, gc, bc].some(c => c.length !== 2 || c[0].y !== 0 || c[1].y !== 255);
        if (hasCurves) {
            blocks.push({
                type: 'curves',
                enabled: true,
                params: { master: mc, r: rc, g: gc, b: bc },
            });
        }

        // CUBE LUT
        const luts = this.lutStack().filter(l => l.strength > 0);
        if (luts.length > 0) {
            blocks.push({
                type: 'cube_lut',
                enabled: true,
                params: { cube_lut: luts[0].content, cube_lut_strength: luts[0].strength },
            });
        }

        // HSL Selective
        const hsl = this.hslConfig();
        const hslActive = Object.values(hsl).some(r =>
            Math.abs(r.hue_shift) > 0.001 || Math.abs(r.saturation) > 0.001 || Math.abs(r.luminance) > 0.001
        );
        if (hslActive) {
            blocks.push({
                type: 'hsl_selective',
                enabled: true,
                params: { hsl_config: hsl },
            });
        }

        // Hue / Saturation
        if (this.hueShift() !== 0 || this.saturation() !== 1.0) {
            blocks.push({
                type: 'hue_saturation',
                enabled: true,
                params: { hue_shift: this.hueShift(), saturation: this.saturation() },
            });
        }

        // Contrast
        if (this.contrast() !== 1.0) {
            blocks.push({
                type: 'contrast',
                enabled: true,
                params: { contrast: this.contrast() },
            });
        }

        // Vignette
        if (this.vignetteAmount() !== 0) {
            blocks.push({
                type: 'vignette',
                enabled: true,
                params: {
                    amount: this.vignetteAmount(),
                    midpoint: this.vignetteMidpoint(),
                    feather: this.vignetteFeather(),
                },
            });
        }

        // Lens Correction
        if (this.lensBarrel() !== 0 || this.lensVKeystone() !== 0 || this.lensHKeystone() !== 0) {
            blocks.push({
                type: 'lens_correction',
                enabled: true,
                params: {
                    barrel: this.lensBarrel(),
                    vertical_keystone: this.lensVKeystone(),
                    horizontal_keystone: this.lensHKeystone(),
                },
            });
        }

        // Sharpening
        if (this.sharpenMethod() !== 'none') {
            const method = this.sharpenMethod();
            const params: Record<string, any> = { method };
            if (method === 'unsharp_mask') {
                params['params'] = { radius: this.sharpenRadius(), percent: this.sharpenPercent(), threshold: this.sharpenThreshold() };
            } else if (method === 'kernel') {
                params['params'] = { strength: this.sharpenStrength() };
            } else if (method === 'high_pass') {
                params['params'] = { radius: this.sharpenRadius(), strength: this.sharpenStrength() };
            }
            blocks.push({ type: 'sharpening', enabled: true, params });
        }

        // Upscale
        const upscaleModel = this.selectedUpscaleModel();
        if (upscaleModel && this.showUpscaleConfirm()) {
            blocks.push({
                type: 'upscale',
                enabled: true,
                params: {
                    model_path: upscaleModel,
                    tile_size: this.upscaleTileSize(),
                    tile_pad: 32,
                    target_scale: this.upscaleTargetScale(),
                    resize_method: this.upscaleResizeMethod(),
                },
            });
        }

        // Sort blocks according to user's pipeline order and apply enabled toggles
        const order = this.pipelineOrder();
        const enabled = this.blockEnabled();
        blocks.sort((a, b) => {
            const ai = order.indexOf(a.type);
            const bi = order.indexOf(b.type);
            return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
        });
        // Apply enable/disable toggles from the pipeline overview
        for (const block of blocks) {
            if (enabled[block.type] === false) {
                block.enabled = false;
            }
        }
        return blocks;
    }

    renderOverlay(): void {
        const pair = this.currentPair();
        if (!pair) return;

        const blocks = this.buildPipelineBlocks();
        if (blocks.length === 0) {
            this.toast.error('No adjustments to render');
            return;
        }

        this.isRendering.set(true);
        // OverlayStore handles the optimistic upsert + rollback toast on
        // failure; see applyUpscale for the pattern.
        void this.overlayStore.renderPipeline(
            this.datasetName(), pair.media_file, blocks, 512, 32, true,
        ).then(result => {
            this.isRendering.set(false);
            if (result.ok) {
                this.hasOverlay.set(true);
                const dims = result.value.dimensions;
                this.toast.success(`Overlay saved (${dims?.[0]}×${dims?.[1]})`);
                this.applied.emit();
            }
        });
    }

    revertOverlay(): void {
        const pair = this.currentPair();
        if (!pair) return;

        // Optimistic delete through the store. The store toasts on failure
        // and restores the row; success path updates local UI immediately.
        this.hasOverlay.set(false);
        this.toast.success('Overlay reverted — original restored');
        this.applied.emit();
        void this.overlayStore.deleteOverlay(this.datasetName(), pair.media_file);
    }

    commitOverlay(): void {
        const pair = this.currentPair();
        if (!pair) return;

        // Optimistic commit through the store. The store toasts on failure
        // and restores the overlay row; success path updates local UI
        // immediately (commit flattens overlay into original, so the row
        // is removed locally and the image reloads to pick up the new
        // base file).
        this.showCommitConfirm.set(false);
        this.isApplying.set(false);
        this.hasOverlay.set(false);
        this.toast.success('Overlay committed — now the original');
        this.applied.emit();
        this.loadImage();
        void this.overlayStore.commitOverlay(this.datasetName(), pair.media_file);
    }

    onCubeFileSelected(event: Event): void {
        const input = event.target as HTMLInputElement;
        const file = input.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = () => {
            const content = reader.result as string;
            const parsed = this.parseCubeString(content);
            if (parsed) {
                const entry: LoadedLut = {
                    id: ++this.lutIdCounter,
                    name: file.name,
                    content,
                    strength: 1.0,
                    parsed,
                };
                this.lutStack.update(stack => [...stack, entry]);
                this.toast.success(`LUT added: ${file.name} (${parsed.size}³)`);
            } else {
                this.toast.error('Invalid CUBE file format');
            }
        };
        reader.readAsText(file);
        input.value = '';
    }

    removeLut(id: number): void {
        this.lutStack.update(stack => stack.filter(l => l.id !== id));
    }

    updateLutStrength(id: number, strength: number): void {
        this.lutStack.update(stack =>
            stack.map(l => l.id === id ? { ...l, strength } : l)
        );
    }

    exportCubeLut(): void {
        const curves: CurvesConfig = {
            master: this.masterCurve(),
            r: this.rCurve(),
            g: this.gCurve(),
            b: this.bCurve(),
        };
        this.datasetService.exportCube(this.datasetName(), curves).subscribe({
            next: (blob) => {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'exported_curves.cube';
                a.click();
                URL.revokeObjectURL(url);
                this.toast.success('CUBE file exported');
            },
            error: () => this.toast.error('Failed to export CUBE file'),
        });
    }

    // ── Apply to Backend ────────────────────────────────────────────────

    applyChanges(): void {
        const pair = this.currentPair();
        if (!pair) return;

        this.isApplying.set(true);

        const adjustments: ImageAdjustments = {};

        // Color match (applied first in backend pipeline)
        if (this.colorMatchApplied() && this.colorMatchRef()) {
            adjustments.color_match = {
                reference_path: this.colorMatchRef()!,
                method: this.colorMatchMethod(),
                strength: this.colorMatchStrength(),
            };
        }

        // Note: LUTs are applied client-side in preview. For backend,
        // we send the first LUT if present (backend supports single LUT).
        const luts = this.lutStack().filter(l => l.strength > 0);
        if (luts.length > 0) {
            // Use first LUT for backend apply (multi-LUT is preview-only)
            adjustments.cube_lut = luts[0].content;
            adjustments.cube_lut_strength = luts[0].strength;
        }

        {
            const mc = this.masterCurve(), rc = this.rCurve(), gc = this.gCurve(), bc = this.bCurve();
            const hasCurves = [mc, rc, gc, bc].some(c =>
                c.length !== 2 || c[0].y !== 0 || c[1].y !== 255
            );
            if (hasCurves) {
                adjustments.curves = { master: mc, r: rc, g: gc, b: bc };
            }
        }

        if (this.hueShift() !== 0) adjustments.hue_shift = this.hueShift();
        if (this.saturation() !== 1.0) adjustments.saturation = this.saturation();
        if (this.contrast() !== 1.0) adjustments.contrast = this.contrast();

        if (this.sharpenMethod() !== 'none') {
            const method = this.sharpenMethod();
            const params: Record<string, number> = {};
            if (method === 'unsharp_mask') {
                params['radius'] = this.sharpenRadius();
                params['percent'] = this.sharpenPercent();
                params['threshold'] = this.sharpenThreshold();
            } else if (method === 'kernel') {
                params['strength'] = this.sharpenStrength();
            } else if (method === 'high_pass') {
                params['radius'] = this.sharpenRadius();
                params['strength'] = this.sharpenStrength();
            }
            adjustments.sharpening = { method, params };
        }

        if (this.wbTemperature() !== 6500 || this.wbTint() !== 0) {
            adjustments.white_balance = { temperature: this.wbTemperature(), tint: this.wbTint() };
        }
        if (this.vignetteAmount() !== 0) {
            adjustments.vignette = {
                amount: this.vignetteAmount(),
                midpoint: this.vignetteMidpoint(),
                feather: this.vignetteFeather(),
            };
        }
        if (this.lensBarrel() !== 0 || this.lensVKeystone() !== 0 || this.lensHKeystone() !== 0) {
            adjustments.lens_correction = {
                barrel: this.lensBarrel(),
                vertical_keystone: this.lensVKeystone(),
                horizontal_keystone: this.lensHKeystone(),
            };
        }
        const hsl = this.hslConfig();
        const hslActive = Object.values(hsl).some(r =>
            Math.abs(r.hue_shift) > 0.001 || Math.abs(r.saturation) > 0.001 || Math.abs(r.luminance) > 0.001
        );
        if (hslActive) {
            adjustments.hsl_selective = hsl as any;
        }

        this.datasetService.applyImageAdjustments(this.datasetName(), pair.media_file, adjustments).subscribe({
            next: () => {
                this.isApplying.set(false);
                this.toast.success('Adjustments applied successfully');
                this.applied.emit();
                // Reload the image as new baseline (cache-bust) so A/B shows the applied state
                this.resetAll();
                this.loadImage();
            },
            error: (err) => {
                this.isApplying.set(false);
                this.toast.error(`Failed to apply adjustments: ${err?.error?.detail || err.message}`);
            },
        });
    }

    // ── Utilities ───────────────────────────────────────────────────────

    getMediaUrl(relativePath: string): string {
        return `${this.mediaBaseUrl()}/${encodeURIComponent(this.datasetName())}/${encodeURIComponent(relativePath)}?t=${Date.now()}`;
    }

    getOverlayUrl(imagePath: string): string {
        return `${this.rtc.apiUrl}/datasets/${encodeURIComponent(this.datasetName())}/overlay/${encodeURIComponent(imagePath)}?t=${Date.now()}`;
    }

    /** Stable URL without cache-buster — safe for use in grids/pickers where change detection runs continuously */
    getStableMediaUrl(relativePath: string): string {
        return `${this.mediaBaseUrl()}/${encodeURIComponent(this.datasetName())}/${encodeURIComponent(relativePath)}`;
    }

    getFilename(path: string): string {
        return path?.split('/').pop() || path;
    }

    // ── A/B Comparison Slider ────────────────────────────────────────────

    onCompareSliderDown(event: MouseEvent): void {
        this.compareDragging = true;
        document.addEventListener('mousemove', this.boundCompareMove);
        document.addEventListener('mouseup', this.boundCompareUp);
        event.preventDefault();
    }

    private onCompareSliderMove(event: MouseEvent): void {
        if (!this.compareDragging) return;
        const canvas = this.previewCanvasRef?.nativeElement;
        if (!canvas) return;
        const container = canvas.parentElement;
        if (!container) return;
        const rect = container.getBoundingClientRect();
        const pos = (event.clientX - rect.left) / rect.width;
        this.comparePosition.set(Math.max(0, Math.min(1, pos)));
    }

    private onCompareSliderUp(): void {
        this.compareDragging = false;
        document.removeEventListener('mousemove', this.boundCompareMove);
        document.removeEventListener('mouseup', this.boundCompareUp);
    }
}
