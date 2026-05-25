import { Component, input, output, signal, computed, inject, HostListener, ViewChild, ElementRef, OnInit, OnDestroy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DatasetService } from '../../../../services/dataset';

@Component({
    selector: 'app-viewer-crop-preview-modal',
    standalone: true,
    imports: [FormsModule],
    template: `
        <div class="fixed inset-0 z-[120] flex items-center justify-center p-6 backdrop-blur-md bg-overlay animate-fadeIn" (click)="close.emit()">
            <div class="bg-surface-low border border-surface-high rounded-theme-2xl shadow-2xl overflow-hidden flex flex-col max-w-[90vw] max-h-[90vh] border-shine"
                 (click)="$event.stopPropagation()">

                <!-- Header -->
                <div class="p-4 border-b border-surface-high flex items-center justify-between bg-surface-mid/50">
                    <div>
                        <h2 class="text-lg font-bold text-white">Crop Preview</h2>
                        <p class="text-xs text-text-subtle font-mono truncate max-w-[400px]">{{ item()?.path }}</p>
                    </div>
                    <div class="flex items-center gap-3">
                        <div class="text-sm font-black text-white">
                            {{ item()?.width }}×{{ item()?.height }}
                            <span class="text-text-disabled mx-1">→</span>
                            <span class="text-brand">{{ effectiveTargetW() }}×{{ effectiveTargetH() }}</span>
                        </div>
                        <button (click)="close.emit()" class="text-text-subtle hover:text-white transition-colors">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                        </button>
                    </div>
                </div>

                <!-- Body -->
                <div class="flex-1 overflow-auto p-6 flex gap-6 items-start">
                    <!-- Image + Crop Overlay -->
                    <div class="flex-1 flex items-center justify-center">
                        <div #containerEl class="relative bg-base overflow-hidden shadow-xl border border-white/5 inline-block select-none"
                             (mousemove)="onMouseMove($event)"
                             (mouseup)="onMouseUp()"
                             (mouseleave)="onMouseUp()">
                            <img #imageEl
                                 [src]="getMediaUrl(item()!.path)"
                                 class="max-w-[65vw] max-h-[65vh] object-contain block pointer-events-none"
                                 draggable="false"
                                 (load)="onImageLoad($event)">

                            <!-- Dark overlay outside the crop window -->
                            @if (hasOverlay()) {
                                <!-- Top dark band -->
                                <div class="absolute left-0 right-0 top-0 bg-base/75 pointer-events-none transition-all duration-75"
                                     [style.height.px]="cropRect().top"></div>
                                <!-- Bottom dark band -->
                                <div class="absolute left-0 right-0 bottom-0 bg-base/75 pointer-events-none transition-all duration-75"
                                     [style.height.px]="renderedH() - cropRect().top - cropRect().height"></div>
                                <!-- Left dark band (between top and bottom) -->
                                <div class="absolute bg-base/75 pointer-events-none transition-all duration-75"
                                     [style.top.px]="cropRect().top"
                                     [style.left.px]="0"
                                     [style.width.px]="cropRect().left"
                                     [style.height.px]="cropRect().height"></div>
                                <!-- Right dark band (between top and bottom) -->
                                <div class="absolute bg-base/75 pointer-events-none transition-all duration-75"
                                     [style.top.px]="cropRect().top"
                                     [style.right.px]="0"
                                     [style.width.px]="renderedW() - cropRect().left - cropRect().width"
                                     [style.height.px]="cropRect().height"></div>

                                <!-- Crop border — draggable -->
                                <div class="absolute border-2 border-brand/80 transition-all duration-75 shadow-[0_0_0_1px_rgba(0,0,0,0.3)]"
                                     [class.cursor-move]="!isResizing"
                                     [style.top.px]="cropRect().top"
                                     [style.left.px]="cropRect().left"
                                     [style.width.px]="cropRect().width"
                                     [style.height.px]="cropRect().height"
                                     (mousedown)="onCropMouseDown($event, 'move')">

                                    <!-- Rule of thirds grid -->
                                    <div class="absolute inset-0 pointer-events-none opacity-50">
                                        <div class="absolute left-1/3 top-0 bottom-0 w-px bg-white/50"></div>
                                        <div class="absolute left-2/3 top-0 bottom-0 w-px bg-white/50"></div>
                                        <div class="absolute top-1/3 left-0 right-0 h-px bg-white/50"></div>
                                        <div class="absolute top-2/3 left-0 right-0 h-px bg-white/50"></div>
                                    </div>

                                    <!-- Corner resize handles -->
                                    <div class="absolute -top-[4px] -left-[4px] w-4 h-4 border-t-2 border-l-2 border-brand cursor-nw-resize z-10 hover:border-white transition-colors"
                                         (mousedown)="onCropMouseDown($event, 'resize-tl')"></div>
                                    <div class="absolute -top-[4px] -right-[4px] w-4 h-4 border-t-2 border-r-2 border-brand cursor-ne-resize z-10 hover:border-white transition-colors"
                                         (mousedown)="onCropMouseDown($event, 'resize-tr')"></div>
                                    <div class="absolute -bottom-[4px] -left-[4px] w-4 h-4 border-b-2 border-l-2 border-brand cursor-sw-resize z-10 hover:border-white transition-colors"
                                         (mousedown)="onCropMouseDown($event, 'resize-bl')"></div>
                                    <div class="absolute -bottom-[4px] -right-[4px] w-4 h-4 border-b-2 border-r-2 border-brand cursor-se-resize z-10 hover:border-white transition-colors"
                                         (mousedown)="onCropMouseDown($event, 'resize-br')"></div>

                                    <!-- Center crosshair -->
                                    <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-5 h-5 pointer-events-none">
                                        <div class="absolute top-1/2 left-0 right-0 h-px bg-brand/40"></div>
                                        <div class="absolute left-1/2 top-0 bottom-0 w-px bg-brand/40"></div>
                                    </div>
                                </div>
                            }
                        </div>
                    </div>

                    <!-- Right Panel: AR + Origin + Info -->
                    <div class="flex flex-col gap-4 w-52 shrink-0">
                        <!-- AR Selector -->
                        <div class="bg-surface-mid/30 border border-surface-high rounded-theme-xl p-4">
                            <h4 class="text-[10px] text-text-subtle font-bold uppercase tracking-widest mb-3">Target Aspect Ratio</h4>
                            <select [(ngModel)]="selectedAR"
                                    (ngModelChange)="onARChange($event)"
                                    class="w-full bg-surface-mid text-white text-xs border border-surface-high rounded-theme-md px-3 py-2 outline-none focus:border-brand transition-colors">
                                @for (ar of arPresets; track ar.value) {
                                    <option [value]="ar.value">{{ ar.label }}</option>
                                }
                            </select>
                            <p class="text-[10px] text-text-subtle mt-2">
                                {{ effectiveTargetW() }}×{{ effectiveTargetH() }}
                                <span class="text-text-disabled ml-1">({{ effectiveAR() }})</span>
                            </p>
                        </div>

                        <!-- Origin quick-position grid -->
                        <div class="bg-surface-mid/30 border border-surface-high rounded-theme-xl p-4">
                            <h4 class="text-[10px] text-text-subtle font-bold uppercase tracking-widest mb-3">Quick Position</h4>
                            <div class="grid grid-cols-3 gap-1.5 place-items-center justify-center mx-auto w-fit">
                                @for (opt of originOptions; track opt.value) {
                                    <button (click)="snapToOrigin(opt.value)"
                                            [class.bg-brand]="selectedOrigin() === opt.value"
                                            [class.text-white]="selectedOrigin() === opt.value"
                                            [class.shadow-lg]="selectedOrigin() === opt.value"
                                            [class.shadow-brand/30]="selectedOrigin() === opt.value"
                                            [class.bg-surface-high]="selectedOrigin() !== opt.value"
                                            [class.text-text-muted]="selectedOrigin() !== opt.value"
                                            class="w-10 h-10 rounded-theme-md flex items-center justify-center transition-all hover:opacity-80 active:scale-90 text-sm font-bold"
                                            [title]="opt.label">
                                        {{ opt.icon }}
                                    </button>
                                }
                            </div>
                            <p class="text-[10px] text-text-muted mt-2 text-center font-medium">{{ selectedOriginLabel() }}</p>
                        </div>

                        <!-- Crop position & dimensions info -->
                        <div class="bg-surface-mid/30 border border-surface-high rounded-theme-xl p-4">
                            <h4 class="text-[10px] text-text-subtle font-bold uppercase tracking-widest mb-2">Crop Region</h4>
                            <div class="text-xs font-mono text-white space-y-1.5">
                                <div class="flex justify-between">
                                    <span class="text-text-subtle">Position</span>
                                    <span class="text-brand font-bold">{{ freeformX() }}, {{ freeformY() }}</span>
                                </div>
                                <div class="flex justify-between">
                                    <span class="text-text-subtle">Size</span>
                                    <span class="text-brand font-bold">{{ effectiveTargetW() }}×{{ effectiveTargetH() }}</span>
                                </div>
                                <div class="flex justify-between">
                                    <span class="text-text-subtle">Δ Width</span>
                                    <span class="text-brand font-bold">{{ cropDeltaW() }}px</span>
                                </div>
                                <div class="flex justify-between">
                                    <span class="text-text-subtle">Δ Height</span>
                                    <span class="text-brand font-bold">{{ cropDeltaH() }}px</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Footer -->
                <div class="p-4 border-t border-surface-high flex items-center justify-between bg-surface-mid/30">
                    <div class="flex items-center gap-2 text-text-subtle text-xs">
                        <kbd class="px-1.5 py-0.5 bg-surface-mid rounded text-[10px] font-mono border border-surface-high">ESC</kbd>
                        <span>to close</span>
                    </div>
                    <div class="flex items-center gap-3">
                        <button (click)="close.emit()" class="px-5 py-2 bg-surface-high hover:bg-white/10 text-white rounded-theme-xl text-xs font-bold transition-all border border-white/5 active:scale-95">
                            Cancel
                        </button>
                        <button (click)="applyCrop()" [disabled]="isCropping() || !needsCrop()"
                                class="px-5 py-2 bg-brand text-white rounded-theme-xl text-xs font-bold shadow-lg shadow-brand/30 hover:shadow-brand/40 transition-all active:scale-95 disabled:opacity-50 flex items-center gap-2">
                            @if (isCropping()) {
                                <svg class="animate-spin h-3.5 w-3.5" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                <span>Cropping...</span>
                            } @else {
                                <span>Apply Crop</span>
                            }
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `,
    styles: []
})
export class ViewerCropPreviewModalComponent implements OnInit, OnDestroy {
    @ViewChild('imageEl') imageEl!: ElementRef<HTMLImageElement>;
    @ViewChild('containerEl') containerEl!: ElementRef<HTMLDivElement>;
    private datasetService = inject(DatasetService);

    item = input.required<any>();
    datasetName = input.required<string>();
    mediaBaseUrl = input.required<string>();
    lastUpdateTime = input<number>(0);

    close = output<void>();
    cropped = output<any>();

    selectedOrigin = signal('center');
    selectedAR = signal('auto');
    isCropping = signal(false);
    renderedW = signal(0);
    renderedH = signal(0);

    // Freeform crop position in natural (source image) pixels
    freeformX = signal(0);
    freeformY = signal(0);

    // Store original targets from backend (for "Auto" restore)
    private originalTargetW = 0;
    private originalTargetH = 0;

    // Effective targets (updated by AR override or resize)
    effectiveTargetW = signal(0);
    effectiveTargetH = signal(0);

    // Drag/resize interaction state
    isDragging = false;
    isResizing = false;
    private interactionMode: 'move' | 'resize-tl' | 'resize-tr' | 'resize-bl' | 'resize-br' = 'move';
    private dragStartX = 0;
    private dragStartY = 0;
    private dragStartFreeformX = 0;
    private dragStartFreeformY = 0;
    private dragStartTargetW = 0;
    private dragStartTargetH = 0;

    // Bound handler reference for cleanup
    private boundGlobalMouseUp = this.onMouseUp.bind(this);

    arPresets = [
        { value: 'auto', label: 'Auto (from analysis)', ratio: 0 },
        { value: '1:1', label: '1:1 — Square', ratio: 1 },
        { value: '4:3', label: '4:3', ratio: 4 / 3 },
        { value: '3:2', label: '3:2', ratio: 3 / 2 },
        { value: '16:10', label: '16:10', ratio: 16 / 10 },
        { value: '16:9', label: '16:9', ratio: 16 / 9 },
        { value: '2:1', label: '2:1', ratio: 2 },
        { value: '21:9', label: '21:9', ratio: 21 / 9 },
    ];

    originOptions = [
        { value: 'top_left', label: 'Top Left', icon: '↖' },
        { value: 'top_center', label: 'Top Center', icon: '↑' },
        { value: 'top_right', label: 'Top Right', icon: '↗' },
        { value: 'center_left', label: 'Center Left', icon: '←' },
        { value: 'center', label: 'Center', icon: '·' },
        { value: 'center_right', label: 'Center Right', icon: '→' },
        { value: 'bottom_left', label: 'Bottom Left', icon: '↙' },
        { value: 'bottom_center', label: 'Bottom Center', icon: '↓' },
        { value: 'bottom_right', label: 'Bottom Right', icon: '↘' },
    ];

    ngOnInit() {
        const itm = this.item();
        if (itm) {
            this.originalTargetW = itm.target_width || itm.width;
            this.originalTargetH = itm.target_height || itm.height;
            this.effectiveTargetW.set(this.originalTargetW);
            this.effectiveTargetH.set(this.originalTargetH);

            // Start centered
            this.freeformX.set(Math.max(0, Math.floor((itm.width - this.originalTargetW) / 2)));
            this.freeformY.set(Math.max(0, Math.floor((itm.height - this.originalTargetH) / 2)));

            // If no analysis-computed targets exist (target == source), auto-compute
            // proper 32px-snapped targets from the image's own AR
            const hasComputedTargets = itm.target_width && itm.target_height
                && (itm.target_width !== itm.width || itm.target_height !== itm.height);
            if (!hasComputedTargets && itm.width > 0 && itm.height > 0) {
                const ar = itm.width / itm.height;
                this.datasetService.calcCropTargets(
                    this.datasetName(), itm.width, itm.height, ar
                ).subscribe(result => {
                    this.originalTargetW = result.target_width;
                    this.originalTargetH = result.target_height;
                    this.effectiveTargetW.set(result.target_width);
                    this.effectiveTargetH.set(result.target_height);
                    this.freeformX.set(Math.max(0, Math.floor((itm.width - result.target_width) / 2)));
                    this.freeformY.set(Math.max(0, Math.floor((itm.height - result.target_height) / 2)));
                });
            }
        }

        // Global mouseup to handle drag ending outside the container
        document.addEventListener('mouseup', this.boundGlobalMouseUp);
    }

    ngOnDestroy() {
        document.removeEventListener('mouseup', this.boundGlobalMouseUp);
    }

    @HostListener('document:keydown.escape')
    onEscKey() {
        if (!this.isCropping()) {
            this.close.emit();
        }
    }

    selectedOriginLabel = computed(() => {
        return this.originOptions.find(o => o.value === this.selectedOrigin())?.label || '';
    });

    effectiveAR = computed(() => {
        const w = this.effectiveTargetW();
        const h = this.effectiveTargetH();
        if (!w || !h) return '—';
        const ar = w / h;
        for (const p of this.arPresets) {
            if (p.ratio > 0 && Math.abs(ar - p.ratio) < 0.02) return p.value;
            if (p.ratio > 0 && Math.abs(1 / ar - p.ratio) < 0.02) return p.value;
        }
        return `${ar.toFixed(2)}:1`;
    });

    needsCrop = computed(() => {
        const itm = this.item();
        if (!itm) return false;
        return itm.width !== this.effectiveTargetW() || itm.height !== this.effectiveTargetH();
    });

    hasOverlay = computed(() => {
        const itm = this.item();
        return itm && this.renderedW() > 0;
    });

    cropDeltaW = computed(() => {
        const itm = this.item();
        return itm ? Math.abs(itm.width - this.effectiveTargetW()) : 0;
    });

    cropDeltaH = computed(() => {
        const itm = this.item();
        return itm ? Math.abs(itm.height - this.effectiveTargetH()) : 0;
    });

    /**
     * Compute the crop rectangle in rendered pixel coordinates from freeform position.
     */
    cropRect = computed(() => {
        const itm = this.item();
        const rW = this.renderedW();
        const rH = this.renderedH();

        if (!itm || rW === 0 || rH === 0) return { top: 0, left: 0, width: 0, height: 0 };

        const natW = itm.width;
        const natH = itm.height;

        // Scale factor from natural to rendered
        const sx = rW / natW;
        const sy = rH / natH;

        // Crop window in rendered px from freeform position
        const cropW = this.effectiveTargetW() * sx;
        const cropH = this.effectiveTargetH() * sy;
        const cropLeft = this.freeformX() * sx;
        const cropTop = this.freeformY() * sy;

        return {
            top: Math.max(0, cropTop),
            left: Math.max(0, cropLeft),
            width: Math.min(cropW, rW),
            height: Math.min(cropH, rH),
        };
    });

    // ── Interaction Handlers ───────────────────────────────────────────

    onCropMouseDown(event: MouseEvent, mode: string) {
        event.preventDefault();
        event.stopPropagation();

        this.interactionMode = mode as any;
        this.dragStartX = event.clientX;
        this.dragStartY = event.clientY;
        this.dragStartFreeformX = this.freeformX();
        this.dragStartFreeformY = this.freeformY();
        this.dragStartTargetW = this.effectiveTargetW();
        this.dragStartTargetH = this.effectiveTargetH();

        if (mode === 'move') {
            this.isDragging = true;
        } else {
            this.isResizing = true;
        }
    }

    onMouseMove(event: MouseEvent) {
        if (!this.isDragging && !this.isResizing) return;

        const itm = this.item();
        if (!itm) return;

        const rW = this.renderedW();
        const rH = this.renderedH();
        const natW = itm.width;
        const natH = itm.height;

        // Delta in rendered pixels
        const dx = event.clientX - this.dragStartX;
        const dy = event.clientY - this.dragStartY;

        // Convert to natural pixels
        const natDx = dx * (natW / rW);
        const natDy = dy * (natH / rH);

        if (this.isDragging) {
            // Move: update freeform position, clamped to bounds
            const tW = this.effectiveTargetW();
            const tH = this.effectiveTargetH();
            const newX = Math.round(Math.max(0, Math.min(natW - tW, this.dragStartFreeformX + natDx)));
            const newY = Math.round(Math.max(0, Math.min(natH - tH, this.dragStartFreeformY + natDy)));
            this.freeformX.set(newX);
            this.freeformY.set(newY);

            // Clear the origin highlight since we're freeform now
            this.selectedOrigin.set('');
        } else if (this.isResizing) {
            this.handleResize(natDx, natDy, natW, natH);
        }
    }

    onMouseUp() {
        this.isDragging = false;
        this.isResizing = false;
    }

    private handleResize(natDx: number, natDy: number, natW: number, natH: number) {
        const ar = this.dragStartTargetW / this.dragStartTargetH;
        const orientation = ar >= 1 ? 'landscape' : 'portrait';

        // Determine scale delta based on which corner is being dragged
        let scaleDelta = 0;
        switch (this.interactionMode) {
            case 'resize-br':
                // Bottom-right: growing means positive dx or dy
                scaleDelta = (Math.abs(natDx) > Math.abs(natDy)) ? natDx : natDy;
                break;
            case 'resize-bl':
                scaleDelta = (Math.abs(natDx) > Math.abs(natDy)) ? -natDx : natDy;
                break;
            case 'resize-tr':
                scaleDelta = (Math.abs(natDx) > Math.abs(natDy)) ? natDx : -natDy;
                break;
            case 'resize-tl':
                scaleDelta = (Math.abs(natDx) > Math.abs(natDy)) ? -natDx : -natDy;
                break;
        }

        // Apply scale to the long side, snapping to 32px
        const origLongSide = Math.max(this.dragStartTargetW, this.dragStartTargetH);
        let newLongSide = this.closest32(origLongSide + scaleDelta);

        // Minimum meaningful size
        newLongSide = Math.max(64, newLongSide);

        // Calculate new target dims from snapped long side
        let [newW, newH] = this.calculateTargetDims(newLongSide, ar, orientation);

        // Ensure target fits within image
        while (newW > natW || newH > natH) {
            newLongSide -= 32;
            if (newLongSide <= 0) {
                newW = this.dragStartTargetW;
                newH = this.dragStartTargetH;
                break;
            }
            [newW, newH] = this.calculateTargetDims(newLongSide, ar, orientation);
        }

        // Adjust position based on corner, keeping the opposite corner anchored
        let newX = this.dragStartFreeformX;
        let newY = this.dragStartFreeformY;

        const deltaW = newW - this.dragStartTargetW;
        const deltaH = newH - this.dragStartTargetH;

        switch (this.interactionMode) {
            case 'resize-tl':
                newX = this.dragStartFreeformX - deltaW;
                newY = this.dragStartFreeformY - deltaH;
                break;
            case 'resize-tr':
                newY = this.dragStartFreeformY - deltaH;
                break;
            case 'resize-bl':
                newX = this.dragStartFreeformX - deltaW;
                break;
            case 'resize-br':
                // Anchor is top-left, no position change needed
                break;
        }

        // Clamp position
        newX = Math.max(0, Math.min(natW - newW, Math.round(newX)));
        newY = Math.max(0, Math.min(natH - newH, Math.round(newY)));

        this.effectiveTargetW.set(newW);
        this.effectiveTargetH.set(newH);
        this.freeformX.set(newX);
        this.freeformY.set(newY);
        this.selectedOrigin.set('');
    }

    // ── Snapping & Calculation ─────────────────────────────────────────

    private closest32(val: number): number {
        return Math.max(32, Math.round(val / 32) * 32);
    }

    private calculateTargetDims(longSide: number, ar: number, orientation: string): [number, number] {
        const targetLong = this.closest32(longSide);
        if (orientation === 'portrait') {
            const rawShort = targetLong * ar;
            return [this.closest32(rawShort), targetLong];
        } else {
            const rawShort = targetLong / ar;
            return [targetLong, this.closest32(rawShort)];
        }
    }

    /**
     * Snap the crop to one of 9 origin positions (quick preset).
     */
    snapToOrigin(origin: string) {
        const itm = this.item();
        if (!itm) return;

        this.selectedOrigin.set(origin);
        const tW = this.effectiveTargetW();
        const tH = this.effectiveTargetH();
        const maxX = itm.width - tW;
        const maxY = itm.height - tH;

        let x = maxX / 2;
        let y = maxY / 2;

        if (origin.includes('left')) x = 0;
        if (origin.includes('right')) x = maxX;
        if (origin.includes('top')) y = 0;
        if (origin.includes('bottom')) y = maxY;

        this.freeformX.set(Math.max(0, Math.round(x)));
        this.freeformY.set(Math.max(0, Math.round(y)));
    }

    /**
     * Recalculate crop targets when AR changes.
     */
    onARChange(arValue: string) {
        const itm = this.item();
        if (!itm) return;

        if (arValue === 'auto') {
            this.effectiveTargetW.set(this.originalTargetW);
            this.effectiveTargetH.set(this.originalTargetH);
            // Re-center
            this.freeformX.set(Math.max(0, Math.floor((itm.width - this.originalTargetW) / 2)));
            this.freeformY.set(Math.max(0, Math.floor((itm.height - this.originalTargetH) / 2)));
            this.selectedOrigin.set('center');
            return;
        }

        const preset = this.arPresets.find(p => p.value === arValue);
        if (!preset || !preset.ratio) return;

        this.datasetService.calcCropTargets(
            this.datasetName(),
            itm.width,
            itm.height,
            preset.ratio
        ).subscribe(result => {
            this.effectiveTargetW.set(result.target_width);
            this.effectiveTargetH.set(result.target_height);

            // Clamp position to new dimensions
            const maxX = itm.width - result.target_width;
            const maxY = itm.height - result.target_height;
            this.freeformX.set(Math.max(0, Math.min(this.freeformX(), maxX)));
            this.freeformY.set(Math.max(0, Math.min(this.freeformY(), maxY)));
            this.selectedOrigin.set('center');
            this.snapToOrigin('center');
        });
    }

    onImageLoad(event: Event) {
        const img = event.target as HTMLImageElement;
        this.renderedW.set(img.clientWidth);
        this.renderedH.set(img.clientHeight);
    }

    applyCrop() {
        const itm = this.item();
        if (!itm) return;
        this.isCropping.set(true);
        this.datasetService.cropImage(
            this.datasetName(),
            itm.path,
            this.effectiveTargetW(),
            this.effectiveTargetH(),
            this.selectedOrigin() || 'center',
            this.freeformX(),
            this.freeformY(),
        ).subscribe({
            next: () => {
                this.isCropping.set(false);
                this.cropped.emit(itm);
                this.close.emit();
            },
            error: (err: any) => {
                this.isCropping.set(false);
                console.error('Crop failed:', err);
            }
        });
    }

    getMediaUrl(relativePath: string): string {
        return `${this.mediaBaseUrl()}/${encodeURIComponent(this.datasetName())}/${encodeURIComponent(relativePath)}?t=${this.lastUpdateTime()}`;
    }
}
