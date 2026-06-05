import {
    ChangeDetectionStrategy,
    Component,
    DestroyRef,
    ElementRef,
    HostListener,
    OnDestroy,
    OnInit,
    computed,
    inject,
    signal,
    viewChild,
} from '@angular/core';
import { IcoComponent } from '../../icons/ico.component';
import { OverlayStore } from '../../state/overlay.store';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';
import { MediaItemStore } from '../../state/media-item.store';

interface CropPreviewData {
    datasetId?: string;
    datasetName?: string;
    /** Path of the source image (relative to dataset root). */
    path?: string;
    /** Source dimensions. */
    width?: number;
    height?: number;
    /** Target bucket dimensions from analysis (used as the initial "Auto" target). */
    target_width?: number;
    target_height?: number;
    /** Majority aspect ratio (W/H) of the image's orientation group. When the
     *  analysis target is missing, "Auto" crops to this AR instead of the
     *  image's own (near-native) AR — so it lands on the group's bucket. */
    majority_ar?: number;
}

interface ARPreset {
    value: string;
    label: string;
    ratio: number;
}

const AR_PRESETS: ReadonlyArray<ARPreset> = [
    { value: 'auto',  label: 'Auto (majority AR)',   ratio: 0 },
    { value: '1:1',   label: '1:1 — Square',         ratio: 1 },
    { value: '4:3',   label: '4:3',                  ratio: 4 / 3 },
    { value: '3:2',   label: '3:2',                  ratio: 3 / 2 },
    { value: '16:10', label: '16:10',                ratio: 16 / 10 },
    { value: '16:9',  label: '16:9',                 ratio: 16 / 9 },
    { value: '2:1',   label: '2:1',                  ratio: 2 },
    { value: '21:9',  label: '21:9',                 ratio: 21 / 9 },
];

interface OriginOption {
    value: string;
    label: string;
    icon: string;
}

const ORIGIN_OPTIONS: ReadonlyArray<OriginOption> = [
    { value: 'top_left',      label: 'Top Left',      icon: '↖' },
    { value: 'top_center',    label: 'Top Center',    icon: '↑' },
    { value: 'top_right',     label: 'Top Right',     icon: '↗' },
    { value: 'center_left',   label: 'Center Left',   icon: '←' },
    { value: 'center',        label: 'Center',        icon: '·' },
    { value: 'center_right',  label: 'Center Right',  icon: '→' },
    { value: 'bottom_left',   label: 'Bottom Left',   icon: '↙' },
    { value: 'bottom_center', label: 'Bottom Center', icon: '↓' },
    { value: 'bottom_right',  label: 'Bottom Right',  icon: '↘' },
];

type InteractionMode = 'move' | 'resize-tl' | 'resize-tr' | 'resize-bl' | 'resize-br';

/**
 * Crop preview modal — feature-complete port of the legacy
 * viewer-crop-preview-modal, restyled to the new design tokens.
 *
 * Capabilities:
 *  - Live image render via the API thumbnail/media endpoint.
 *  - Drag-to-move on the crop window, plus corner-handle resize that
 *    snaps the long side to multiples of 32px (training-friendly).
 *  - AR preset selector (1:1, 4:3, 3:2, 16:10, 16:9, 2:1, 21:9, Auto).
 *  - 9-position quick-origin grid (top-left … bottom-right).
 *  - Crop region info panel (position, size, Δ width/height vs source).
 *  - "Apply Crop" wired to ``DatasetService.cropImage`` with the actual
 *    crop_x/crop_y/target dims.
 *
 * Legacy bug fixed: switching the AR previously trusted the backend's
 * `calcCropTargets` response, which can fall back to raw source dims
 * when the requested AR doesn't fit any 32-aligned target. The frontend
 * now re-snaps both dimensions client-side with {@link closest32}, so
 * the crop window is always training-compatible regardless of how the
 * backend resolves the AR.
 */
@Component({
    selector: 'app-modal-crop-preview',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head cp-head">
            <div>
                <div class="eyebrow brand">CROP PREVIEW</div>
                <div class="modal-title mono">{{ data.path ?? '—' }}</div>
            </div>
            <div class="cp-dims-head">
                <span class="mono">{{ srcW }}×{{ srcH }}</span>
                <app-ico name="ArrowRight" [size]="14" class="cp-arrow"/>
                <span class="mono cp-target">{{ effectiveTargetW() }}×{{ effectiveTargetH() }}</span>
                <button class="icon-btn cp-close" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
            </div>
        </div>

        <div class="modal-body cp-body">
            <!-- Stage holds the canvas centered; the canvas shrink-wraps the
                 image so dim bands and the crop window position image-relative,
                 not stage-relative (which would mis-center on letterboxing). -->
            <div class="cp-stage"
                 (mousemove)="onMouseMove($event)"
                 (mouseup)="onMouseUp()"
                 (mouseleave)="onMouseUp()">
                @if (imageUrl(); as url) {
                    <div class="cp-canvas" [style.aspectRatio]="srcAR">
                        <img #imageEl
                             [src]="url"
                             class="cp-image"
                             draggable="false"
                             (load)="onImageLoad($event)"
                             alt=""/>

                        @if (hasOverlay()) {
                            <!-- Dark dimming outside the crop window (4 bands) -->
                            <div class="cp-dim cp-dim-top"
                                 [style.height.px]="cropRect().top"></div>
                            <div class="cp-dim cp-dim-bot"
                                 [style.height.px]="renderedH() - cropRect().top - cropRect().height"></div>
                            <div class="cp-dim cp-dim-left"
                                 [style.top.px]="cropRect().top"
                                 [style.width.px]="cropRect().left"
                                 [style.height.px]="cropRect().height"></div>
                            <div class="cp-dim cp-dim-right"
                                 [style.top.px]="cropRect().top"
                                 [style.width.px]="renderedW() - cropRect().left - cropRect().width"
                                 [style.height.px]="cropRect().height"></div>

                            <!-- Crop window: draggable + 4 resize handles + rule-of-thirds + crosshair -->
                            <div class="cp-window"
                                 [class.cp-moving]="!isResizing"
                                 [style.top.px]="cropRect().top"
                                 [style.left.px]="cropRect().left"
                                 [style.width.px]="cropRect().width"
                                 [style.height.px]="cropRect().height"
                                 (mousedown)="onCropMouseDown($event, 'move')">
                                <div class="cp-thirds">
                                    <span class="cp-line v" style="left:33.33%"></span>
                                    <span class="cp-line v" style="left:66.66%"></span>
                                    <span class="cp-line h" style="top:33.33%"></span>
                                    <span class="cp-line h" style="top:66.66%"></span>
                                </div>
                                <span class="cp-handle tl" (mousedown)="onCropMouseDown($event, 'resize-tl')"></span>
                                <span class="cp-handle tr" (mousedown)="onCropMouseDown($event, 'resize-tr')"></span>
                                <span class="cp-handle bl" (mousedown)="onCropMouseDown($event, 'resize-bl')"></span>
                                <span class="cp-handle br" (mousedown)="onCropMouseDown($event, 'resize-br')"></span>
                                <div class="cp-cross">
                                    <span class="cp-cross-h"></span>
                                    <span class="cp-cross-v"></span>
                                </div>
                            </div>
                        }
                    </div>
                } @else {
                    <div class="cp-loading">No image — open from a dataset card.</div>
                }
            </div>

            <!-- Right panel: AR / origin grid / region info -->
            <div class="cp-panel">
                <div class="card cp-card">
                    <div class="cp-card-title">Target Aspect Ratio</div>
                    <select class="input cp-select"
                            [value]="selectedAR()"
                            (change)="onARSelectChange($event)">
                        @for (ar of arPresets; track ar.value) {
                            <option [value]="ar.value">{{ ar.label }}</option>
                        }
                    </select>
                    <div class="cp-sub mono">
                        {{ effectiveTargetW() }}×{{ effectiveTargetH() }}
                        <span class="cp-subtle">({{ effectiveAR() }})</span>
                    </div>
                </div>

                <div class="card cp-card">
                    <div class="cp-card-title">Quick Position</div>
                    <div class="cp-origin-grid">
                        @for (opt of originOptions; track opt.value) {
                            <button type="button"
                                    class="cp-origin-btn"
                                    [class.active]="selectedOrigin() === opt.value"
                                    [title]="opt.label"
                                    (click)="snapToOrigin(opt.value)">{{ opt.icon }}</button>
                        }
                    </div>
                    <div class="cp-sub cp-sub-center">{{ selectedOriginLabel() }}</div>
                </div>

                <div class="card cp-card">
                    <div class="cp-card-title">Crop Region</div>
                    <div class="cp-info">
                        <span class="cp-info-k">Position</span>
                        <span class="cp-info-v mono">{{ freeformX() }}, {{ freeformY() }}</span>
                        <span class="cp-info-k">Size</span>
                        <span class="cp-info-v mono">{{ effectiveTargetW() }}×{{ effectiveTargetH() }}</span>
                        <span class="cp-info-k">Δ Width</span>
                        <span class="cp-info-v mono">{{ cropDeltaW() }} px</span>
                        <span class="cp-info-k">Δ Height</span>
                        <span class="cp-info-v mono">{{ cropDeltaH() }} px</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="modal-foot cp-foot">
            <div class="cp-hint">
                <kbd class="cp-kbd">ESC</kbd> close ·
                Drag the window to move · drag a corner to resize (snaps to 32 px)
            </div>
            <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
            <button class="btn primary" type="button"
                    [disabled]="isCropping() || !needsCrop()"
                    (click)="applyCrop()">
                @if (isCropping()) {
                    Cropping…
                } @else {
                    <app-ico name="Check" [size]="12"/> Apply Crop
                }
            </button>
        </div>
    `,
    styles: [`
        .cp-head {
            display: flex; align-items: center; justify-content: space-between;
            gap: 14px;
        }
        .modal-title { font-size: 13px; font-weight: 600; margin-top: 2px;
            color: var(--color-text-secondary);
            max-width: 320px;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        }
        .eyebrow {
            font-size: 10px; font-weight: 700; letter-spacing: 0.14em;
            text-transform: uppercase; color: var(--color-text-muted);
        }
        .eyebrow.brand { color: var(--color-brand-light); }
        .cp-dims-head {
            display: flex; align-items: center; gap: 6px;
            font-size: 13px; font-weight: 700;
            color: var(--color-text-secondary);
        }
        .cp-dims-head .cp-target { color: var(--color-brand-light); }
        .cp-arrow { color: var(--color-text-disabled); }
        .cp-close { margin-left: 12px; }

        .cp-body {
            display: flex; gap: 14px;
            padding: 14px 18px !important;
            min-height: 0;
        }

        .cp-stage {
            flex: 1;
            min-width: 0;
            background: var(--color-base);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
            display: flex; align-items: center; justify-content: center;
            overflow: hidden;
            min-height: 480px;
            user-select: none;
            padding: 8px;
        }
        /* Canvas takes the image's aspect ratio and grows to fill whichever
           stage axis binds first — so a landscape image stretches wide and a
           portrait image stretches tall, never letterboxing. Overlay
           coordinates remain image-relative because the canvas IS the image. */
        .cp-canvas {
            position: relative;
            display: block;
            max-width: 100%;
            /* Definite viewport-based cap (not % — the modal uses max-height,
               so a percentage parent height is indefinite and gets ignored).
               Keeps tall portrait images shrunk-to-fit and never pushes the
               footer (Apply Crop) past the 92vh modal. Offset ≈ head + foot +
               body/stage padding. */
            max-height: calc(92vh - 188px);
        }
        .cp-image {
            display: block;
            width: 100%;
            height: 100%;
            object-fit: cover;
            pointer-events: none;
        }
        .cp-loading {
            color: var(--color-text-muted);
            font-size: 12px;
            padding: 32px;
        }

        .cp-dim {
            position: absolute;
            background: oklch(0.08 0.01 265 / 0.72);
            pointer-events: none;
        }
        .cp-dim-top   { left: 0; right: 0; top: 0; }
        .cp-dim-bot   { left: 0; right: 0; bottom: 0; }
        .cp-dim-left  { left: 0; }
        .cp-dim-right { right: 0; }

        .cp-window {
            position: absolute;
            border: 1.5px solid var(--color-brand);
            box-shadow:
                0 0 0 1px oklch(0 0 0 / 0.45),
                0 0 16px oklch(0.68 0.13 55 / 0.35);
            cursor: default;
        }
        .cp-window.cp-moving { cursor: move; }

        .cp-handle {
            position: absolute;
            width: 12px; height: 12px;
            background: var(--color-brand);
            border: 1.5px solid white;
            border-radius: 2px;
            z-index: 5;
        }
        .cp-handle.tl { top: -6px; left: -6px;    cursor: nw-resize; }
        .cp-handle.tr { top: -6px; right: -6px;   cursor: ne-resize; }
        .cp-handle.bl { bottom: -6px; left: -6px; cursor: sw-resize; }
        .cp-handle.br { bottom: -6px; right: -6px;cursor: se-resize; }

        .cp-thirds { position: absolute; inset: 0; pointer-events: none; opacity: 0.45; }
        .cp-line { position: absolute; background: white; }
        .cp-line.v { top: 0; bottom: 0; width: 1px; }
        .cp-line.h { left: 0; right: 0; height: 1px; }

        .cp-cross {
            position: absolute; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            width: 22px; height: 22px;
            pointer-events: none;
        }
        .cp-cross-h { position: absolute; left: 0; right: 0; top: 50%;
            height: 1px; background: oklch(0.68 0.13 55 / 0.6); }
        .cp-cross-v { position: absolute; top: 0; bottom: 0; left: 50%;
            width: 1px; background: oklch(0.68 0.13 55 / 0.6); }

        /* Right panel */
        .cp-panel {
            display: flex; flex-direction: column; gap: 12px;
            width: 220px; flex-shrink: 0;
        }
        .cp-card { padding: 12px !important; margin: 0 !important; }
        .cp-card-title {
            font-size: 10px; font-weight: 700;
            letter-spacing: 0.10em; text-transform: uppercase;
            color: var(--color-text-subtle);
            margin-bottom: 8px;
        }
        .cp-select {
            width: 100%;
            font-size: 12px;
            padding: 6px 10px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
            color: var(--color-text-primary);
            outline: none;
        }
        .cp-select:focus { border-color: var(--color-brand); }
        .cp-sub {
            font-size: 10.5px;
            color: var(--color-text-muted);
            margin-top: 8px;
        }
        .cp-sub-center { text-align: center; }
        .cp-subtle { color: var(--color-text-disabled); margin-left: 4px; }

        .cp-origin-grid {
            display: grid; grid-template-columns: repeat(3, 1fr);
            gap: 4px; width: fit-content; margin: 0 auto;
        }
        .cp-origin-btn {
            width: 36px; height: 36px;
            border-radius: var(--radius-theme-md);
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            color: var(--color-text-muted);
            font-size: 15px; font-weight: 700;
            cursor: pointer;
            transition: 100ms;
            display: inline-flex; align-items: center; justify-content: center;
        }
        .cp-origin-btn:hover { color: var(--color-text-primary); background: var(--color-surface-high); }
        .cp-origin-btn.active {
            background: var(--color-brand); color: white;
            border-color: var(--color-brand);
            box-shadow: 0 2px 6px oklch(0.68 0.13 55 / 0.4);
        }

        .cp-info {
            display: grid;
            grid-template-columns: 1fr auto;
            row-gap: 4px;
            font-size: 11.5px;
        }
        .cp-info-k { color: var(--color-text-subtle); }
        .cp-info-v { color: var(--color-brand-light); font-weight: 700; }

        .cp-foot {
            display: flex; align-items: center; gap: 10px;
        }
        .cp-hint {
            margin-right: auto;
            font-size: 11px; color: var(--color-text-muted);
            display: flex; align-items: center; gap: 6px;
        }
        .cp-kbd {
            font-family: var(--font-mono);
            font-size: 10px;
            padding: 1px 5px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: 3px;
            color: var(--color-text-secondary);
        }

        .btn.primary { display: inline-flex; align-items: center; gap: 6px; }
        .btn.primary:disabled { opacity: 0.4; cursor: not-allowed; }
    `],
})
export class CropPreviewModalComponent implements OnInit, OnDestroy {
    protected overlay = inject(OverlayStore);
    private rtc = inject(RuntimeConfigService);
    private datasetsApi = inject(DatasetService);
    private toast = inject(ToastService);
    private destroyRef = inject(DestroyRef);
    private mediaItems = inject(MediaItemStore);

    private imageEl = viewChild<ElementRef<HTMLImageElement>>('imageEl');

    protected data: CropPreviewData = (this.overlay.topModal()?.data as CropPreviewData) ?? {};

    protected readonly arPresets = AR_PRESETS;
    protected readonly originOptions = ORIGIN_OPTIONS;

    protected readonly srcW = this.data.width ?? 0;
    protected readonly srcH = this.data.height ?? 0;
    /** CSS aspect-ratio string for the canvas — drives the stretch-to-fit. */
    protected readonly srcAR = `${this.srcW || 16} / ${this.srcH || 9}`;

    /** Original analysis target (used by the "Auto" preset). */
    private originalTargetW = 0;
    private originalTargetH = 0;

    protected effectiveTargetW = signal<number>(0);
    protected effectiveTargetH = signal<number>(0);

    protected freeformX = signal<number>(0);
    protected freeformY = signal<number>(0);

    protected selectedOrigin = signal<string>('center');
    protected selectedAR = signal<string>('auto');
    protected isCropping = signal<boolean>(false);

    protected renderedW = signal<number>(0);
    protected renderedH = signal<number>(0);

    private isDragging = false;
    protected isResizing = false;
    private interactionMode: InteractionMode = 'move';
    private dragStartX = 0;
    private dragStartY = 0;
    private dragStartFreeformX = 0;
    private dragStartFreeformY = 0;
    private dragStartTargetW = 0;
    private dragStartTargetH = 0;

    private boundGlobalMouseUp = this.onMouseUp.bind(this);
    private resizeObserver: ResizeObserver | null = null;

    ngOnInit(): void {
        // Seed targets from analysis (falls back to source dims when missing).
        const tW = this.data.target_width ?? this.srcW;
        const tH = this.data.target_height ?? this.srcH;
        this.originalTargetW = tW;
        this.originalTargetH = tH;
        this.effectiveTargetW.set(tW);
        this.effectiveTargetH.set(tH);

        // Center the initial crop window.
        this.freeformX.set(Math.max(0, Math.floor((this.srcW - tW) / 2)));
        this.freeformY.set(Math.max(0, Math.floor((this.srcH - tH) / 2)));

        // If the analysis didn't produce a useful target (target == source —
        // typical for un-harmonized datasets), ask the backend for one. Prefer
        // the orientation group's majority AR so "Auto" lands on the group's
        // bucket (e.g. 16:9); fall back to the image's own AR only when the
        // majority AR is unknown. Snap the response to 32px ourselves.
        const hasAnalysisTarget = tW !== this.srcW || tH !== this.srcH;
        if (!hasAnalysisTarget && this.srcW > 0 && this.srcH > 0 && this.data.datasetName) {
            const ar = this.data.majority_ar && this.data.majority_ar > 0
                ? this.data.majority_ar
                : this.srcW / this.srcH;
            this.datasetsApi
                .calcCropTargets(this.data.datasetName, this.srcW, this.srcH, ar)
                .subscribe(res => this.applyTargetsFromBackend(res.target_width, res.target_height));
        }

        document.addEventListener('mouseup', this.boundGlobalMouseUp);
        this.destroyRef.onDestroy(() => {
            document.removeEventListener('mouseup', this.boundGlobalMouseUp);
        });
    }

    ngOnDestroy(): void {
        // Belt-and-suspenders — destroyRef already removes the listener.
    }

    @HostListener('document:keydown.escape')
    onEscKey(): void {
        if (!this.isCropping()) this.overlay.closeModal();
    }

    protected imageUrl = computed<string | null>(() => {
        const name = this.data.datasetName;
        const path = this.data.path;
        if (!name || !path) return null;
        // Use the /media endpoint with the full-res image (the /thumbnail
        // endpoint serves a 256-px webp which is too coarse for crop work).
        return `${this.rtc.apiUrl}/datasets/${encodeURIComponent(name)}/media?image_rel_path=${encodeURIComponent(path)}`;
    });

    protected selectedOriginLabel = computed(() =>
        ORIGIN_OPTIONS.find(o => o.value === this.selectedOrigin())?.label ?? 'Freeform'
    );

    protected effectiveAR = computed(() => {
        const w = this.effectiveTargetW();
        const h = this.effectiveTargetH();
        if (!w || !h) return '—';
        const ar = w / h;
        for (const p of AR_PRESETS) {
            if (p.ratio > 0 && Math.abs(ar - p.ratio) < 0.02) return p.value;
            if (p.ratio > 0 && Math.abs(1 / ar - p.ratio) < 0.02) return p.value;
        }
        return `${ar.toFixed(2)}:1`;
    });

    protected needsCrop = computed(() => {
        return this.srcW !== this.effectiveTargetW() || this.srcH !== this.effectiveTargetH();
    });

    protected hasOverlay = computed(() => this.renderedW() > 0 && this.srcW > 0);

    protected cropDeltaW = computed(() => Math.abs(this.srcW - this.effectiveTargetW()));
    protected cropDeltaH = computed(() => Math.abs(this.srcH - this.effectiveTargetH()));

    /** Crop rectangle in rendered (on-screen) pixels — projected from natural px. */
    protected cropRect = computed(() => {
        const rW = this.renderedW();
        const rH = this.renderedH();
        if (this.srcW === 0 || this.srcH === 0 || rW === 0 || rH === 0) {
            return { top: 0, left: 0, width: 0, height: 0 };
        }
        const sx = rW / this.srcW;
        const sy = rH / this.srcH;
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

    // ── Interaction handlers ──────────────────────────────────────────

    protected onCropMouseDown(event: MouseEvent, mode: InteractionMode): void {
        event.preventDefault();
        event.stopPropagation();
        this.interactionMode = mode;
        this.dragStartX = event.clientX;
        this.dragStartY = event.clientY;
        this.dragStartFreeformX = this.freeformX();
        this.dragStartFreeformY = this.freeformY();
        this.dragStartTargetW = this.effectiveTargetW();
        this.dragStartTargetH = this.effectiveTargetH();
        if (mode === 'move') this.isDragging = true;
        else this.isResizing = true;
    }

    protected onMouseMove(event: MouseEvent): void {
        if (!this.isDragging && !this.isResizing) return;
        if (this.srcW === 0 || this.srcH === 0) return;

        const rW = this.renderedW();
        const rH = this.renderedH();
        const dx = event.clientX - this.dragStartX;
        const dy = event.clientY - this.dragStartY;
        const natDx = dx * (this.srcW / rW);
        const natDy = dy * (this.srcH / rH);

        if (this.isDragging) {
            const tW = this.effectiveTargetW();
            const tH = this.effectiveTargetH();
            const newX = Math.round(Math.max(0, Math.min(this.srcW - tW, this.dragStartFreeformX + natDx)));
            const newY = Math.round(Math.max(0, Math.min(this.srcH - tH, this.dragStartFreeformY + natDy)));
            this.freeformX.set(newX);
            this.freeformY.set(newY);
            this.selectedOrigin.set('');
        } else if (this.isResizing) {
            this.handleResize(natDx, natDy);
        }
    }

    protected onMouseUp(): void {
        this.isDragging = false;
        this.isResizing = false;
    }

    private handleResize(natDx: number, natDy: number): void {
        const ar = this.dragStartTargetW / this.dragStartTargetH;
        const orientation = ar >= 1 ? 'landscape' : 'portrait';

        let scaleDelta = 0;
        switch (this.interactionMode) {
            case 'resize-br': scaleDelta = Math.abs(natDx) > Math.abs(natDy) ?  natDx :  natDy; break;
            case 'resize-bl': scaleDelta = Math.abs(natDx) > Math.abs(natDy) ? -natDx :  natDy; break;
            case 'resize-tr': scaleDelta = Math.abs(natDx) > Math.abs(natDy) ?  natDx : -natDy; break;
            case 'resize-tl': scaleDelta = Math.abs(natDx) > Math.abs(natDy) ? -natDx : -natDy; break;
        }

        const origLongSide = Math.max(this.dragStartTargetW, this.dragStartTargetH);
        let newLongSide = Math.max(64, this.closest32(origLongSide + scaleDelta));

        let [newW, newH] = this.calculateTargetDims(newLongSide, ar, orientation);
        while ((newW > this.srcW || newH > this.srcH) && newLongSide > 0) {
            newLongSide -= 32;
            [newW, newH] = newLongSide > 0
                ? this.calculateTargetDims(newLongSide, ar, orientation)
                : [this.dragStartTargetW, this.dragStartTargetH];
        }

        const deltaW = newW - this.dragStartTargetW;
        const deltaH = newH - this.dragStartTargetH;
        let newX = this.dragStartFreeformX;
        let newY = this.dragStartFreeformY;
        if (this.interactionMode === 'resize-tl') { newX -= deltaW; newY -= deltaH; }
        else if (this.interactionMode === 'resize-tr') { newY -= deltaH; }
        else if (this.interactionMode === 'resize-bl') { newX -= deltaW; }

        newX = Math.max(0, Math.min(this.srcW - newW, Math.round(newX)));
        newY = Math.max(0, Math.min(this.srcH - newH, Math.round(newY)));

        this.effectiveTargetW.set(newW);
        this.effectiveTargetH.set(newH);
        this.freeformX.set(newX);
        this.freeformY.set(newY);
        this.selectedOrigin.set('');
    }

    // ── 32px snap helpers (training-target invariant) ─────────────────

    private closest32(val: number): number {
        return Math.max(32, Math.round(val / 32) * 32);
    }

    private calculateTargetDims(longSide: number, ar: number, orientation: string): [number, number] {
        const targetLong = this.closest32(longSide);
        if (orientation === 'portrait') {
            return [this.closest32(targetLong * ar), targetLong];
        }
        return [targetLong, this.closest32(targetLong / ar)];
    }

    // ── 9-grid origin quick-position ──────────────────────────────────

    protected snapToOrigin(origin: string): void {
        this.selectedOrigin.set(origin);
        const tW = this.effectiveTargetW();
        const tH = this.effectiveTargetH();
        const maxX = this.srcW - tW;
        const maxY = this.srcH - tH;
        let x = maxX / 2;
        let y = maxY / 2;
        if (origin.includes('left'))   x = 0;
        if (origin.includes('right'))  x = maxX;
        if (origin.includes('top'))    y = 0;
        if (origin.includes('bottom')) y = maxY;
        this.freeformX.set(Math.max(0, Math.round(x)));
        this.freeformY.set(Math.max(0, Math.round(y)));
    }

    // ── AR change — with the 32px snap fix ────────────────────────────

    protected onARSelectChange(e: Event): void {
        const v = (e.target as HTMLSelectElement).value;
        this.selectedAR.set(v);

        if (v === 'auto') {
            this.effectiveTargetW.set(this.originalTargetW);
            this.effectiveTargetH.set(this.originalTargetH);
            this.freeformX.set(Math.max(0, Math.floor((this.srcW - this.originalTargetW) / 2)));
            this.freeformY.set(Math.max(0, Math.floor((this.srcH - this.originalTargetH) / 2)));
            this.selectedOrigin.set('center');
            return;
        }

        const preset = AR_PRESETS.find(p => p.value === v);
        if (!preset || !preset.ratio || !this.data.datasetName) return;

        this.datasetsApi
            .calcCropTargets(this.data.datasetName, this.srcW, this.srcH, preset.ratio)
            .subscribe(res => this.applyTargetsFromBackend(res.target_width, res.target_height));
    }

    /**
     * Apply target dims returned by the backend, **client-snapped to
     * multiples of 32px**. Fixes the legacy bug where calc-crop-targets
     * could fall back to raw source dims (when no 32-aligned target fit
     * the requested AR), leaving the crop window at a size the trainer
     * would later reject — forcing the user to re-open and crop again.
     */
    private applyTargetsFromBackend(tw: number, th: number): void {
        let snapW = this.closest32(tw);
        let snapH = this.closest32(th);
        // Never exceed the source — shrink until both fit.
        while ((snapW > this.srcW || snapH > this.srcH) && (snapW > 32 || snapH > 32)) {
            if (snapW > this.srcW) snapW = Math.max(32, snapW - 32);
            if (snapH > this.srcH) snapH = Math.max(32, snapH - 32);
        }
        this.effectiveTargetW.set(snapW);
        this.effectiveTargetH.set(snapH);
        this.freeformX.set(Math.max(0, Math.min(this.freeformX(), this.srcW - snapW)));
        this.freeformY.set(Math.max(0, Math.min(this.freeformY(), this.srcH - snapH)));
        this.selectedOrigin.set('center');
        this.snapToOrigin('center');
    }

    // ── Image loaded — capture rendered dims for crop projection ──────

    protected onImageLoad(event: Event): void {
        const img = event.target as HTMLImageElement;
        this.renderedW.set(img.clientWidth);
        this.renderedH.set(img.clientHeight);

        // Keep renderedW/H in sync when the canvas is resized (window resize,
        // panel collapse, etc.) — overlay coords are derived from these.
        if (!this.resizeObserver) {
            this.resizeObserver = new ResizeObserver(() => {
                this.renderedW.set(img.clientWidth);
                this.renderedH.set(img.clientHeight);
            });
            this.resizeObserver.observe(img);
            this.destroyRef.onDestroy(() => this.resizeObserver?.disconnect());
        }
    }

    // ── Apply crop — POST to /crop and close on success ───────────────

    protected applyCrop(): void {
        if (!this.data.datasetName || !this.data.path) return;
        this.isCropping.set(true);
        this.datasetsApi
            .cropImage(
                this.data.datasetName,
                this.data.path,
                this.effectiveTargetW(),
                this.effectiveTargetH(),
                this.selectedOrigin() || 'center',
                this.freeformX(),
                this.freeformY(),
            )
            .subscribe({
                next: () => {
                    this.isCropping.set(false);
                    // Bump the global media-cache rev so every <img> that
                    // points at the cropped file (details / browse grid /
                    // filmstrip) appends a fresh ``?t=`` and forces the
                    // browser to re-fetch. Without this the URL is
                    // unchanged, the browser doesn't even issue a
                    // revalidation request, and the user keeps seeing
                    // the pre-crop bytes — meanwhile the backend has
                    // already overwritten the source on disk, so each
                    // subsequent crop silently destroys more of the
                    // original.
                    this.mediaItems.bumpMedia();
                    this.toast.success(`Cropped ${this.data.path}`);
                    this.overlay.closeModal();
                },
                error: (err: { error?: { detail?: string }; message?: string }) => {
                    this.isCropping.set(false);
                    this.toast.error(`Crop failed: ${err?.error?.detail || err?.message || 'unknown'}`);
                },
            });
    }
}
