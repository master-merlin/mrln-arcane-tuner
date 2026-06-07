import { ChangeDetectionStrategy, Component, ElementRef, HostListener, computed, effect, inject, input, output, signal, viewChild } from '@angular/core';
import { IcoComponent } from '../../../../icons/ico.component';
import { RuntimeConfigService } from '../../../../services/runtime-config.service';
import { CanvasFooterComponent, CanvasMeta } from '../../../shared/canvas-footer.component';
import { PanZoomDirective } from '../../../shared/pan-zoom.directive';
import { buildCanvasMeta } from '../../../shared/media-meta';
import { OverlayStore, Overlay } from '../../../../state/overlay.store';
import { PipelineEditorState } from '../pipeline-editor.state';
import { PreviewPipeline } from '../preview/preview-pipeline';

/**
 * Appends a dedicated `cors=1` cache key to a media URL. The canvas pixel
 * pipeline fetches the image in CORS mode (crossOrigin) via this isolated
 * URL so its cache entry is NEVER shared with the plain (no-cors) `<img>`
 * loads used by Browse/Details. That isolation is what makes the editor
 * immune to the "no-cors body replayed for a cors request" cache poisoning
 * that otherwise breaks `getImageData`.
 */
export function withCorsParam(url: string): string {
    return url + (url.includes('?') ? '&' : '?') + 'cors=1';
}

/** Subset of `Overlay` this module needs. Inlined to keep the helpers pure. */
interface OverlayUrlInfo {
    dataset_name: string;
    overlay_file: string;
    hash?: string;
}

/**
 * Build the URL the visible `<img class="layer base">` shows beneath the
 * canvas. Prefers the rendered overlay PNG when present (so an unattached
 * editor still displays the saved result), and falls back to the original
 * otherwise. The hash query (`?h=<sha>`) busts the browser cache when the
 * backend rewrites the overlay file under the same path; `&r=<rev>` covers
 * Bake/Revert, which replaces the original on disk.
 */
export function buildDisplayUrl(
    mediaBaseUrl: string,
    datasetName: string,
    mediaFile: string,
    overlay: OverlayUrlInfo | null,
    sourceRev: number,
): string {
    if (overlay?.overlay_file) {
        const hash = overlay.hash ? `?h=${overlay.hash}` : `?h=ov`;
        const revQ = sourceRev > 0 ? `&r=${sourceRev}` : '';
        return `${mediaBaseUrl}/${encodeURIComponent(overlay.dataset_name)}/${overlay.overlay_file}${hash}${revQ}`;
    }
    const revQ = sourceRev > 0 ? `?r=${sourceRev}` : '';
    return `${mediaBaseUrl}/${encodeURIComponent(datasetName)}/${encodeURIComponent(mediaFile)}${revQ}`;
}

/**
 * Build the URL the canvas pixel pipeline reads. Always points at the
 * ORIGINAL image, even when a saved overlay PNG exists.
 *
 * Why not source the overlay PNG when it's available? The PreviewPipeline
 * applies the recipe held in the sliders on top of whatever pixels it
 * loads. The overlay PNG already has that recipe baked in, so sourcing it
 * would apply the recipe twice — once on disk, once live — producing the
 * user-reported "save → reload → values applied again" double-application
 * (e.g. HSL orange-on-top-of-orange after Save). Sourcing the original
 * keeps the editor canvas equivalent to what the backend's Save would
 * produce: `original + recipe`. Browse and Details consume the saved
 * overlay PNG, which is just the baked output of that same pipeline — so
 * the canvas and the PNG always agree at Save time.
 *
 * `sourceRev` is bumped on Bake/Revert (which rewrites or deletes the
 * original on disk), so the canvas re-fetches even if the URL would
 * otherwise be cache-stable.
 */
export function buildPixelSourceUrl(
    mediaBaseUrl: string,
    datasetName: string,
    mediaFile: string,
    sourceRev: number,
): string {
    const revQ = sourceRev > 0 ? `?r=${sourceRev}` : '';
    return `${mediaBaseUrl}/${encodeURIComponent(datasetName)}/${encodeURIComponent(mediaFile)}${revQ}`;
}

@Component({
    selector: 'app-edit-canvas',
    standalone: true,
    imports: [IcoComponent, CanvasFooterComponent, PanZoomDirective],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="stage hover-host" #stage
             appPanZoom [appPanZoomTarget]="imageStage"
             [zoom]="zoom()" (zoomChange)="zoom.set($event)">
            <button type="button" class="ab-btn"
                    [class.on]="compareOn()"
                    (click)="toggleCompare()"
                    title="Toggle A/B comparison">A/B</button>

            <button type="button" class="nav-btn left hover-show" (click)="prev.emit()" title="Previous">
                <app-ico name="ChevronLeft" [size]="18"/>
            </button>
            <button type="button" class="nav-btn right hover-show" (click)="next.emit()" title="Next">
                <app-ico name="ChevronRight" [size]="18"/>
            </button>

            <div class="image-stage" #imageStage
                 [class.compare-on]="compareOn()"
                 [style.--ab-split.%]="splitPercent()">
                <img class="layer base"
                     [src]="displayUrl()"
                     [alt]="mediaFile()"
                     loading="eager"
                     decoding="async"
                     (error)="onDisplayError()"/>
                <canvas #previewCanvas class="layer overlay" aria-hidden="true"></canvas>
                @if (compareOn()) {
                    <span class="ab-label left">BEFORE</span>
                    <span class="ab-label right">AFTER</span>
                    <div class="ab-divider"
                         (pointerdown)="onSplitPointerDown($event)"
                         (pointermove)="onSplitPointerMove($event)"
                         (pointerup)="onSplitPointerUp($event)"
                         (pointercancel)="onSplitPointerCancel($event)">
                        <div class="ab-handle">↔</div>
                    </div>
                }
                @if (preview.showSpinner()) {
                    <div class="pipeline-busy" aria-hidden="true">···</div>
                }
            </div>
        </div>

        <app-canvas-footer [meta]="meta()" [zoom]="zoom()" (zoomChange)="zoom.set($event)"
                           [fullscreen]="isFullscreen()" (toggleFullscreen)="toggleFullscreen()">
            @if (state.dirty()) {
                <span class="chip warning dirty-chip"
                      role="status"
                      data-testid="edit-dirty-chip"
                      aria-label="Unsaved adjustments — modified"
                      title="You have unsaved adjustments — Save to commit or Revert to discard">
                    <app-ico name="Edit3" [size]="10"/> Modified
                </span>
            }
        </app-canvas-footer>
    `,
    styles: [`
        :host { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
        .stage {
            flex: 1; position: relative;
            display: flex; align-items: center; justify-content: center;
            padding: 14px 16px; min-height: 0;
            background: var(--color-base);
            /* Clip the zoomed/panned image-stage at the stage edges so it reads
               as the image filling the canvas — not a rounded card with a drop
               shadow scaling up (and never spilling onto the footer). */
            overflow: hidden;
        }
        .image-stage {
            position: relative;
            max-width: 100%; max-height: 100%;
            display: inline-flex;
            border-radius: var(--radius-theme-lg);
            overflow: hidden;
        }
        .layer { display: block; max-width: 100%; max-height: 100%; object-fit: contain; }
        .layer.overlay { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
        .image-stage.compare-on .layer.overlay {
            clip-path: inset(0 0 0 var(--ab-split, 50%));
        }
        .nav-btn {
            position: absolute; top: 50%; transform: translateY(-50%);
            z-index: 5; width: 40px; height: 40px;
            border-radius: 999px;
            background: oklch(0.10 0.01 265 / 0.65);
            color: #fff;
            border: 1px solid oklch(0.95 0 0 / 0.10);
            backdrop-filter: blur(6px);
            box-shadow: 0 2px 10px oklch(0 0 0 / 0.4);
            cursor: pointer;
            display: flex; align-items: center; justify-content: center;
            opacity: 0; transition: opacity 120ms;
        }
        .nav-btn.left  { left: 18px; }
        .nav-btn.right { right: 18px; }
        .hover-host:hover .hover-show { opacity: 1; }
        .ab-btn {
            position: absolute; top: 18px; left: 50%; transform: translateX(-50%);
            z-index: 6;
            display: inline-flex; align-items: center; gap: 6px;
            padding: 6px 14px;
            font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
            background: oklch(0.10 0.01 265 / 0.65);
            color: oklch(0.92 0 0 / 0.85);
            border: 1px solid oklch(0.95 0 0 / 0.10);
            border-radius: var(--radius-theme-md);
            cursor: pointer;
            backdrop-filter: blur(8px);
            box-shadow: 0 2px 10px oklch(0 0 0 / 0.35);
        }
        .ab-btn.on {
            background: color-mix(in oklab, var(--color-brand) 70%, transparent);
            color: #fff;
            border-color: var(--color-brand);
        }
        .ab-label {
            position: absolute; top: 14px;
            font-family: var(--font-mono); font-size: 10px;
            padding: 2px 7px;
            background: oklch(0.10 0.01 265 / 0.7);
            color: oklch(0.95 0 0 / 0.85);
            border-radius: 3px;
            letter-spacing: 0.12em; font-weight: 600;
        }
        .ab-label.left  { left: 14px; }
        .ab-label.right { right: 14px; }
        .ab-divider {
            position: absolute; top: 0; bottom: 0;
            left: var(--ab-split, 50%); width: 2px;
            background: oklch(0.99 0 0 / 0.9);
            box-shadow: 0 0 10px oklch(0 0 0 / 0.6);
            cursor: ew-resize;
            transform: translateX(-1px);
        }
        .ab-handle {
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            width: 34px; height: 34px; border-radius: 999px;
            background: oklch(0.97 0 0 / 0.92);
            color: oklch(0.25 0.01 265);
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 14px;
            box-shadow: 0 2px 12px oklch(0 0 0 / 0.45);
            cursor: ew-resize;
        }
        .pipeline-busy {
            position: absolute; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            z-index: 7;
            font-size: 28px; font-weight: 700;
            color: oklch(0.95 0 0 / 0.9);
            background: oklch(0.10 0.01 265 / 0.55);
            padding: 6px 16px;
            border-radius: var(--radius-theme-md);
            backdrop-filter: blur(8px);
            letter-spacing: 0.2em;
            animation: pipeline-pulse 1.2s ease-in-out infinite;
        }
        @keyframes pipeline-pulse {
            0%, 100% { opacity: 0.45; }
            50%      { opacity: 1.0; }
        }
        /* Smaller-than-default chip proportions + uppercase tracking to
           read as a transient status badge rather than a content tag.
           Color/background/border come from the global \`.chip.warning\`. */
        .dirty-chip {
            padding: 2px 8px;
            font-size: 10px; font-weight: 600;
            letter-spacing: 0.04em; text-transform: uppercase;
        }
    `],
})
export class EditCanvasComponent {
    datasetName = input.required<string>();
    mediaFile = input.required<string>();
    hasOverlay = input<boolean>(false);
    metadata = input<Record<string, unknown> | null>(null);

    prev = output<void>();
    next = output<void>();

    protected compareOn = signal<boolean>(false);
    protected splitPercent = signal<number>(50);
    /** Canvas zoom factor (1 = 100%), driven by the footer's zoom controls. */
    protected zoom = signal<number>(1);
    private dragging = signal<boolean>(false);
    private dragPointerId: number | null = null;
    private stageRef = viewChild<ElementRef<HTMLElement>>('stage');
    private canvasRef = viewChild<ElementRef<HTMLCanvasElement>>('previewCanvas');

    /**
     * Hidden CORS image used ONLY to feed the canvas pixel pipeline. Kept
     * separate from the visible `<img>` (which is plain/no-cors and always
     * renders) so display never depends on CORS succeeding.
     */
    private corsLoader: HTMLImageElement | null = null;

    private overlay = inject(OverlayStore);
    private rtc = inject(RuntimeConfigService);
    private host = inject(ElementRef<HTMLElement>);
    protected state = inject(PipelineEditorState);
    protected preview = inject(PreviewPipeline);

    /** Fullscreen state for the canvas (host element = stage + footer). */
    protected isFullscreen = signal<boolean>(false);

    protected toggleFullscreen(): void {
        if (document.fullscreenElement) {
            document.exitFullscreen?.();
        } else {
            this.host.nativeElement.requestFullscreen?.().catch(() => {});
        }
    }

    @HostListener('document:fullscreenchange')
    protected onFullscreenChange(): void {
        this.isFullscreen.set(document.fullscreenElement === this.host.nativeElement);
    }

    /**
     * Set when an overlay-PNG load fails (e.g., recipe-only state where the
     * JSON exists but the rendered PNG was deleted from disk). Forces
     * `displayUrl` to use the original image — the PreviewPipeline then
     * replays the saved recipe via `state.blocks()`. Reset on identity change.
     */
    private overlayLoadFailed = signal<boolean>(false);

    /**
     * URL for the visible `<img class="layer base">`. Prefers the rendered
     * overlay PNG when one is on disk (so an unattached editor — e.g. a
     * frame where the canvas pixel pipeline hasn't drawn yet — still shows
     * the saved result). Falls back to the original if the overlay PNG
     * 404s (`overlayLoadFailed`) or no overlay exists.
     *
     * IMPORTANT: This is the BASE-LAYER URL only. The canvas pixel pipeline
     * reads `pixelSourceUrl` (the original), not this URL, so the live
     * preview is `original + recipe`, never `overlay + recipe` (which would
     * double-apply). See `buildPixelSourceUrl` for the rationale.
     */
    protected displayUrl = computed<string>(() => {
        const rev = this.state.sourceRev();
        let overlayInfo: OverlayUrlInfo | null = null;
        if (this.hasOverlay() && !this.overlayLoadFailed()) {
            const id = `${this.datasetName()}/${this.mediaFile()}`;
            const ov = (this.overlay.entities() ?? []).find((o: Overlay) => o.id === id);
            if (ov?.overlay_file) {
                overlayInfo = { dataset_name: ov.dataset_name, overlay_file: ov.overlay_file, hash: ov.hash };
            }
        }
        return buildDisplayUrl(this.rtc.mediaBaseUrl, this.datasetName(), this.mediaFile(), overlayInfo, rev);
    });

    /**
     * URL for the canvas pixel pipeline — always the original. The
     * PreviewPipeline applies the slider recipe on top of these pixels;
     * sourcing the overlay PNG (already recipe-baked) would double-apply.
     * See `buildPixelSourceUrl` for the full explanation.
     */
    protected pixelSourceUrl = computed<string>(() =>
        buildPixelSourceUrl(this.rtc.mediaBaseUrl, this.datasetName(), this.mediaFile(), this.state.sourceRev()),
    );

    constructor() {
        // Reset the overlay-fallback flag whenever the image identity
        // changes, so the next image starts by trying its overlay (if any).
        let lastKey = '';
        effect(() => {
            const key = `${this.datasetName()}/${this.mediaFile()}`;
            if (key !== lastKey) {
                lastKey = key;
                this.overlayLoadFailed.set(false);
                this.zoom.set(1);   // reset zoom when the image changes
            }
        });

        // Drive the canvas pixel pipeline off the ORIGINAL image, never the
        // overlay PNG. The PreviewPipeline applies the slider recipe on top
        // of these pixels; if we sourced the overlay PNG (already
        // recipe-baked) instead, every edit would apply the recipe twice
        // (user-reported "save → reload → values applied again"). The
        // visible <img class="layer base"> still uses `displayUrl` so an
        // unattached editor renders the saved overlay PNG plainly — the
        // canvas overlay (this load) drives the live preview on top.
        //
        // We load a SEPARATE crossorigin copy purely to feed getImageData;
        // if the CORS load fails (or taints), we degrade to the static
        // <img> — the image is never broken, worst case is "no live preview".
        effect(() => {
            const url = this.pixelSourceUrl();
            this.loadCanvasSource(url);
        });
    }

    /** Start (or restart) the crossorigin pixel load for the canvas. */
    private loadCanvasSource(url: string): void {
        this.preview.detach();
        this.clearCanvas();
        if (this.corsLoader) {
            this.corsLoader.onload = null;
            this.corsLoader.onerror = null;
        }
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.decoding = 'async';
        img.onload = () => {
            const canvas = this.canvasRef()?.nativeElement;
            if (canvas) this.preview.attach(canvas, img);
        };
        img.onerror = () => {
            // CORS or network failure for the pixel copy. The visible <img>
            // keeps showing the image; we just have no live preview overlay.
            this.preview.detach();
            this.clearCanvas();
        };
        img.src = withCorsParam(url);
        this.corsLoader = img;
    }

    /** Wipe the overlay canvas so a failed/superseded load leaves no stale pixels. */
    private clearCanvas(): void {
        const canvas = this.canvasRef()?.nativeElement;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    protected onDisplayError(): void {
        // Overlay PNG missing (recipe-only state) — flip the visible <img>
        // to the original. `displayUrl` recomputes; the canvas pixel
        // pipeline does NOT change here (it has always been sourced from
        // the original, see `pixelSourceUrl`), so the live preview keeps
        // rendering `original + recipe` regardless of whether the saved
        // PNG is reachable.
        if (this.hasOverlay() && !this.overlayLoadFailed()) {
            this.overlayLoadFailed.set(true);
        }
        // If the original itself fails to display there is nothing more we
        // can do here — the file is genuinely unreachable.
    }

    toggleCompare(): void { this.compareOn.update(v => !v); }

    onSplitPointerDown(e: PointerEvent): void {
        if (!this.compareOn()) return;
        if (this.dragging()) return;
        (e.target as Element).setPointerCapture(e.pointerId);
        this.dragPointerId = e.pointerId;
        this.dragging.set(true);
        e.preventDefault();
    }
    onSplitPointerMove(e: PointerEvent): void {
        if (!this.dragging() || e.pointerId !== this.dragPointerId) return;
        const stage = this.stageRef()?.nativeElement;
        if (!stage) return;
        const rect = stage.getBoundingClientRect();
        const pct = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
        this.splitPercent.set(pct);
    }
    onSplitPointerUp(e: PointerEvent): void {
        if (e.pointerId !== this.dragPointerId) return;
        this.endDrag(e);
    }
    onSplitPointerCancel(e: PointerEvent): void {
        if (e.pointerId !== this.dragPointerId) return;
        this.endDrag(e);
    }
    private endDrag(e: PointerEvent): void {
        this.dragging.set(false);
        this.dragPointerId = null;
        (e.target as Element).releasePointerCapture?.(e.pointerId);
    }

    /**
     * Footer metadata strip. Same shape Details mode shows — resolution,
     * AR, orientation, file size — so the two surfaces stay in sync via
     * the shared `buildCanvasMeta` helper. `hasOverlay` comes from the
     * input (already-resolved by the parent mode), not the metadata blob.
     */
    protected meta = computed<CanvasMeta>(() => ({
        ...buildCanvasMeta(this.metadata()),
        file: this.mediaFile(),
        hasOverlay: this.hasOverlay(),
    }));
}
