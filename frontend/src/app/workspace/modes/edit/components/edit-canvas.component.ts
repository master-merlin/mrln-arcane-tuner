import { ChangeDetectionStrategy, Component, ElementRef, computed, effect, inject, input, output, signal, viewChild } from '@angular/core';
import { IcoComponent } from '../../../../icons/ico.component';
import { RuntimeConfigService } from '../../../../services/runtime-config.service';
import { CanvasFooterComponent, CanvasMeta } from '../../../shared/canvas-footer.component';
import { OverlayStore, Overlay } from '../../../../state/overlay.store';
import { PipelineEditorState } from '../pipeline-editor.state';
import { PreviewPipeline } from '../preview/preview-pipeline';

@Component({
    selector: 'app-edit-canvas',
    standalone: true,
    imports: [IcoComponent, CanvasFooterComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="stage hover-host" #stage>
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

            <div class="image-stage"
                 [class.compare-on]="compareOn()"
                 [style.--ab-split.%]="splitPercent()">
                <img #sourceImg
                     class="layer base"
                     [src]="sourceUrl()"
                     [alt]="mediaFile()"
                     crossorigin="anonymous"
                     loading="eager"
                     decoding="sync"
                     (load)="onSourceLoaded()"
                     (error)="onSourceError()"/>
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
                <div class="file-label">
                    <app-ico name="Image" [size]="11"/>
                    <span class="mono">{{ mediaFile() }}</span>
                </div>
            </div>
        </div>

        <app-canvas-footer [meta]="meta()"/>
    `,
    styles: [`
        :host { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
        .stage {
            flex: 1; position: relative;
            display: flex; align-items: center; justify-content: center;
            padding: 14px 16px; min-height: 0;
            background: var(--color-base);
        }
        .image-stage {
            position: relative;
            max-width: 100%; max-height: 100%;
            display: inline-flex;
            border-radius: var(--radius-theme-lg);
            box-shadow: var(--shadow-lg, 0 8px 24px rgba(0,0,0,0.25));
            overflow: hidden;
        }
        .layer { display: block; max-width: 100%; max-height: 100%; object-fit: contain; }
        .layer.overlay { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
        .image-stage.compare-on .layer.overlay {
            clip-path: inset(0 0 0 var(--ab-split, 50%));
        }
        .file-label {
            position: absolute; top: 14px; left: 14px;
            display: inline-flex; align-items: center; gap: 6px;
            padding: 3px 8px;
            font-family: var(--font-mono); font-size: 11px;
            background: oklch(0.10 0.01 265 / 0.7);
            color: var(--color-text-secondary);
            border-radius: 4px;
            backdrop-filter: blur(6px);
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
    `],
})
export class EditCanvasComponent {
    datasetName = input.required<string>();
    mediaFile = input.required<string>();
    hasOverlay = input<boolean>(false);

    prev = output<void>();
    next = output<void>();

    protected compareOn = signal<boolean>(false);
    protected splitPercent = signal<number>(50);
    private dragging = signal<boolean>(false);
    private dragPointerId: number | null = null;
    private stageRef = viewChild<ElementRef<HTMLElement>>('stage');
    private canvasRef = viewChild<ElementRef<HTMLCanvasElement>>('previewCanvas');
    private sourceImgRef = viewChild<ElementRef<HTMLImageElement>>('sourceImg');

    private overlay = inject(OverlayStore);
    private rtc = inject(RuntimeConfigService);
    private state = inject(PipelineEditorState);
    protected preview = inject(PreviewPipeline);

    /**
     * Set when an overlay-PNG load fails (e.g., recipe-only state where the
     * JSON exists but the rendered PNG was deleted from disk). Forces
     * `sourceUrl` to use the original image — the PreviewPipeline then
     * replays the saved recipe via `state.blocks()`. Reset on identity change.
     */
    private overlayLoadFailed = signal<boolean>(false);

    /**
     * Source URL:
     *  - if a rendered overlay PNG exists AND the previous attempt didn't 404,
     *    use it (state 2 in spec — "baked Overlay file")
     *  - else use the original (state 1, 3, or 4 — recipe-only replay falls here)
     * `sourceRev` is appended on Bake/Revert to bust the browser cache after
     * the original is replaced or the overlay is deleted.
     */
    protected sourceUrl = computed<string>(() => {
        const rev = this.state.sourceRev();
        if (this.hasOverlay() && !this.overlayLoadFailed()) {
            const id = `${this.datasetName()}/${this.mediaFile()}`;
            const ov = (this.overlay.entities() ?? []).find((o: Overlay) => o.id === id);
            if (ov?.overlay_file) {
                const hash = ov.hash ? `?h=${ov.hash}` : `?h=ov`;
                const revQ = rev > 0 ? `&r=${rev}` : '';
                return `${this.rtc.mediaBaseUrl}/${encodeURIComponent(ov.dataset_name)}/${ov.overlay_file}${hash}${revQ}`;
            }
        }
        const revQ = rev > 0 ? `?r=${rev}` : '';
        return `${this.rtc.mediaBaseUrl}/${encodeURIComponent(this.datasetName())}/${encodeURIComponent(this.mediaFile())}${revQ}`;
    });

    constructor() {
        // Identity change → detach + reset overlay-fail flag so the next
        // image gets a clean slate.
        let lastKey = '';
        effect(() => {
            const key = `${this.datasetName()}/${this.mediaFile()}`;
            if (key === lastKey) return;
            lastKey = key;
            this.overlayLoadFailed.set(false);
            this.preview.detach();
        });
    }

    protected onSourceLoaded(): void {
        const canvas = this.canvasRef()?.nativeElement;
        const img = this.sourceImgRef()?.nativeElement;
        if (!canvas || !img) return;
        this.preview.attach(canvas, img);
    }

    protected onSourceError(): void {
        // If the overlay PNG failed (recipe-only state with missing file),
        // flip to the original URL and let the canvas pipeline replay the
        // recipe. Don't detach yet — the retry will re-attach on success.
        if (this.hasOverlay() && !this.overlayLoadFailed()) {
            this.overlayLoadFailed.set(true);
            return;
        }
        this.preview.detach();
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

    protected meta = computed<CanvasMeta>(() => ({
        res: null, ar: null, orientation: null, size: null,
        hpsLabel: null, hpsTone: null, hasOverlay: this.hasOverlay(),
    }));
}
