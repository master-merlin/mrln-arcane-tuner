// frontend/src/app/workspace/modes/edit/preview/preview-pipeline.ts
import { DestroyRef, Injectable, computed, effect, inject, signal } from '@angular/core';
import { PipelineEditorState } from '../pipeline-editor.state';
import type { ApplyFn, HistogramData } from './preview-types';
import type {
    OperationKind, ColorToneParams, WBParams, CurvesParams,
    HslParams, VignetteParams, LensParams, SharpenParams, LutParams,
} from '../operation-defs';
import { applyWhiteBalance } from '../panels/white-balance.math';
import { applyCurves } from '../panels/curves.math';
import { applyColorTone } from '../panels/color-tone.math';
import { applyHslSelective } from '../panels/hsl.math';
import { applyVignette } from '../panels/vignette.math';
import { applyLensCorrection } from '../panels/lens.math';
import { applySharpen } from '../panels/sharpen.math';
import { applyLutStack, type LutStackPreviewParams } from '../panels/lut.math';

const MAX_PREVIEW_SIZE = 2048;
const SPINNER_THRESHOLD_MS = 80;

/**
 * Component-scoped (NOT root). Provided in EditMode's
 * `providers: [PipelineEditorState, PreviewPipeline]` so it dies
 * with the mode.
 *
 * Owns the canvas and the original ImageData. Subscribes to
 * `state.blocks()` + `state.parsedCubes()` via an effect; coalesces
 * updates onto requestAnimationFrame; walks `state.operationOrder()`
 * and dispatches to per-op math modules. AI ops and color_match
 * have no preview implementation and are silently skipped — backend
 * Save handles them authoritatively.
 */
@Injectable()
export class PreviewPipeline {
    private state = inject(PipelineEditorState);
    private destroyRef = inject(DestroyRef);

    private canvas: HTMLCanvasElement | null = null;
    private ctx: CanvasRenderingContext2D | null = null;
    private original: ImageData | null = null;
    private working: Uint8ClampedArray | null = null;
    private rafHandle: number | null = null;
    private lastRenderMs = 0;

    readonly rendering = signal<boolean>(false);
    readonly histogram = signal<HistogramData | null>(null);

    /** Convenience for templates: spinner only after the slow-pass threshold. */
    readonly showSpinner = computed(() =>
        this.rendering() && this.lastRenderMs > SPINNER_THRESHOLD_MS,
    );

    constructor() {
        effect(() => {
            // Subscribe to all the inputs the render reads.
            this.state.blocks();
            this.state.parsedCubes();
            this.scheduleRender();
        });
        this.destroyRef.onDestroy(() => this.detach());
    }

    /**
     * Called by EditCanvas once the source <img> has loaded. Sizes the
     * canvas (capped at MAX_PREVIEW_SIZE), draws the source, captures
     * the immutable original ImageData, and schedules the first render.
     */
    attach(canvas: HTMLCanvasElement, source: HTMLImageElement): void {
        this.detach();
        let pw = source.naturalWidth, ph = source.naturalHeight;
        if (pw === 0 || ph === 0) return;
        if (pw > MAX_PREVIEW_SIZE || ph > MAX_PREVIEW_SIZE) {
            const scale = MAX_PREVIEW_SIZE / Math.max(pw, ph);
            pw = Math.round(pw * scale);
            ph = Math.round(ph * scale);
        }
        canvas.width = pw;
        canvas.height = ph;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.drawImage(source, 0, 0, pw, ph);
        this.canvas = canvas;
        this.ctx = ctx;
        this.original = ctx.getImageData(0, 0, pw, ph);
        this.working = new Uint8ClampedArray(this.original.data.length);
        this.scheduleRender();
    }

    detach(): void {
        if (this.rafHandle != null) {
            cancelAnimationFrame(this.rafHandle);
            this.rafHandle = null;
        }
        this.canvas = null;
        this.ctx = null;
        this.original = null;
        this.working = null;
        this.rendering.set(false);
    }

    private scheduleRender(): void {
        if (this.rafHandle != null || !this.original) return;
        this.rafHandle = requestAnimationFrame(() => {
            this.rafHandle = null;
            this.render();
        });
    }

    private readonly applyByKind: ReadonlyMap<OperationKind, ApplyFn<any>> = new Map<OperationKind, ApplyFn<any>>([
        ['white_balance', applyWhiteBalance],
        ['curves',        applyCurves],
        ['color_tone',    applyColorTone],
        ['hsl_selective', applyHslSelective],
        ['vignette',      applyVignette],
        ['lens',          applyLensCorrection],
        ['sharpen',       applySharpen],
        ['lut',           applyLutStack as ApplyFn<any>],
        // denoise, face_restore, upscale, color_match: intentionally absent.
    ]);

    private render(): void {
        if (!this.ctx || !this.original || !this.working) return;
        const t0 = performance.now();
        this.rendering.set(true);

        // Copy original → working.
        this.working.set(this.original.data);
        const w = this.original.width, h = this.original.height;

        const order = this.state.operationOrder();
        const cubes = this.state.parsedCubes();
        for (const kind of order) {
            const op = this.opByKind(kind);
            if (!op || !op.enabled) continue;
            const apply = this.applyByKind.get(kind);
            if (!apply) continue;
            if (kind === 'lut') {
                const params: LutStackPreviewParams = {
                    ...(op.params as LutParams),
                    parsedCubes: cubes,
                };
                apply(this.working, w, h, params);
            } else {
                apply(this.working, w, h, op.params);
            }
        }

        // Paint result + compute histogram.
        const out = new ImageData(w, h);
        out.data.set(this.working);
        this.ctx.putImageData(out, 0, 0);
        this.histogram.set(this.computeHistogram(this.working));

        this.lastRenderMs = performance.now() - t0;
        this.rendering.set(false);
    }

    private opByKind(kind: OperationKind): { enabled: boolean; params: unknown } | null {
        switch (kind) {
            case 'white_balance':  return this.state.whiteBalance();
            case 'curves':         return this.state.curves();
            case 'color_tone':     return this.state.colorTone();
            case 'hsl_selective':  return this.state.hslSelective();
            case 'vignette':       return this.state.vignette();
            case 'lens':           return this.state.lens();
            case 'sharpen':        return this.state.sharpen();
            case 'lut':            return this.state.lut();
            // Skipped in preview:
            case 'denoise':        return null;
            case 'face_restore':   return null;
            case 'upscale':        return null;
            case 'color_match':    return null;
        }
    }

    private computeHistogram(data: Uint8ClampedArray): HistogramData {
        const r = new Array(256).fill(0);
        const g = new Array(256).fill(0);
        const b = new Array(256).fill(0);
        const luminance = new Array(256).fill(0);
        for (let i = 0; i < data.length; i += 4) {
            r[data[i]]++;
            g[data[i + 1]]++;
            b[data[i + 2]]++;
            const lum = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
            luminance[Math.min(255, lum)]++;
        }
        return { r, g, b, luminance };
    }
}
