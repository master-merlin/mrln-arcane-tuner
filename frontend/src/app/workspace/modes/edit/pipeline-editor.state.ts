import { Injectable, computed, inject, signal } from '@angular/core';
import { OverlayStore } from '../../../state/overlay.store';
import { DatasetService, PipelineBlock } from '../../../services/dataset';
import {
    OperationKind, PIPELINE_ORDER, BACKEND_TYPE_FOR, DEFAULT_PARAMS,
    WBParams, CurvesParams, LutParams, ColorMatchParams, HslParams,
    ColorToneParams, VignetteParams, LensParams, SharpenParams,
    RestoreParams, UpscaleParams,
} from './operation-defs';

export interface OpEntry<P> {
    kind: OperationKind;
    enabled: boolean;
    params: P;
}

type AnyOp =
    | OpEntry<WBParams> | OpEntry<CurvesParams> | OpEntry<LutParams>
    | OpEntry<ColorMatchParams> | OpEntry<HslParams> | OpEntry<ColorToneParams>
    | OpEntry<VignetteParams> | OpEntry<LensParams> | OpEntry<SharpenParams>
    | OpEntry<RestoreParams> | OpEntry<UpscaleParams>;

function mkOp<K extends keyof typeof DEFAULT_PARAMS>(
    kind: K, enabled = false,
): OpEntry<typeof DEFAULT_PARAMS[K]> {
    return { kind: kind as OperationKind, enabled, params: structuredClone(DEFAULT_PARAMS[kind]) };
}

/**
 * Component-scoped (NOT root) edit-mode state. Provided in EditMode's
 * `providers: [PipelineEditorState]` so it dies with the mode. Holds
 * working state for 12 pipeline operations + user-reorderable order.
 *
 * Saved overlays live in OverlayStore (the source of truth for what's
 * on disk). This service is the live in-flight working set, synced via
 * hydrate() on mount and applyAndSave() on commit.
 */
@Injectable()
export class PipelineEditorState {
    private overlay = inject(OverlayStore);
    private datasets = inject(DatasetService);

    readonly datasetName = signal<string>('');
    readonly mediaFile = signal<string>('');

    readonly whiteBalance = signal(mkOp('white_balance'));
    readonly curves       = signal(mkOp('curves'));
    readonly lut          = signal(mkOp('lut'));
    readonly colorMatch   = signal(mkOp('color_match'));
    readonly hslSelective = signal(mkOp('hsl_selective'));
    readonly colorTone    = signal(mkOp('color_tone'));
    readonly vignette     = signal(mkOp('vignette'));
    readonly lens         = signal(mkOp('lens'));
    readonly sharpen      = signal(mkOp('sharpen'));
    readonly denoise      = signal(mkOp('denoise'));
    readonly faceRestore  = signal(mkOp('face_restore'));
    readonly upscale      = signal(mkOp('upscale'));

    /** User-mutable pipeline order. Mirrors PIPELINE_ORDER initially. */
    readonly operationOrder = signal<OperationKind[]>([...PIPELINE_ORDER]);

    /** Snapshot of last-saved state — used to compute `dirty`. */
    private savedSnapshot = signal<string>('');

    /** Live JSON of all ops + order — cheap to deep-compare via stringify. */
    private liveJson = computed(() => JSON.stringify({
        order: this.operationOrder(),
        ops: this.allOps().map(o => ({ kind: o.kind, enabled: o.enabled, params: o.params })),
    }));

    readonly dirty = computed<boolean>(() => this.liveJson() !== this.savedSnapshot());

    /** Every op as a flat array. Color Match always included (separate flow). */
    private allOps(): AnyOp[] {
        return [
            this.whiteBalance(), this.curves(), this.lut(), this.colorMatch(),
            this.hslSelective(), this.colorTone(), this.vignette(), this.lens(),
            this.sharpen(), this.denoise(), this.faceRestore(), this.upscale(),
        ];
    }

    private signalFor(kind: OperationKind) {
        switch (kind) {
            case 'white_balance':  return this.whiteBalance;
            case 'curves':         return this.curves;
            case 'lut':            return this.lut;
            case 'color_match':    return this.colorMatch;
            case 'hsl_selective':  return this.hslSelective;
            case 'color_tone':     return this.colorTone;
            case 'vignette':       return this.vignette;
            case 'lens':           return this.lens;
            case 'sharpen':        return this.sharpen;
            case 'denoise':        return this.denoise;
            case 'face_restore':   return this.faceRestore;
            case 'upscale':        return this.upscale;
        }
    }

    /**
     * Ordered, enabled-only blocks ready for renderPipeline().
     * - Color Match is emitted FIRST (backend convention).
     * - color_tone expands to hue_saturation + contrast blocks.
     * - Order is by operationOrder() for everything else.
     */
    readonly blocks = computed<PipelineBlock[]>(() => {
        const out: PipelineBlock[] = [];

        // Color Match always first (backend applies it first regardless).
        const cm = this.colorMatch();
        if (cm.enabled && cm.params.reference_path) {
            out.push({ type: 'color_match', enabled: true, params: { ...cm.params } });
        }

        for (const kind of this.operationOrder()) {
            const op = this.signalFor(kind)();
            if (!op.enabled) continue;
            const t = BACKEND_TYPE_FOR[kind];
            if (Array.isArray(t)) {
                // color_tone → hue_saturation + contrast
                const p = op.params as ColorToneParams;
                out.push({ type: 'hue_saturation', enabled: true, params: { hue_shift: p.hue_shift, saturation: p.saturation } });
                out.push({ type: 'contrast', enabled: true, params: { contrast: p.contrast } });
            } else {
                out.push({ type: t, enabled: true, params: { ...op.params } });
            }
        }
        return out;
    });

    /** Toggle the enabled flag for a single op. */
    setEnabled(kind: OperationKind, enabled: boolean): void {
        const s = this.signalFor(kind);
        s.update(o => ({ ...o, enabled }));
    }

    /** Reorder operationOrder by moving item at `from` to position `to`. */
    moveOperation(from: number, to: number): void {
        this.operationOrder.update(arr => {
            if (from < 0 || from >= arr.length) return arr;
            const next = arr.slice();
            const [item] = next.splice(from, 1);
            const dest = Math.max(0, Math.min(next.length, to));
            next.splice(dest, 0, item);
            return next;
        });
    }

    /** Reset just one panel to defaults. */
    resetPanel(kind: OperationKind): void {
        this.signalFor(kind).set(mkOp(kind as keyof typeof DEFAULT_PARAMS) as any);
    }

    /** Stamp the current state as the saved snapshot. */
    markClean(): void {
        this.savedSnapshot.set(this.liveJson());
    }

    /** Reset everything to defaults + mark clean. Called by hydrate (no overlay). */
    resetAll(): void {
        this.whiteBalance.set(mkOp('white_balance'));
        this.curves.set(mkOp('curves'));
        this.lut.set(mkOp('lut'));
        this.colorMatch.set(mkOp('color_match'));
        this.hslSelective.set(mkOp('hsl_selective'));
        this.colorTone.set(mkOp('color_tone'));
        this.vignette.set(mkOp('vignette'));
        this.lens.set(mkOp('lens'));
        this.sharpen.set(mkOp('sharpen'));
        this.denoise.set(mkOp('denoise'));
        this.faceRestore.set(mkOp('face_restore'));
        this.upscale.set(mkOp('upscale'));
        this.operationOrder.set([...PIPELINE_ORDER]);
        this.markClean();
    }

    // hydrate() / applyAndSave() / revert() are wired in Phase 3 (Task 9).
    async hydrate(_datasetName: string, _mediaFile: string): Promise<void> { this.resetAll(); }
    async applyAndSave(): Promise<void> { this.markClean(); }
    async revert(): Promise<void> { this.resetAll(); }
}
