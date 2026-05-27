import { Injectable, computed, inject, signal } from '@angular/core';
import { Overlay, OverlayStore } from '../../../state/overlay.store';
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
    readonly lut          = signal<OpEntry<LutParams>>(mkOp('lut') as OpEntry<LutParams>);
    readonly colorMatch   = signal(mkOp('color_match'));
    readonly hslSelective = signal(mkOp('hsl_selective'));
    readonly colorTone    = signal(mkOp('color_tone'));
    readonly vignette     = signal<OpEntry<VignetteParams>>(mkOp('vignette') as OpEntry<VignetteParams>);
    readonly lens         = signal<OpEntry<LensParams>>(mkOp('lens') as OpEntry<LensParams>);
    readonly sharpen      = signal<OpEntry<SharpenParams>>(mkOp('sharpen') as OpEntry<SharpenParams>);
    readonly denoise      = signal(mkOp('denoise'));
    readonly faceRestore  = signal(mkOp('face_restore'));
    readonly upscale      = signal(mkOp('upscale'));

    /** User-mutable pipeline order. Mirrors PIPELINE_ORDER initially. */
    readonly operationOrder = signal<OperationKind[]>([...PIPELINE_ORDER]);

    /** URL of the most-recent live-preview overlay PNG (from renderPipeline
     *  with replaceRecipe=false). Null until first render completes. */
    readonly previewOverlay = signal<{ url: string; hash: string } | null>(null);

    /** True while a render request is in flight (drives a small spinner). */
    readonly rendering = signal<boolean>(false);

    private pendingRender = false;

    /**
     * Trigger a render based on current blocks(). Called by the debounced
     * effect in EditMode whenever blocks() changes with stable image identity.
     *
     * `replaceRecipe` defaults false (preview); applyAndSave passes true.
     * In-flight handling: one slot queue; if a new call arrives mid-flight,
     * one more render fires after the current finishes (latest wins).
     */
    async renderNow(replaceRecipe = false): Promise<void> {
        const name = this.datasetName();
        const file = this.mediaFile();
        if (!name || !file) return;

        if (this.rendering()) { this.pendingRender = true; return; }
        this.rendering.set(true);
        try {
            const result = await this.overlay.renderPipeline(
                name, file, this.blocks(), 512, 32, replaceRecipe,
            );
            if (result.ok) {
                const r = result.value;
                this.previewOverlay.set({ url: r.overlay, hash: r.hash });
            }
        } finally {
            this.rendering.set(false);
            if (this.pendingRender) {
                this.pendingRender = false;
                void this.renderNow(false);
            }
        }
    }

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

    /**
     * Hydrate working state from the saved overlay recipe. If no overlay
     * exists, reset to defaults. Always marks clean (dirty=false) afterwards.
     */
    async hydrate(datasetName: string, mediaFile: string): Promise<void> {
        this.datasetName.set(datasetName);
        this.mediaFile.set(mediaFile);

        // Load (or refresh) the overlay row in the store.
        await this.overlay.loadFor(datasetName, mediaFile);
        const id = `${datasetName}/${mediaFile}`;
        const row = (this.overlay.entities() ?? []).find((o: Overlay) => o.id === id);

        // No overlay → defaults.
        if (!row?.operations || row.operations.length === 0) {
            this.resetAll();
            this.markClean();
            return;
        }

        // Reset first, then apply each recipe op into the corresponding signal.
        this.resetAll();
        for (const op of row.operations) {
            this.applyRecipeOp(op.type, op.params ?? {}, op.enabled !== false);
        }
        this.markClean();
    }

    /**
     * Project a backend-type op (e.g. `hue_saturation`) into the matching
     * frontend signal. `hue_saturation` + `contrast` both feed `color_tone`.
     * Unknown types are ignored (forward-compat).
     */
    private applyRecipeOp(type: string, params: any, enabled: boolean): void {
        switch (type) {
            case 'denoise':
                this.denoise.update(o => ({ ...o, enabled, params: { ...o.params, ...params } })); return;
            case 'face_restore':
                this.faceRestore.update(o => ({ ...o, enabled, params: { ...o.params, ...params } })); return;
            case 'white_balance':
                this.whiteBalance.update(o => ({ ...o, enabled, params: { ...o.params, ...params } })); return;
            case 'curves':
                this.curves.update(o => ({ ...o, enabled, params: { ...o.params, ...params } })); return;
            case 'cube_lut':
                this.lut.update(o => ({ ...o, enabled, params: { ...o.params, ...params } })); return;
            case 'color_match':
                this.colorMatch.update(o => ({ ...o, enabled, params: { ...o.params, ...params } })); return;
            case 'hsl_selective':
                // HslParams is a per-band dict — replace wholesale (not merge), so removing a band on the server actually removes it here.
                this.hslSelective.update(o => ({ ...o, enabled, params: { ...params } })); return;
            // `hue_saturation` and `contrast` are two backend ops that collapse into the single frontend `color_tone` panel.
            // Use `enabled || o.enabled` so the second op doesn't clobber the first's enable flag.
            case 'hue_saturation':
                this.colorTone.update(o => ({
                    ...o, enabled: enabled || o.enabled,
                    params: { ...o.params, hue_shift: params.hue_shift ?? 0, saturation: params.saturation ?? 1 },
                })); return;
            case 'contrast':
                this.colorTone.update(o => ({
                    ...o, enabled: enabled || o.enabled,
                    params: { ...o.params, contrast: params.contrast ?? 1 },
                })); return;
            case 'vignette':
                this.vignette.update(o => ({ ...o, enabled, params: { ...o.params, ...params } })); return;
            case 'lens_correction':
                this.lens.update(o => ({ ...o, enabled, params: { ...o.params, ...params } })); return;
            case 'sharpening':
                this.sharpen.update(o => ({ ...o, enabled, params: { ...o.params, ...params } })); return;
            case 'upscale':
                this.upscale.update(o => ({ ...o, enabled, params: { ...o.params, ...params } })); return;
        }
    }

    /**
     * Promote the current preview to the saved recipe (replaceRecipe=true).
     * Snapshots state so `dirty` clears.
     */
    async applyAndSave(): Promise<void> {
        await this.renderNow(true);
        this.markClean();
    }

    /**
     * Delete the saved overlay and reset working state to defaults.
     */
    async revert(): Promise<void> {
        const name = this.datasetName();
        const file = this.mediaFile();
        if (name && file) {
            await this.overlay.deleteOverlay(name, file);
        }
        this.resetAll();
        this.previewOverlay.set(null);
        this.markClean();
    }
}
