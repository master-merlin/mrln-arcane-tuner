import { Injectable, computed, inject, signal } from '@angular/core';
import { Overlay, OverlayStore } from '../../../state/overlay.store';
import { PipelineBlock } from '../../../services/dataset';
import {
    OperationKind, PIPELINE_ORDER, BACKEND_TYPE_FOR, DEFAULT_PARAMS,
    WBParams, CurvesParams, LutParams, ColorMatchParams, HslParams,
    ColorToneParams, VignetteParams, LensParams, SharpenParams,
    RestoreParams, UpscaleParams,
} from './operation-defs';
import type { CubeLut } from './preview/preview-types';

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
 * Live preview rendering is owned by `PreviewPipeline` (sibling
 * service); this store is the source of truth for op state. Backend
 * round-trips are owned by `applyAndSave()` (Save) and `bake()`.
 */
@Injectable()
export class PipelineEditorState {
    private overlay = inject(OverlayStore);

    readonly datasetName = signal<string>('');
    readonly mediaFile = signal<string>('');

    readonly whiteBalance = signal(mkOp('white_balance'));
    readonly curves       = signal(mkOp('curves'));
    readonly lut          = signal<OpEntry<LutParams>>(mkOp('lut') as OpEntry<LutParams>);
    readonly colorMatch   = signal<OpEntry<ColorMatchParams>>(mkOp('color_match') as OpEntry<ColorMatchParams>);
    readonly hslSelective = signal(mkOp('hsl_selective'));
    readonly colorTone    = signal(mkOp('color_tone'));
    readonly vignette     = signal<OpEntry<VignetteParams>>(mkOp('vignette') as OpEntry<VignetteParams>);
    readonly lens         = signal<OpEntry<LensParams>>(mkOp('lens') as OpEntry<LensParams>);
    readonly sharpen      = signal<OpEntry<SharpenParams>>(mkOp('sharpen') as OpEntry<SharpenParams>);
    readonly denoise      = signal<OpEntry<RestoreParams>>(mkOp('denoise') as OpEntry<RestoreParams>);
    readonly faceRestore  = signal<OpEntry<RestoreParams>>(mkOp('face_restore') as OpEntry<RestoreParams>);
    readonly upscale      = signal<OpEntry<UpscaleParams>>(mkOp('upscale') as OpEntry<UpscaleParams>);

    /** User-mutable pipeline order. Mirrors PIPELINE_ORDER initially. */
    readonly operationOrder = signal<OperationKind[]>([...PIPELINE_ORDER]);

    /**
     * Transient (per-session) cache of parsed .cube files keyed by
     * filename. Populated by LutPanel on import. Not persisted — on
     * reload the user must re-import .cube files for preview, while
     * backend Save still applies them authoritatively.
     */
    readonly parsedCubes = signal<ReadonlyMap<string, CubeLut>>(new Map());

    /**
     * Bumped on Bake-in completion. EditCanvas appends `?r=<rev>` to
     * sourceUrl so the browser re-fetches the (newly baked) original.
     */
    readonly sourceRev = signal<number>(0);

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

        const cm = this.colorMatch();
        if (cm.enabled && cm.params.reference_path) {
            out.push({ type: 'color_match', enabled: true, params: { ...cm.params } });
        }

        for (const kind of this.operationOrder()) {
            const op = this.signalFor(kind)();
            if (!op.enabled) continue;
            const t = BACKEND_TYPE_FOR[kind];
            if (Array.isArray(t)) {
                const p = op.params as ColorToneParams;
                out.push({ type: 'hue_saturation', enabled: true, params: { hue_shift: p.hue_shift, saturation: p.saturation } });
                out.push({ type: 'contrast', enabled: true, params: { contrast: p.contrast } });
            } else {
                out.push({ type: t, enabled: true, params: { ...op.params } });
            }
        }
        return out;
    });

    /** Add or replace a parsed .cube file in the transient cache. */
    ingestCube(filename: string, cube: CubeLut): void {
        this.parsedCubes.update(m => {
            const next = new Map(m);
            next.set(filename, cube);
            return next;
        });
    }

    setEnabled(kind: OperationKind, enabled: boolean): void {
        const s = this.signalFor(kind);
        s.update(o => ({ ...o, enabled }));
    }

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

    resetPanel(kind: OperationKind): void {
        this.signalFor(kind).set(mkOp(kind as keyof typeof DEFAULT_PARAMS) as any);
    }

    markClean(): void {
        this.savedSnapshot.set(this.liveJson());
    }

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
     * Load editor state for an image. When `hasOverlay` is false we skip the
     * overlay-recipe fetch entirely — the recipe endpoint 404s for every
     * un-edited image, so fetching unconditionally produced a guaranteed 404
     * (wasted request + confusing error noise) on every image without an
     * overlay. The `has_overlay` flag is kept in sync by the WS bridge, so it
     * is the authoritative gate.
     */
    async hydrate(datasetName: string, mediaFile: string, hasOverlay: boolean): Promise<void> {
        this.datasetName.set(datasetName);
        this.mediaFile.set(mediaFile);

        if (hasOverlay) {
            await this.overlay.loadFor(datasetName, mediaFile);
            const id = `${datasetName}/${mediaFile}`;
            const row = (this.overlay.entities() ?? []).find((o: Overlay) => o.id === id);
            if (row?.operations && row.operations.length > 0) {
                this.resetAll();
                for (const op of row.operations) {
                    this.applyRecipeOp(op.type, op.params ?? {}, op.enabled !== false);
                }
                this.markClean();
                return;
            }
        }

        // No overlay (flag false, or flagged but recipe was empty/missing).
        this.resetAll();
        this.markClean();
    }

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
                this.hslSelective.update(o => ({ ...o, enabled, params: { ...params } })); return;
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
     * Save: render the authoritative overlay PNG + persist the recipe.
     * Backend uses replace_recipe=true so it sources from the original
     * (not from any existing overlay PNG).
     */
    async applyAndSave(): Promise<void> {
        const name = this.datasetName();
        const file = this.mediaFile();
        if (!name || !file) return;
        await this.overlay.renderPipeline(name, file, this.blocks(), 512, 32, true);
        this.markClean();
    }

    /**
     * Bake: flatten the saved overlay into the original file. Backend
     * deletes the recipe + PNG and replaces the original on disk.
     * Requires a saved overlay AND clean state (no in-flight edits).
     */
    async bake(): Promise<void> {
        const name = this.datasetName();
        const file = this.mediaFile();
        if (!name || !file) return;
        await this.overlay.commitOverlay(name, file);
        this.resetAll();
        this.sourceRev.update(r => r + 1);
    }

    /** Delete the saved overlay and reset working state to defaults. */
    async revert(): Promise<void> {
        const name = this.datasetName();
        const file = this.mediaFile();
        if (name && file) {
            await this.overlay.deleteOverlay(name, file);
        }
        this.resetAll();
        // The overlay URL the canvas was displaying just became invalid on
        // disk. Bump sourceRev so the <img> re-fetches even if the new URL
        // (now the original) happens to match what the browser has cached.
        this.sourceRev.update(r => r + 1);
    }
}
