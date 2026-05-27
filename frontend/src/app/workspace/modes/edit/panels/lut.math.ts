// frontend/src/app/workspace/modes/edit/panels/lut.math.ts
import type { ApplyFn } from '../preview/preview-types';
import type { CubeLut } from '../preview/preview-types';
import type { LutParams } from '../operation-defs';
import { trilinearInterp } from '../preview/utils/lut';

/**
 * LUT stack params for the preview path — enriches LutParams with a
 * parser cache. The coordinator (PreviewPipeline) combines the panel
 * state (filenames + strength + enabled) with PipelineEditorState's
 * `parsedCubes` map and passes the merged object here.
 *
 * Skips entries whose file isn't in `parsedCubes` (user hasn't
 * re-imported the .cube file in this session — render without that
 * LUT and let Save handle the authoritative apply server-side).
 */
export interface LutStackPreviewParams extends LutParams {
    parsedCubes: ReadonlyMap<string, CubeLut>;
}

export const applyLutStack: ApplyFn<LutStackPreviewParams> = (pixels, _w, _h, params) => {
    const active = params.luts.filter(l => l.enabled && l.strength > 0 && params.parsedCubes.has(l.file));
    if (active.length === 0) return;

    for (let i = 0; i < pixels.length; i += 4) {
        let r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
        for (const lut of active) {
            const parsed = params.parsedCubes.get(lut.file)!;
            const [lr, lg, lb] = trilinearInterp(parsed, r, g, b);
            r = Math.round(r + (lr - r) * lut.strength);
            g = Math.round(g + (lg - g) * lut.strength);
            b = Math.round(b + (lb - b) * lut.strength);
        }
        pixels[i]     = r;
        pixels[i + 1] = g;
        pixels[i + 2] = b;
    }
};
