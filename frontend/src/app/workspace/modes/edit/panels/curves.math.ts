// frontend/src/app/workspace/modes/edit/panels/curves.math.ts
import type { ApplyFn } from '../preview/preview-types';
import type { CurvesParams } from '../operation-defs';
import { buildChannelLUT } from '../preview/utils/lut';

/**
 * Apply tone curves: Master first (luminance-ish — actually per-channel
 * since legacy applied master before per-channel), then per-channel R/G/B.
 * Each curve is mapped through a 256-entry LUT built via Catmull-Rom.
 * Identity curves (passthrough) are still cheap (LUT[i] === i).
 */
export const applyCurves: ApplyFn<CurvesParams> = (pixels, _w, _h, params) => {
    const mLut = buildChannelLUT(params.master);
    const rLut = buildChannelLUT(params.r);
    const gLut = buildChannelLUT(params.g);
    const bLut = buildChannelLUT(params.b);
    for (let i = 0; i < pixels.length; i += 4) {
        pixels[i]     = rLut[mLut[pixels[i]]];
        pixels[i + 1] = gLut[mLut[pixels[i + 1]]];
        pixels[i + 2] = bLut[mLut[pixels[i + 2]]];
    }
};
