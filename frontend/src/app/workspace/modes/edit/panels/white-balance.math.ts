// frontend/src/app/workspace/modes/edit/panels/white-balance.math.ts
import type { ApplyFn } from '../preview/preview-types';
import type { WBParams } from '../operation-defs';
import { clamp8, computeWBFactors } from '../preview/utils/color';

/**
 * Multiply RGB channels by WB factors derived from temperature (K) + tint.
 * Identity at temp=6500, tint=0 — caller may skip the call entirely in
 * that case, but the function is also safe to invoke (it's just slower).
 */
export const applyWhiteBalance: ApplyFn<WBParams> = (pixels, _w, _h, params) => {
    if (params.temperature === 6500 && params.tint === 0) return;
    const { wbR, wbG, wbB } = computeWBFactors(params.temperature, params.tint);
    for (let i = 0; i < pixels.length; i += 4) {
        pixels[i]     = clamp8(pixels[i]     * wbR);
        pixels[i + 1] = clamp8(pixels[i + 1] * wbG);
        pixels[i + 2] = clamp8(pixels[i + 2] * wbB);
    }
};
