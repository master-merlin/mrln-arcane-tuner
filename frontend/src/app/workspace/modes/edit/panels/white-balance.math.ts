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

/**
 * Gray-world auto white-balance: given the source image's average R/G/B,
 * find (temperature, tint) that neutralizes the cast. wbR/wbB is monotonic
 * in temperature (low T = warm-source correction → wbR<1, wbB>1), so a
 * binary search converges to the T whose R/B correction ratio matches
 * the target avgB/avgR. Tint then nudges green toward the (R,B) midpoint.
 */
export function estimateAutoWB(
    avgR: number, avgG: number, avgB: number,
): { temperature: number; tint: number } {
    if (avgR <= 0 || avgG <= 0 || avgB <= 0) return { temperature: 6500, tint: 0 };
    const targetRBRatio = avgB / avgR;

    let lo = 2000, hi = 12000;
    for (let i = 0; i < 32; i++) {
        const mid = (lo + hi) / 2;
        const { wbR, wbB } = computeWBFactors(mid, 0);
        if (wbR / wbB < targetRBRatio) lo = mid;
        else hi = mid;
    }
    const temperature = Math.max(2000, Math.min(12000,
        Math.round(((lo + hi) / 2) / 100) * 100));

    const { wbR, wbG, wbB } = computeWBFactors(temperature, 0);
    const corrR = avgR * wbR, corrG = avgG * wbG, corrB = avgB * wbB;
    const targetG = (corrR + corrB) / 2;
    // Invert wbG's tint coefficient (1 + tf*0.3) where tf = tint/100.
    const tfRaw = (targetG / corrG - 1) / 0.3;
    const tint = Math.max(-100, Math.min(100, Math.round(tfRaw * 100)));

    return { temperature, tint };
}
