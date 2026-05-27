// frontend/src/app/workspace/modes/edit/panels/vignette.math.ts
import type { ApplyFn } from '../preview/preview-types';
import type { VignetteParams } from '../operation-defs';
import { clamp8 } from '../preview/utils/color';

/**
 * Radial vignette. amount > 0 = darken corners; amount < 0 = brighten.
 * midpoint and feather are 0..1 (fraction of normalized radius).
 *
 * Note: `shape` and `apply_before_lut` from VignetteParams are honored
 * by the backend on Save. For preview, shape=='circular' is the only
 * implemented mode; rectangular falls back to circular here.
 */
export const applyVignette: ApplyFn<VignetteParams> = (pixels, w, h, params) => {
    if (params.amount === 0) return;
    const { amount, midpoint, feather } = params;
    const cx = w / 2, cy = h / 2;
    const featherVal = Math.max(feather, 0.01);

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const dx = (x - cx) / cx;
            const dy = (y - cy) / cy;
            const radius = Math.sqrt(dx * dx + dy * dy) / Math.SQRT2;
            const mask = Math.max(0, Math.min(1, (radius - midpoint) / featherVal));
            const mult = amount > 0 ? (1 - amount * mask) : (1 + Math.abs(amount) * mask);

            const idx = (y * w + x) * 4;
            pixels[idx]     = clamp8(pixels[idx]     * mult);
            pixels[idx + 1] = clamp8(pixels[idx + 1] * mult);
            pixels[idx + 2] = clamp8(pixels[idx + 2] * mult);
        }
    }
};
