// frontend/src/app/workspace/modes/edit/panels/color-tone.math.ts
import type { ApplyFn } from '../preview/preview-types';
import type { ColorToneParams } from '../operation-defs';
import { clamp8, rgbToHsl, hslToRgb } from '../preview/utils/color';

/**
 * Apply hue shift (degrees), saturation multiplier, and contrast in
 * one pass. Matches the legacy editor's color-tone block which fans
 * out to backend `hue_saturation` + `contrast` blocks on save.
 *
 * Identity: hue_shift=0, saturation=1, contrast=1.
 */
export const applyColorTone: ApplyFn<ColorToneParams> = (pixels, _w, _h, params) => {
    const { hue_shift, saturation, contrast } = params;
    const hsActive = hue_shift !== 0 || saturation !== 1;
    const cActive = contrast !== 1;
    if (!hsActive && !cActive) return;

    for (let i = 0; i < pixels.length; i += 4) {
        let r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];

        if (hsActive) {
            const [h, s, l] = rgbToHsl(r, g, b);
            const newH = ((h + hue_shift / 360) % 1 + 1) % 1;
            const newS = Math.max(0, Math.min(1, s * saturation));
            [r, g, b] = hslToRgb(newH, newS, l);
        }

        if (cActive) {
            r = clamp8(contrast * (r - 128) + 128);
            g = clamp8(contrast * (g - 128) + 128);
            b = clamp8(contrast * (b - 128) + 128);
        }

        pixels[i]     = r;
        pixels[i + 1] = g;
        pixels[i + 2] = b;
    }
};
