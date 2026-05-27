// frontend/src/app/workspace/modes/edit/panels/hsl.math.ts
import type { ApplyFn } from '../preview/preview-types';
import type { HslParams } from '../operation-defs';
import { rgbToHsl, hslToRgb } from '../preview/utils/color';

/**
 * Per-color-range HSL adjustments. Each named range has a center hue
 * (degrees) and a half-width; pixels within the range receive
 * hue_shift / saturation / luminance deltas weighted by a cosine
 * falloff outside the band center. Matches legacy band definitions and
 * weights exactly.
 */
const HSL_RANGES: Record<string, [number, number]> = {
    reds:     [0,   30],
    oranges:  [30,  30],
    yellows:  [60,  30],
    greens:   [120, 40],
    cyans:    [180, 30],
    blues:    [240, 40],
    purples:  [285, 30],
    magentas: [330, 30],
};

export const applyHslSelective: ApplyFn<HslParams> = (pixels, _w, _h, params) => {
    const entries = Object.entries(params).filter(([_, adj]) =>
        Math.abs(adj.hue_shift) > 0.001 ||
        Math.abs(adj.saturation) > 0.001 ||
        Math.abs(adj.luminance) > 0.001
    );
    if (entries.length === 0) return;

    for (let i = 0; i < pixels.length; i += 4) {
        let [h, s, l] = rgbToHsl(pixels[i], pixels[i + 1], pixels[i + 2]);
        if (s <= 0.01) continue;  // achromatic — skip

        const hueDeg = h * 360;
        for (const [rangeName, adj] of entries) {
            const range = HSL_RANGES[rangeName];
            if (!range) continue;
            const [center, width] = range;
            let d = Math.abs(hueDeg - center);
            d = Math.min(d, 360 - d);
            if (d > width * 1.5) continue;

            const falloff = Math.max(0, Math.min(1, 1 - (d - width) / (width * 0.5 + 0.001)));
            const weight = 0.5 * (1 + Math.cos(Math.PI * (1 - falloff)));
            h = ((h + (adj.hue_shift / 360) * weight) % 1 + 1) % 1;
            s = Math.max(0, Math.min(1, s + (adj.saturation / 100) * weight));
            l = Math.max(0, Math.min(1, l + (adj.luminance / 200) * weight));
        }

        const [r2, g2, b2] = hslToRgb(h, s, l);
        pixels[i]     = r2;
        pixels[i + 1] = g2;
        pixels[i + 2] = b2;
    }
};
