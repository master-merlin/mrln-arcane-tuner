// frontend/src/app/workspace/modes/edit/preview/utils/color.ts

/** Clamp a number to [0, 255] and round to integer. */
export function clamp8(n: number): number {
    return n < 0 ? 0 : n > 255 ? 255 : Math.round(n);
}

/** RGB (0..255) → HSL (h,s,l in 0..1). */
export function rgbToHsl(r: number, g: number, b: number): [number, number, number] {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const l = (max + min) / 2;
    if (max === min) return [0, 0, l];
    const d = max - min;
    const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    let h = 0;
    if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
    else if (max === g) h = ((b - r) / d + 2) / 6;
    else h = ((r - g) / d + 4) / 6;
    return [h, s, l];
}

/** HSL (0..1) → RGB (0..255, integer). */
export function hslToRgb(h: number, s: number, l: number): [number, number, number] {
    if (s === 0) {
        const v = Math.round(l * 255);
        return [v, v, v];
    }
    const hue2rgb = (p: number, q: number, t: number): number => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    return [
        Math.round(hue2rgb(p, q, h + 1 / 3) * 255),
        Math.round(hue2rgb(p, q, h) * 255),
        Math.round(hue2rgb(p, q, h - 1 / 3) * 255),
    ];
}

/**
 * White-balance RGB multipliers from a target temperature (K) and tint.
 * Tannenbaum/Bartleson approximation matches the legacy editor exactly
 * — change at your own risk; it's calibrated against the backend.
 */
export function computeWBFactors(
    temperature: number,
    tint: number,
): { wbR: number; wbG: number; wbB: number } {
    const kelvinToRgb = (temp: number): [number, number, number] => {
        const t = Math.max(1000, Math.min(40000, temp)) / 100;
        const red = t <= 66 ? 1.0 : Math.min(1, Math.max(0, 329.698727446 * Math.pow(t - 60, -0.1332047592) / 255));
        const green = t <= 66
            ? Math.min(1, Math.max(0, (99.4708025861 * Math.log(t) - 161.1195681661) / 255))
            : Math.min(1, Math.max(0, 288.1221695283 * Math.pow(t - 60, -0.0755148492) / 255));
        const blue = t >= 66 ? 1.0 : t <= 19 ? 0.0 : Math.min(1, Math.max(0, (138.5177312231 * Math.log(t - 10) - 305.0447927307) / 255));
        return [red, green, blue];
    };

    const [tR, tG, tB] = kelvinToRgb(temperature);
    const [nR, nG, nB] = kelvinToRgb(6500);
    let rScale = nR / Math.max(tR, 0.001);
    let gScale = nG / Math.max(tG, 0.001);
    let bScale = nB / Math.max(tB, 0.001);

    const tf = tint / 100;
    gScale *= (1 + tf * 0.3);
    rScale *= (1 - tf * 0.1);
    bScale *= (1 - tf * 0.1);

    return { wbR: rScale, wbG: gScale, wbB: bScale };
}
