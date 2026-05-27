// frontend/src/app/workspace/modes/edit/preview/utils/blur.ts

/**
 * Two-pass separable box blur as a fast Gaussian approximation.
 * Returns a new Uint8ClampedArray; does not mutate `src`. Used by
 * sharpen.math.ts (unsharp mask + high-pass).
 *
 * Algorithm matches the legacy editor's `boxBlur` exactly.
 */
export function boxBlur(
    src: Uint8ClampedArray,
    w: number,
    h: number,
    radius: number,
): Uint8ClampedArray {
    const temp = new Uint8ClampedArray(src);
    const out = new Uint8ClampedArray(src);
    const r = Math.max(1, Math.round(radius));
    const size = r * 2 + 1;

    // Horizontal pass: src → temp
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            let rSum = 0, gSum = 0, bSum = 0;
            for (let k = -r; k <= r; k++) {
                const sx = Math.max(0, Math.min(w - 1, x + k));
                const idx = (y * w + sx) * 4;
                rSum += src[idx];
                gSum += src[idx + 1];
                bSum += src[idx + 2];
            }
            const idx = (y * w + x) * 4;
            temp[idx]     = rSum / size;
            temp[idx + 1] = gSum / size;
            temp[idx + 2] = bSum / size;
            temp[idx + 3] = src[idx + 3];
        }
    }

    // Vertical pass: temp → out
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            let rSum = 0, gSum = 0, bSum = 0;
            for (let k = -r; k <= r; k++) {
                const sy = Math.max(0, Math.min(h - 1, y + k));
                const idx = (sy * w + x) * 4;
                rSum += temp[idx];
                gSum += temp[idx + 1];
                bSum += temp[idx + 2];
            }
            const idx = (y * w + x) * 4;
            out[idx]     = rSum / size;
            out[idx + 1] = gSum / size;
            out[idx + 2] = bSum / size;
            out[idx + 3] = temp[idx + 3];
        }
    }

    return out;
}
