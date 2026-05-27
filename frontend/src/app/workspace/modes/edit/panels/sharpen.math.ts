// frontend/src/app/workspace/modes/edit/panels/sharpen.math.ts
import type { ApplyFn } from '../preview/preview-types';
import type { SharpenParams } from '../operation-defs';
import { clamp8 } from '../preview/utils/color';
import { boxBlur } from '../preview/utils/blur';

function applyUnsharp(
    pixels: Uint8ClampedArray, w: number, h: number,
    radius: number, amount: number, threshold: number,
): void {
    const blurred = boxBlur(pixels, w, h, radius);
    const orig = new Uint8ClampedArray(pixels);
    for (let i = 0; i < pixels.length; i += 4) {
        for (let ch = 0; ch < 3; ch++) {
            const o = orig[i + ch];
            const diff = o - blurred[i + ch];
            pixels[i + ch] = Math.abs(diff) >= threshold
                ? clamp8(o + diff * amount)
                : o;
        }
    }
}

function applyKernel(
    pixels: Uint8ClampedArray, w: number, h: number, strength: number,
): void {
    const src = new Uint8ClampedArray(pixels);
    const s = strength;
    const center = 1 + 4 * s;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const idx = (y * w + x) * 4;
            for (let ch = 0; ch < 3; ch++) {
                const c = src[idx + ch] * center;
                const t = y > 0 ? src[((y - 1) * w + x) * 4 + ch] : src[idx + ch];
                const b = y < h - 1 ? src[((y + 1) * w + x) * 4 + ch] : src[idx + ch];
                const l = x > 0 ? src[(y * w + x - 1) * 4 + ch] : src[idx + ch];
                const r = x < w - 1 ? src[(y * w + x + 1) * 4 + ch] : src[idx + ch];
                pixels[idx + ch] = clamp8(c - s * (t + b + l + r));
            }
        }
    }
}

function applyHighPass(
    pixels: Uint8ClampedArray, w: number, h: number,
    radius: number, strength: number,
): void {
    const blurred = boxBlur(pixels, w, h, radius);
    const orig = new Uint8ClampedArray(pixels);
    for (let i = 0; i < pixels.length; i += 4) {
        for (let ch = 0; ch < 3; ch++) {
            const o = orig[i + ch];
            const hp = o - blurred[i + ch] + 128;
            const blended = o < 128
                ? (2 * o * hp) / 255
                : 255 - (2 * (255 - o) * (255 - hp)) / 255;
            pixels[i + ch] = clamp8(o + (blended - o) * strength);
        }
    }
}

/**
 * Sharpening: dispatches by method. Identity skip: caller can also
 * skip by leaving the op disabled, but the function is safe to call
 * with any params — each method has its own no-op fast paths via the
 * radius/strength values.
 *
 * Maps SharpenParams.amount (0..200 from the panel slider, where the
 * legacy used percent) to a 0..2 multiplier for the unsharp method.
 */
export const applySharpen: ApplyFn<SharpenParams> = (pixels, w, h, params) => {
    switch (params.method) {
        case 'unsharp':
            applyUnsharp(pixels, w, h, params.radius, params.amount / 100, params.threshold);
            return;
        case 'kernel':
            applyKernel(pixels, w, h, params.strength);
            return;
        case 'high_pass':
            applyHighPass(pixels, w, h, params.radius, params.strength);
            return;
    }
};
