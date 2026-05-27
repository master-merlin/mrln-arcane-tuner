// frontend/src/app/workspace/modes/edit/panels/lens.math.ts
import type { ApplyFn } from '../preview/preview-types';
import type { LensParams } from '../operation-defs';

/** Solve the 8-coefficient inverse perspective transform via Gaussian elimination. */
function computePerspectiveCoeffs(dst: number[], src: number[]): number[] {
    const A: number[][] = [];
    const B: number[] = [];
    for (let i = 0; i < 4; i++) {
        const dx = dst[i * 2], dy = dst[i * 2 + 1];
        const sx = src[i * 2], sy = src[i * 2 + 1];
        A.push([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy]);
        A.push([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy]);
        B.push(sx);
        B.push(sy);
    }
    const n = 8;
    const M = A.map((row, i) => [...row, B[i]]);
    for (let col = 0; col < n; col++) {
        let maxRow = col;
        for (let row = col + 1; row < n; row++) {
            if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
        }
        [M[col], M[maxRow]] = [M[maxRow], M[col]];
        const pivot = M[col][col];
        if (Math.abs(pivot) < 1e-12) continue;
        for (let j = col; j <= n; j++) M[col][j] /= pivot;
        for (let row = 0; row < n; row++) {
            if (row === col) continue;
            const factor = M[row][col];
            for (let j = col; j <= n; j++) M[row][j] -= factor * M[col][j];
        }
    }
    return M.map(row => row[n]);
}

/**
 * Barrel/pincushion + keystone correction by reverse mapping with
 * bilinear interpolation. Out-of-bounds source samples become black.
 * Matches legacy `applyLensCorrectionToCanvas`. `auto_crop` from
 * LensParams is a backend-only post-process — preview does not crop.
 */
export const applyLensCorrection: ApplyFn<LensParams> = (pixels, w, h, params) => {
    const { barrel, v_keystone, h_keystone } = params;
    if (barrel === 0 && v_keystone === 0 && h_keystone === 0) return;

    const src = new Uint8ClampedArray(pixels);
    const cx = w / 2, cy = h / 2;
    const k = barrel * 0.5;

    const hasKeystone = v_keystone !== 0 || h_keystone !== 0;
    let perspCoeffs: number[] | null = null;
    if (hasKeystone) {
        const vk = Math.tan(v_keystone * Math.PI / 360);
        const hk = Math.tan(h_keystone * Math.PI / 360);
        const x0 = hk * w * 0.5;
        const y0 = vk * h * 0.5;
        perspCoeffs = computePerspectiveCoeffs(
            [0, 0, w, 0, w, h, 0, h],
            [x0, y0, w - x0, -y0, w + x0, h + y0, -x0, h - y0],
        );
    }

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            let sx = x, sy = y;

            if (barrel !== 0) {
                const xn = (sx - cx) / cx;
                const yn = (sy - cy) / cy;
                const r = Math.sqrt(xn * xn + yn * yn);
                if (r > 0) {
                    const rNew = r * (1 + k * r * r);
                    const factor = rNew / r;
                    sx = cx + (sx - cx) * factor;
                    sy = cy + (sy - cy) * factor;
                }
            }

            if (perspCoeffs) {
                const c = perspCoeffs;
                const denom = c[6] * sx + c[7] * sy + 1;
                if (Math.abs(denom) > 1e-10) {
                    sx = (c[0] * sx + c[1] * sy + c[2]) / denom;
                    sy = (c[3] * sx + c[4] * sy + c[5]) / denom;
                }
            }

            const outIdx = (y * w + x) * 4;
            if (sx < 0 || sx >= w - 1 || sy < 0 || sy >= h - 1) {
                pixels[outIdx]     = 0;
                pixels[outIdx + 1] = 0;
                pixels[outIdx + 2] = 0;
                pixels[outIdx + 3] = 255;
            } else {
                const x0 = Math.floor(sx), y0 = Math.floor(sy);
                const fx = sx - x0, fy = sy - y0;
                const i00 = (y0 * w + x0) * 4;
                const i10 = i00 + 4;
                const i01 = ((y0 + 1) * w + x0) * 4;
                const i11 = i01 + 4;
                for (let c = 0; c < 3; c++) {
                    pixels[outIdx + c] = Math.round(
                        src[i00 + c] * (1 - fx) * (1 - fy) +
                        src[i10 + c] * fx * (1 - fy) +
                        src[i01 + c] * (1 - fx) * fy +
                        src[i11 + c] * fx * fy,
                    );
                }
                pixels[outIdx + 3] = 255;
            }
        }
    }
};
