// frontend/src/app/workspace/modes/edit/preview/utils/lut.ts
import type { CurvePoint } from '../../operation-defs';
import type { CubeLut } from '../preview-types';

const CATMULL_TENSION = 0.4;

/**
 * Build a 256-entry channel LUT from CurvePoint anchors using a
 * Catmull-Rom spline (matches the legacy curves editor exactly).
 */
export function buildChannelLUT(points: CurvePoint[]): Uint8Array {
    const lut = new Uint8Array(256);
    if (points.length < 2) {
        for (let i = 0; i < 256; i++) lut[i] = i;
        return lut;
    }

    const sorted = [...points].sort((a, b) => a.x - b.x);
    const xs = sorted.map(p => p.x);
    const ys = sorted.map(p => p.y);
    const tau = 1 - CATMULL_TENSION;

    for (let i = 0; i < 256; i++) {
        if (i <= xs[0]) { lut[i] = ys[0]; continue; }
        if (i >= xs[xs.length - 1]) { lut[i] = ys[ys.length - 1]; continue; }

        let seg = 0;
        while (seg < xs.length - 2 && xs[seg + 1] < i) seg++;

        const x0 = xs[seg], x1 = xs[seg + 1];
        const y0 = ys[seg], y1 = ys[seg + 1];
        const t = (i - x0) / (x1 - x0);

        const ym1 = seg > 0 ? ys[seg - 1] : 2 * y0 - y1;
        const y2 = seg < xs.length - 2 ? ys[seg + 2] : 2 * y1 - y0;
        const t2 = t * t, t3 = t2 * t;

        const val = tau * 0.5 * (
            (2 * y0) + (-ym1 + y1) * t +
            (2 * ym1 - 5 * y0 + 4 * y1 - y2) * t2 +
            (-ym1 + 3 * y0 - 3 * y1 + y2) * t3
        ) + (1 - tau) * (y0 + (y1 - y0) * t);

        lut[i] = Math.max(0, Math.min(255, Math.round(val)));
    }
    return lut;
}

/**
 * Trilinear interpolation against a 3D CUBE LUT. Inputs are 0..255 RGB;
 * outputs are 0..255 RGB (rounded, clamped). Matches legacy behavior.
 */
export function trilinearInterp(
    lut: CubeLut,
    r: number,
    g: number,
    b: number,
): [number, number, number] {
    const s = lut.size;
    const scale = (s - 1) / 255;

    const rf = r * scale, gf = g * scale, bf = b * scale;
    const r0 = Math.floor(rf), g0 = Math.floor(gf), b0 = Math.floor(bf);
    const r1 = Math.min(r0 + 1, s - 1);
    const g1 = Math.min(g0 + 1, s - 1);
    const b1 = Math.min(b0 + 1, s - 1);
    const dr = rf - r0, dg = gf - g0, db = bf - b0;

    const idx = (ri: number, gi: number, bi: number) => (bi * s * s + gi * s + ri) * 3;
    const c000 = idx(r0, g0, b0), c100 = idx(r1, g0, b0);
    const c010 = idx(r0, g1, b0), c110 = idx(r1, g1, b0);
    const c001 = idx(r0, g0, b1), c101 = idx(r1, g0, b1);
    const c011 = idx(r0, g1, b1), c111 = idx(r1, g1, b1);

    const t = lut.table;
    const result: [number, number, number] = [0, 0, 0];
    for (let ch = 0; ch < 3; ch++) {
        const v000 = t[c000 + ch], v100 = t[c100 + ch];
        const v010 = t[c010 + ch], v110 = t[c110 + ch];
        const v001 = t[c001 + ch], v101 = t[c101 + ch];
        const v011 = t[c011 + ch], v111 = t[c111 + ch];

        const c00 = v000 * (1 - dr) + v100 * dr;
        const c10 = v010 * (1 - dr) + v110 * dr;
        const c01 = v001 * (1 - dr) + v101 * dr;
        const c11 = v011 * (1 - dr) + v111 * dr;

        const c0 = c00 * (1 - dg) + c10 * dg;
        const c1 = c01 * (1 - dg) + c11 * dg;

        result[ch] = Math.max(0, Math.min(255, Math.round((c0 * (1 - db) + c1 * db) * 255)));
    }
    return result;
}

/**
 * Parse a .cube file's text content into a 3D LUT. Returns null on
 * malformed input or unsupported LUT_1D_SIZE. Matches legacy parser.
 */
export function parseCubeString(content: string): CubeLut | null {
    const lines = content.split(/\r?\n/);
    let size = 0;
    const entries: number[] = [];

    for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line || line.startsWith('#') || line.startsWith('TITLE') ||
            line.startsWith('DOMAIN_MIN') || line.startsWith('DOMAIN_MAX')) continue;

        if (line.startsWith('LUT_3D_SIZE')) {
            size = parseInt(line.split(/\s+/)[1], 10);
            continue;
        }
        if (line.startsWith('LUT_1D_SIZE')) return null;  // 1D not supported

        const parts = line.split(/\s+/).map(Number);
        if (parts.length >= 3 && !isNaN(parts[0])) {
            entries.push(parts[0], parts[1], parts[2]);
        }
    }

    if (size === 0 || entries.length !== size * size * size * 3) return null;
    return { size, table: new Float32Array(entries) };
}
