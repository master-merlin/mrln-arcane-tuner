// frontend/src/app/workspace/modes/edit/preview/preview-types.ts

/**
 * The uniform signature every per-op math module exports.
 *
 * Mutates `pixels` (RGBA Uint8ClampedArray, length = w*h*4) in place.
 * For ops that genuinely cannot operate in-place (spatial warps that
 * need an untouched source while writing the output), the module may
 * allocate its own scratch buffer internally and write back into
 * `pixels` at the end — the contract stays the same from the caller's
 * point of view.
 */
export type ApplyFn<P> = (
    pixels: Uint8ClampedArray,
    w: number,
    h: number,
    params: P,
) => void;

/** Parsed .cube file as kept in PipelineEditorState's transient cache. */
export interface CubeLut {
    size: number;
    table: Float32Array;  // length = size * size * size * 3 (RGB triples)
}

export interface HistogramData {
    r: number[];
    g: number[];
    b: number[];
    luminance: number[];
}
