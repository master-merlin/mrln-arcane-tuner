/**
 * Frame-count "family" rules for video-LoRA clip training.
 *
 * Many video diffusion trainers require a clip's trainable frame count to
 * satisfy `frames ≡ 1 (mod n)` for some n (the temporal-compression stride
 * of the VAE). The two families surfaced in the UI are:
 *   - "4n+1"  (frames % 4 === 1) — e.g. 5, 9, 13, … 81
 *   - "8n+1"  (frames % 8 === 1) — e.g. 9, 17, 25, … 81
 *
 * These helpers are pure so the segment-preview table, the trim editor, and
 * the modals all compute the same pass/fail verdicts from a frame count.
 */

export interface FrameFamily {
    /** Display label (e.g. "4n+1"). */
    label: string;
    /** The modulus n in `frames % n === 1`. */
    modulus: number;
}

/** The frame-rule families surfaced across the video-curation UI. */
export const FRAME_FAMILIES: readonly FrameFamily[] = [
    { label: '4n+1', modulus: 4 },
    { label: '8n+1', modulus: 8 },
] as const;

/** True when `frames` satisfies `frames % family.modulus === 1` (and ≥ 1). */
export function passesFamily(frames: number, family: FrameFamily): boolean {
    return Number.isFinite(frames) && frames >= 1 && frames % family.modulus === 1;
}

/**
 * Estimated trainable frame count over a [start, end] window at `fps`.
 * Rounds the duration × fps product; returns 0 for a non-positive window
 * or a missing/zero fps (so the UI shows a neutral "—" rather than NaN).
 */
export function estimateFrames(startS: number, endS: number, fps: number | undefined): number {
    if (!fps || fps <= 0) return 0;
    const dur = endS - startS;
    if (!Number.isFinite(dur) || dur <= 0) return 0;
    return Math.round(dur * fps);
}
