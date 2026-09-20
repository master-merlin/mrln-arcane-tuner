/**
 * Frame-count rules for video-LoRA clip training.
 *
 * A video family states its trainable frame counts as an `Nn+M` rule
 * (`architecture_params["video.frame_rule"]`): a clip is valid when
 * `frames >= M` and `(frames - M) % N === 0`. The backend owns that contract
 * (`video_contract.frame_predicate`, parsing in
 * `BucketManager._parse_frame_step`); this module re-derives it so the UI's
 * verdict is the one training will reach. Invariant: the offset `M` is the
 * rule's FLOOR, not always 1 — a bare still satisfies `Nn+1` and nothing else.
 *
 * The rule string always comes from the active definition (served as
 * `DefinitionRef.frame_rule`); no family is named here. Without an active
 * rule the UI shows the generic fallback rows, so model-agnostic browsing
 * keeps its guidance.
 *
 * These helpers are pure so the segment-preview table and the trim editor
 * compute the same pass/fail verdicts from a frame count.
 */

export interface FrameRule {
    /** Normalised display label (e.g. "4n+1"). */
    label: string;
    /** N in `Nn+M`: the distance between two valid frame counts. */
    step: number;
    /** M in `Nn+M`: the smallest valid frame count. */
    offset: number;
}

/** Mirrors the backend's `_FRAME_RULE_RE` (whitespace and case tolerated). */
const FRAME_RULE_RE = /^\s*(\d+)\s*n\s*\+\s*(\d+)\s*$/i;

/**
 * Parse an `Nn+M` rule. Null for a missing or unrecognised rule and for a
 * step or offset below 1 — the same inputs the backend parser rejects.
 */
export function parseFrameRule(rule: string | null | undefined): FrameRule | null {
    if (!rule) return null;
    const m = FRAME_RULE_RE.exec(rule);
    if (!m) return null;
    const step = Number.parseInt(m[1], 10);
    const offset = Number.parseInt(m[2], 10);
    if (!(step >= 1) || !(offset >= 1)) return null;
    return { label: `${step}n+${offset}`, step, offset };
}

/** True when `frames` is on the rule's ladder: M, M+N, M+2N, … */
export function passesRule(frames: number, rule: FrameRule): boolean {
    return Number.isInteger(frames) && frames >= rule.offset && (frames - rule.offset) % rule.step === 0;
}

/** Guidance rows shown when no model-aware definition supplies a rule. */
const FALLBACK_FRAME_RULES: readonly FrameRule[] = [
    { label: '4n+1', step: 4, offset: 1 },
    { label: '8n+1', step: 8, offset: 1 },
];

/**
 * The rows to render: exactly one, labelled with the active definition's
 * rule, when that rule parses; the generic fallback rows otherwise.
 */
export function frameRulesFor(activeRule: string | null | undefined): readonly FrameRule[] {
    const parsed = parseFrameRule(activeRule);
    return parsed ? [parsed] : FALLBACK_FRAME_RULES;
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
