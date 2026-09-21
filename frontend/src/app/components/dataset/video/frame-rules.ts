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
 * TRAINING frame count over a [start, end] window: the number ingestion
 * reaches and the frame rule judges. Mirrors the trainer's
 * `available_frames = int(eff_dur * vid_target_fps)` (`pipeline_data.py`)
 * exactly — FLOOR, never round, for every family — on the clock ingestion
 * uses: the definition's stated `ingestFps` when there is one (every clip is
 * resampled to it), the clip's own `sourceFps` otherwise.
 *
 * Returns 0 for a non-positive window or when neither clock is known (so the
 * UI shows a neutral "—" rather than NaN). Not modelled: a run's
 * `target_fps` override and frame stride — run settings, unknown here.
 */
export function estimateFrames(
    startS: number,
    endS: number,
    sourceFps: number | undefined,
    ingestFps?: number | null,
): number {
    const fps = ingestFps && ingestFps > 0 ? ingestFps : sourceFps;
    if (!fps || fps <= 0) return 0;
    const dur = endS - startS;
    if (!Number.isFinite(dur) || dur <= 0) return 0;
    return Math.floor(dur * fps);
}

/** A window's frame counts on both clocks: what trains, and what the file holds. */
export interface ClipFrames {
    /** Frames ingestion trains on — the number the frame rule judges. */
    training: number;
    /** Frames the FILE holds over the window, at its own fps. */
    source: number;
    /** The fps `training` was counted on (0 when neither clock is known). */
    clockFps: number;
    /** The file's own fps (0 when unknown). */
    sourceFps: number;
    /** True when a stated ingestion clock differs from the file's fps. */
    resampled: boolean;
}

/** Mirrors ingestion's `abs(source_fps - target_fps) > 1e-6` resample test. */
const FPS_EPSILON = 1e-6;

export function clipFrames(
    startS: number,
    endS: number,
    sourceFps: number | undefined,
    ingestFps: number | null | undefined,
): ClipFrames {
    const src = sourceFps && sourceFps > 0 ? sourceFps : 0;
    const clock = ingestFps && ingestFps > 0 ? ingestFps : 0;
    return {
        training: estimateFrames(startS, endS, sourceFps, ingestFps),
        source: estimateFrames(startS, endS, sourceFps),
        clockFps: clock || src,
        sourceFps: src,
        resampled: clock > 0 && src > 0 && Math.abs(src - clock) > FPS_EPSILON,
    };
}

/** An fps for display: at most two decimals, no trailing zeros (24, 29.97). */
export function fpsLabel(fps: number): string {
    return String(Number(fps.toFixed(2)));
}

/** What training does with a clip of `frames` training frames under `rule`. */
export interface RuleOutcome {
    /** Frames used: the largest ladder value <= frames; 0 when skipped. */
    used: number;
    /** Below the rule's floor: ingestion skips the clip (`short_clip_skipped`). */
    skipped: boolean;
    /** Off the ladder: ingestion cuts the clip down (`clip_frames_snapped`). */
    cut: boolean;
}

/**
 * Upper bound: a run's / dataset's own frame cap can cut further, and that
 * cap is the user's setting, not the rule's verdict.
 */
export function ruleOutcome(frames: number, rule: FrameRule): RuleOutcome {
    if (!(frames >= rule.offset)) return { used: 0, skipped: true, cut: false };
    const used = rule.offset + Math.floor((frames - rule.offset) / rule.step) * rule.step;
    return { used, skipped: false, cut: used < frames };
}

/**
 * The rules the backend clip health checks (`clip_health.FAMILY_RULES`:
 * 4k+1 and 8k+1). It does not know the active definition, so a pass there
 * says nothing about any other rule.
 */
const CLIP_HEALTH_CHECKED_RULES: readonly string[] = ['4n+1', '8n+1'];

/** Label of the rules a clip-health result covers, for the UI to name. */
export const CLIP_HEALTH_CHECKED_LABEL = CLIP_HEALTH_CHECKED_RULES.join(' / ');

/** True when a clip-health pass covers `activeRule` (or no rule is active). */
export function clipHealthCovers(activeRule: string | null | undefined): boolean {
    const parsed = parseFrameRule(activeRule);
    return !parsed || CLIP_HEALTH_CHECKED_RULES.includes(parsed.label);
}
