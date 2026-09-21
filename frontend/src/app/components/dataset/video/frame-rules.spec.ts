/**
 * frame-rules — the general `Nn+M` frame rule, re-derived from the backend
 * contract (`video_contract.frame_predicate` / `BucketManager._parse_frame_step`).
 * The rule STRING always comes from a definition; nothing here names a family.
 */
import {
    clipFrames, clipHealthCovers, estimateFrames, frameRulesFor, parseFrameRule, passesRule, ruleOutcome,
} from './frame-rules';

function rule(text: string) {
    const parsed = parseFrameRule(text);
    if (!parsed) throw new Error(`fixture rule did not parse: ${text}`);
    return parsed;
}

describe('frame-rules', () => {
    it('parses_Nn_plus_M', () => {
        expect(parseFrameRule('4n+1')).toEqual({ label: '4n+1', step: 4, offset: 1 });
        expect(parseFrameRule('17n+5')).toEqual({ label: '17n+5', step: 17, offset: 5 });
        // The backend regex tolerates whitespace and case; the label is normalised.
        expect(parseFrameRule(' 8 N + 1 ')).toEqual({ label: '8n+1', step: 8, offset: 1 });
    });

    it('rejects what the backend parser rejects (null = no constraint known)', () => {
        for (const bad of [null, undefined, '', 'n+1', '4n', '4n-1', '4n+0', '0n+1', '4.5n+1', 'abc']) {
            expect(parseFrameRule(bad)).toBeNull();
        }
    });

    it('passesRule_reproduces_4n1_and_8n1', () => {
        // Today's verdict was `frames >= 1 && frames % modulus === 1`.
        for (const [text, modulus] of [['4n+1', 4], ['8n+1', 8]] as const) {
            const r = rule(text);
            for (let frames = -2; frames <= 200; frames++) {
                expect(passesRule(frames, r)).toBe(frames >= 1 && frames % modulus === 1);
            }
        }
    });

    it('passesRule_17n5_accepts_ladder_rejects_97', () => {
        const r = rule('17n+5');
        for (const ok of [5, 22, 39, 56, 73, 90, 107]) {
            expect(passesRule(ok, r)).toBe(true);
        }
        // 1 is below the floor (a bare still is NOT valid); 97 is 4n+1/8n+1
        // valid but not on this ladder; 18 is `% 17 === 1`.
        for (const bad of [0, 1, 4, 18, 97, 106, 108, NaN, Infinity, 22.5]) {
            expect(passesRule(bad, r)).toBe(false);
        }
    });

    it('frameRulesFor: a parsable active rule yields exactly ONE row labelled with it', () => {
        expect(frameRulesFor('19n+5')).toEqual([{ label: '19n+5', step: 19, offset: 5 }]);
    });

    it('frameRulesFor: no / unparsable active rule falls back to the two generic rows', () => {
        for (const none of [null, undefined, '', 'garbage']) {
            expect(frameRulesFor(none).map(r => r.label)).toEqual(['4n+1', '8n+1']);
        }
    });

    it('estimateFrames is duration × fps and is 0 without fps', () => {
        expect(estimateFrames(0, 2, 24)).toBe(48);
        expect(estimateFrames(0, 2, undefined)).toBe(0);
        expect(estimateFrames(2, 2, 24)).toBe(0);
    });

    it('estimateFrames FLOORS, as ingestion does (int(eff_dur * fps)) — for every family', () => {
        // 0.98 s × 24 fps = 23.52: ingestion trains 23, a rounding UI said 24.
        expect(estimateFrames(0, 0.98, 24)).toBe(23);
        expect(estimateFrames(0, 0.98, 24, null)).toBe(23);
    });

    it('estimateFrames counts TRAINING frames on the stated ingestion clock', () => {
        // LANE-92 VERIFY 3.01: a 30 fps file under a 24 fps clock.
        expect(estimateFrames(0, 5 / 30, 30, 24)).toBe(4); // 5 source frames -> 4
        expect(estimateFrames(0, 3, 30, 24)).toBe(72); // 90 source frames -> 72
        // No clock stated (null / absent / 0): the clip keeps its own fps.
        expect(estimateFrames(0, 3, 30, null)).toBe(90);
        expect(estimateFrames(0, 3, 30)).toBe(90);
        expect(estimateFrames(0, 3, 30, 0)).toBe(90);
        // The clock needs no source fps: ingestion resamples whatever the file is.
        expect(estimateFrames(0, 3, undefined, 24)).toBe(72);
    });

    it('clipFrames keeps the SOURCE count beside the training count and says when they differ', () => {
        expect(clipFrames(0, 3, 30, 24)).toEqual({ training: 72, source: 90, clockFps: 24, sourceFps: 30, resampled: true });
        expect(clipFrames(0, 3, 24, 24)).toEqual({ training: 72, source: 72, clockFps: 24, sourceFps: 24, resampled: false });
        expect(clipFrames(0, 3, 30, null)).toEqual({ training: 90, source: 90, clockFps: 30, sourceFps: 30, resampled: false });
    });

    it('ruleOutcome: below the floor is skipped; off the ladder is cut to the largest legal length', () => {
        const rule = parseFrameRule('17n+5')!;
        expect(ruleOutcome(4, rule)).toEqual({ used: 0, skipped: true, cut: false });
        expect(ruleOutcome(72, rule)).toEqual({ used: 56, skipped: false, cut: true });
        expect(ruleOutcome(56, rule)).toEqual({ used: 56, skipped: false, cut: false });
        expect(ruleOutcome(5, rule)).toEqual({ used: 5, skipped: false, cut: false });
    });

    it('clipHealthCovers: only the rules the backend clip health checks (4n+1, 8n+1), or no active rule', () => {
        expect(clipHealthCovers(null)).toBe(true);
        expect(clipHealthCovers('4n+1')).toBe(true);
        expect(clipHealthCovers(' 8N + 1 ')).toBe(true);
        expect(clipHealthCovers('17n+5')).toBe(false);
    });
});
