/**
 * frame-rules — the general `Nn+M` frame rule, re-derived from the backend
 * contract (`video_contract.frame_predicate` / `BucketManager._parse_frame_step`).
 * The rule STRING always comes from a definition; nothing here names a family.
 */
import { estimateFrames, frameRulesFor, parseFrameRule, passesRule } from './frame-rules';

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

    it('estimateFrames rounds duration × fps and is 0 without fps', () => {
        expect(estimateFrames(0, 2, 24)).toBe(48);
        expect(estimateFrames(0, 2, undefined)).toBe(0);
        expect(estimateFrames(2, 2, 24)).toBe(0);
    });
});
