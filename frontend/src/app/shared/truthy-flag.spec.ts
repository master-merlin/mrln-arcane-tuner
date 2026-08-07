import { describe, it, expect } from 'vitest';
import { isTruthyFlag } from './truthy-flag';

/**
 * `isTruthyFlag` is NOT a pure extraction of what the training form used to
 * do inline: it trims, it lower-cases, and it reads the string `"0"` as false.
 * Those three rules changed the answer at the pre-existing
 * `isAdaptiveTargetingOn()` call site, where a config or template can hand the
 * form a round-tripped string instead of a boolean — so they are pinned here
 * rather than left to the next reader to infer from the implementation.
 */
describe('isTruthyFlag', () => {
    it.each([
        // [label, input, expected]
        ['a real boolean true', true, true],
        ['a real boolean false', false, false],
        ['the string "true" (round-tripped through a template/JSON editor)', 'true', true],
        ['the string "false" — truthy as a string, and the whole reason this exists',
            'false', false],
        ['the string "0"', '0', false],
        ['the string "1"', '1', true],
        ['whitespace around "true"', ' true ', true],
        ['whitespace around "false"', '  false\n', false],
        ['upper-case "FALSE"', 'FALSE', false],
        ['the empty string', '', false],
        ['an absent value (undefined)', undefined, false],
        ['an explicit null', null, false],
        ['the number 0', 0, false],
        ['the number 1', 1, true],
    ])('reads %s as %s → %s', (_label, input, expected) => {
        expect(isTruthyFlag(input)).toBe(expected);
    });
});
