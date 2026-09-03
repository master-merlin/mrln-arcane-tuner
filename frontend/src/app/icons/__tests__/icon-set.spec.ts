import { describe, expect, it } from 'vitest';

import { iconSet, type IconKey } from '../icon-set';

/**
 * The icon set is finite on purpose, and this is half of what keeps it so.
 *
 * On 2026-09-03 a `@lucide/angular` minor bump (1.18.0 -> 1.37.0) took the
 * initial bundle from 1.68 MB to 6.34 MB raw and broke the production build's
 * budget, with no change to this app at all. The cause was `import { icons }`
 * — the barrel of every icon Lucide ships — looked up with a runtime-computed
 * key, which no bundler can narrow. Replacing it with a named map of the icons
 * actually used took main to 538 kB, below even the pre-bump 1.68 MB.
 *
 * Two things could undo that:
 *
 *   1. `IconKey` widens to `string`, so a requested icon is no longer proven
 *      to exist and every typo becomes a blank space in front of a user. That
 *      is checked here, where the type lives.
 *   2. Someone re-adds `import { icons }` to "just get all of them". Catching
 *      that needs a scan of every source file, and this environment has no
 *      filesystem — the Angular test tsconfig carries no node types on
 *      purpose. So it lives in `backend/tests/test_frontend_icon_set.py`,
 *      beside the repo's other whole-tree scanners, where it runs in ~1 s
 *      every gate. Split deliberately; neither half covers the other.
 */

describe('the icon set stays finite', () => {
    it('IconKey is the exact set of listed names, not string', () => {
        const keys = Object.keys(iconSet);
        expect(keys.length).toBeGreaterThan(0);
        expect(keys).toContain('Database');

        // @ts-expect-error — a name outside the set must not typecheck. This
        // directive FAILS THE BUILD if the expression ever becomes legal,
        // which is precisely the widening we want to hear about. It is the
        // real assertion in this test; the runtime check below only stops the
        // line from being dead code.
        const bogus: IconKey = 'NotARealIconName';
        expect(keys).not.toContain(bogus);
    });

    it('every entry resolves to real icon data', () => {
        // The map is generated from a scan, so a wrong entry is a plausible
        // mistake: a name that exists but points at the wrong import, or an
        // import that silently became undefined after an upstream rename.
        for (const [name, cmp] of Object.entries(iconSet)) {
            expect(cmp, `iconSet.${name} is not defined`).toBeDefined();
            const data = (cmp as unknown as { icon?: { node?: unknown[] } }).icon;
            expect(data, `iconSet.${name} has no static icon data`).toBeDefined();
            expect(
                Array.isArray(data?.node),
                `iconSet.${name}.icon.node is not an array of SVG nodes`,
            ).toBe(true);
        }
    });

    it('entries are sorted, so the generated file keeps diffing cleanly', () => {
        const keys = Object.keys(iconSet);
        expect(keys).toEqual([...keys].sort());
    });
});
