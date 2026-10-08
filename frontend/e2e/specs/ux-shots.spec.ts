import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { test, expect } from '../fixtures/test';
import { latestMetrics } from '../../src/app/shared/job-metrics';
import { jobs, runningJobStepLogs } from '../fixtures/api-data';

/**
 * LANE-143 UAT evidence producer. Runs ONLY with `UX_SHOTS=1`.
 *
 *   UX_SHOTS=1 UX_SHOTS_SET=L5 [UX_SHOTS_BASELINE=L4] \
 *     npm --prefix frontend run e2e -- ux-shots --workers=1
 *
 * Writes `.agent/workdir/lane-143/shots/<set>/on-<screen>-<theme>.png`
 * (1440x900), `shots-manifest.json`, `sweep.md` and `baseline-diff.md`.
 * Closing round (T6): V2 is the only rendering, so this head shoots the ON
 * state only (set L5) and compares every ON shot to the L4 ON shot at
 * +-NOISE per channel, the topbar actions row masked (the retired ux-toggle
 * leaves that row 40px narrower). Sets L0-L4 shot OFF and ON on earlier heads;
 * their rows stay in `SETS` as the record and as baselines, but this head
 * refuses to re-shoot them (it no longer renders OFF). What a set shoots and
 * asserts is ONE table, `SETS`, keyed by the set's layer: its states, screens,
 * PNG count and the ON icon count of every screen. Every capture also has a
 * full-length twin under `<set>/full/`.
 */
const SET = process.env['UX_SHOTS_SET'] ?? 'L0';
const STAGE = /^L(\d+)/.exec(SET)?.[1];
/**
 * VERIFY round 2 (e85b2d8c) 1.01: the OFF-vs-main compare is not optional. Every set
 * except the baseline-generating L0 compares its OFF shots to a baseline set: the
 * named `UX_SHOTS_BASELINE`, else `L0`. L0 (stage 0) is exempt -- it IS the baseline.
 */
function resolveBaseline(stage: string | undefined, env: string | undefined, dflt = 'L0'): string | undefined {
    if (stage === '0') return undefined;
    return env && env.trim() ? env.trim() : dflt;
}
const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const SHOTS_ROOT = path.join(REPO_ROOT, '.agent', 'workdir', 'lane-143', 'shots');
const OUT = path.join(SHOTS_ROOT, SET);

/**
 * Pixels of two same-size PNGs whose largest channel delta exceeds `noise`.
 * Chromium's GPU raster is not byte-stable between runs: two captures of the
 * SAME build differ by +-1 on a few dozen anti-aliased edge pixels (measured:
 * 35 px on L0 vs L0m training/dark). A real rendering change moves whole
 * glyphs or fills by far more, so the compare tolerates +-NOISE per channel
 * and still demands 0 pixels beyond it. Wall-clock text is excluded too: the
 * mock's job timestamps are `Date.now()`-relative and the browser clock is
 * real, so "11:21 elapsed" / "started 10:27 PM" / "finish 00:32" differ on
 * every run of the SAME build (measured: 303 px beyond +-2 on L0 vs L0m
 * jobs/dark, all inside the run header and the ETA tile). The skipped boxes
 * are those of elements whose OWN text holds a clock time (h:mm), listed per
 * file in baseline-diff.md.
 */
const NOISE = 2;
type Rect = { x: number; y: number; w: number; h: number };
function pixelDiff(a: Buffer, b: Buffer, skip: Rect[] = []): { size: string; beyond: number; noise: number } {
    // The PNG codec Playwright itself ships; no new dependency.
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { PNG } = require('playwright-core/lib/utilsBundle');
    const pa = PNG.sync.read(a);
    const pb = PNG.sync.read(b);
    if (pa.width !== pb.width || pa.height !== pb.height) {
        return { size: `${pa.width}x${pa.height} vs ${pb.width}x${pb.height}`, beyond: -1, noise: 0 };
    }
    let beyond = 0;
    let noise = 0;
    for (let i = 0; i < pa.data.length; i += 4) {
        const px = (i / 4) % pa.width;
        const py = Math.floor(i / 4 / pa.width);
        if (skip.some((r) => px >= r.x && px < r.x + r.w && py >= r.y && py < r.y + r.h)) continue;
        let d = 0;
        for (let c = 0; c < 4; c++) d = Math.max(d, Math.abs(pa.data[i + c] - pb.data[i + c]));
        if (d > NOISE) beyond++;
        else if (d > 0) noise++;
    }
    return { size: `${pa.width}x${pa.height}`, beyond, noise };
}

const THEMES = ['dark', 'light'] as const;
type State = 'off' | 'on';
const BOTH: State[] = ['off', 'on'];
const FIVE = ['datasets', 'training', 'templates', 'server', 'jobs'];
/** Icons ON per screen once L2 lands (spec: 6 datasets, 4 server, 6 jobs, 4 projects, 6 training = 26). */
const KPI_ICONS: Record<string, number> = { datasets: 6, training: 6, templates: 0, server: 4, jobs: 6, projects: 4 };
const NO_ICONS: Record<string, number> = Object.fromEntries(Object.keys(KPI_ICONS).map((k) => [k, 0]));
/**
 * `states` defaults to OFF + ON; `compare` is the state compared to the baseline set
 * (`baseline`, default L0): OFF-vs-main for L1-L4, ON-vs-L4 for the closing round L5.
 */
type SetSpec = {
    screens: string[]; shots: number; iconsOn: Record<string, number>; extra?: string[];
    states?: State[]; compare?: State; baseline?: string;
};
const statesOf = (spec: SetSpec): State[] => spec.states ?? BOTH;
/** The ONE table: a round's evidence is whatever its row says, nothing derived elsewhere. */
const SETS: Record<string, SetSpec> = {
    L0: { screens: FIVE, shots: 20, iconsOn: NO_ICONS },
    L1: { screens: FIVE, shots: 20, iconsOn: NO_ICONS },
    L2: { screens: [...FIVE, 'projects'], shots: 24, iconsOn: KPI_ICONS },
    L3: { screens: FIVE, shots: 20, iconsOn: KPI_ICONS, extra: ['on-modal-dark.png', 'on-modal-light.png', 'on-workspace-dark.png', 'on-workspace-light.png'] },
};
/** L3b = L3 after the user's round-3 answer (ew-grid-padding=12px): same screens, same extras. */
SETS['L3b'] = { ...SETS['L3'] };
/** L3c = L3b after the verify-r1 border/ring remediation: same screens, same extras. */
SETS['L3c'] = { ...SETS['L3'] };
/** L3d = L3c after the verify-r2 component-CTA remediation: plus one hovered modal CTA per theme. */
SETS['L3d'] = { ...SETS['L3'], extra: [...(SETS['L3'].extra ?? []), 'on-modal-hover-dark.png', 'on-modal-hover-light.png'] };
/** L4 = L3d after the user's round-4 answer (live-peak-vram-card=match-estimate-card-layout): same screens, same extras. */
SETS['L4'] = { ...SETS['L3d'] };
/**
 * L5 = the closing round (the user's decision 2026-10-08, UAT-LANE-143.4 ok): V2 is the
 * only rendering and the toggle is gone. ON only, 6 screens (projects joins so all 26 icons
 * are shot; L4 has no projects shot, recorded as an exception by design); every ON shot and
 * every extra is compared to L4's at +-NOISE with the topbar actions row masked.
 */
SETS['L5'] = { ...SETS['L4'], screens: [...FIVE, 'projects'], states: ['on'], shots: 12, compare: 'on', baseline: 'L4' };

/**
 * Fails BEFORE any capture when the baseline set has no shots on disk (the compare
 * would otherwise be skipped or die late): every OFF image its SETS row promises
 * must exist. Throws a `LANE-143:` message naming the first missing file.
 */
function assertBaselineOnDisk(root: string, baseline: string, state: State = 'off'): void {
    const stage = /^L(\d+)/.exec(baseline)?.[1];
    const row = stage === undefined ? undefined : SETS[`L${stage}`];
    if (!row) throw new Error(`LANE-143: unknown baseline set ${baseline}`);
    for (const screen of row.screens) {
        for (const theme of THEMES) {
            const file = `${state}-${screen}-${theme}.png`;
            if (!fs.existsSync(path.join(root, baseline, file))) {
                throw new Error(`LANE-143: baseline set ${baseline} has no shots on disk (missing ${file}); shoot ${baseline} first or name UX_SHOTS_BASELINE`);
            }
        }
    }
}

/**
 * VERIFY round 1 (68cdc6ef) 1.02: a missing baseline image is a FAILURE unless the
 * baseline SET lacks that screen by design (its `SETS` row never shot it: `projects`
 * joins at L2, so L0 has no projects shots). Returns the exception text to record in
 * baseline-diff.md, `null` when the image must be compared, and throws `LANE-143:`
 * when a required image is missing -- the old fail-open treated ANY missing file as
 * "not in baseline" and compared nothing.
 */
function baselineException(root: string, baseline: string, screen: string, file: string): string | null {
    const stage = /^L(\d+)/.exec(baseline)?.[1];
    const row = stage === undefined ? undefined : SETS[`L${stage}`];
    if (!row) throw new Error(`LANE-143: unknown baseline set ${baseline}`);
    if (!row.screens.includes(screen)) return `${screen} not in ${baseline} by design (its SETS row shoots ${row.screens.join(', ')})`;
    if (!fs.existsSync(path.join(root, baseline, file))) {
        throw new Error(`LANE-143: baseline ${baseline} lacks required ${file} (${screen} is in its SETS row)`);
    }
    return null;
}

/**
 * VERIFY round 1 (68cdc6ef) 1.01: every `.kpi-icon` must sit clear of everything
 * else painted in its tile. Measured against what is PAINTED, not block boxes (a
 * full-width label div would "overlap" any corner): the glyph rects of every
 * non-blank text node and the boxes of leaf elements (no element children, or an
 * svg/img/canvas) in the same `.kpi`, the icon's own subtree excluded. Self-contained
 * so it runs in the page (`page.evaluate(iconOverlaps)`).
 */
function iconOverlaps(): string[] {
    const hits: string[] = [];
    const meet = (a: DOMRect, b: DOMRect) =>
        a.width > 0 && a.height > 0 && b.width > 0 && b.height > 0 &&
        a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
    const name = (el: Element) => el.tagName.toLowerCase() + (el.getAttribute('class') ? '.' + el.getAttribute('class')!.trim().split(/\s+/).join('.') : '');
    for (const icon of Array.from(document.querySelectorAll('.kpi-icon'))) {
        const tile = icon.closest('.kpi');
        if (!tile) continue;
        const r = icon.getBoundingClientRect();
        const label = tile.querySelector('.kpi-label')?.textContent?.trim() ?? '?';
        for (const el of Array.from(tile.querySelectorAll('*'))) {
            if (el === icon || icon.contains(el) || el.contains(icon)) continue;
            if (getComputedStyle(el).visibility === 'hidden') continue;
            const tag = el.tagName.toLowerCase();
            if (tag !== 'svg' && el.closest('svg')) continue; // an svg's parts are covered by the svg
            const cs = getComputedStyle(el);
            const hasText = Array.from(el.childNodes).some((n) => n.nodeType === Node.TEXT_NODE && (n.textContent ?? '').trim());
            const paintsBox =
                !/rgba\(0, 0, 0, 0\)|transparent/.test(cs.backgroundColor) ||
                (parseFloat(cs.borderTopWidth) > 0 && !/rgba\(0, 0, 0, 0\)|transparent/.test(cs.borderTopColor)) ||
                cs.backgroundImage !== 'none';
            // a box counts when it paints itself (fill, border, image) or is a text-less leaf (bar, svg, img);
            // a text-only block (a full-width label div) counts only by its glyphs, below
            const leaf = (el.children.length === 0 && !hasText) || ['svg', 'img', 'canvas'].includes(tag);
            if ((leaf || paintsBox) && meet(r, el.getBoundingClientRect())) {
                hits.push(`${label}: .kpi-icon overlaps ${name(el)}`);
                continue;
            }
            for (const n of Array.from(el.childNodes)) {
                if (n.nodeType !== Node.TEXT_NODE || !(n.textContent ?? '').trim()) continue;
                const range = document.createRange();
                range.selectNodeContents(n);
                if (Array.from(range.getClientRects()).some((t) => meet(r, t))) {
                    hits.push(`${label}: .kpi-icon overlaps the text "${(n.textContent ?? '').trim().slice(0, 24)}" of ${name(el)}`);
                }
            }
        }
    }
    return hits;
}

/**
 * User polish 2026-10-07: no `.chip`/`.tag` inside a `.ds-card` may extend past its card's
 * box. Measured on main (switch OFF) the "N suppressed" chip already overflowed by 50 px, so
 * the guard is asserted for the ON state only (the fix lives in ux-v2.css). Runs in the page.
 */
function cardChipOverflows(): string[] {
    const hits: string[] = [];
    for (const card of Array.from(document.querySelectorAll('.ds-card'))) {
        const b = card.getBoundingClientRect();
        for (const c of Array.from(card.querySelectorAll('.chip, .tag'))) {
            const r = c.getBoundingClientRect();
            if (r.width > 0 && (r.right > b.right + 0.5 || r.left < b.left - 0.5)) {
                hits.push(`${(c.textContent ?? '').trim().slice(0, 24)} [${Math.round(r.left)}..${Math.round(r.right)}] past card [${Math.round(b.left)}..${Math.round(b.right)}]`);
            }
        }
    }
    return hits;
}

/**
 * The user's UAT round 4 answer (2026-10-08, live-peak-vram-card=match-estimate-card-layout):
 * every KPI tile in the Training right rail (the ESTIMATE wall's five and the LIVE ESTIMATE
 * PEAK VRAM tile) sits in its card with the SAME horizontal inset and the same inner padding
 * as the wall's first tile. Returns one line per tile that differs. Runs in the page.
 */
function estimateTileMismatches(): string[] {
    const tiles = Array.from(document.querySelectorAll<HTMLElement>('.ts-estimate-rail .kpi.compact'));
    const ref = document.querySelector<HTMLElement>('.ew-grid > .kpi');
    if (!ref) return tiles.length ? ['no .ew-grid > .kpi reference tile'] : [];
    const geom = (k: HTMLElement) => {
        const r = k.getBoundingClientRect();
        const c = k.parentElement!.closest('.card')!.getBoundingClientRect();
        const cs = getComputedStyle(k);
        return { left: Math.round(r.left - c.left), right: Math.round(c.right - r.right), pad: cs.padding };
    };
    const want = geom(ref);
    const hits: string[] = [];
    for (const k of tiles) {
        const g = geom(k);
        if (g.left !== want.left || g.right !== want.right || g.pad !== want.pad) {
            const label = (k.querySelector('.kpi-label')?.textContent ?? '?').trim();
            hits.push(`${label}: inset ${g.left}/${g.right} pad ${g.pad} vs wall ${want.left}/${want.right} pad ${want.pad}`);
        }
    }
    return hits;
}

/** Runs without UX_SHOTS: the rail-inset check fires on an edge-to-edge tile and stays quiet on an inset one. */
test('LANE-143 the estimate rail inset check fires on an edge-to-edge live tile (negative control)', async ({ page }) => {
    const rail = (liveMargin: number) => `
        <aside class="ts-estimate-rail" style="width:250px">
          <div class="card"><div class="ew-grid" style="padding:12px"><div class="kpi compact" style="padding:10px"><div class="kpi-label">WALL TIME</div></div></div></div>
          <div class="card"><div class="rail"><div class="kpi compact" style="padding:10px;margin:0 ${liveMargin}px"><div class="kpi-label">PEAK VRAM</div></div></div></div>
        </aside>`;
    await page.setContent(rail(0));
    const hit = await page.evaluate(estimateTileMismatches);
    expect(hit.length, `LANE-143: an edge-to-edge live tile must be reported, got ${JSON.stringify(hit)}`).toBeGreaterThan(0);
    await page.setContent(rail(12));
    expect(await page.evaluate(estimateTileMismatches), 'LANE-143: a live tile inset like the wall must not be reported').toEqual([]);
});

/** Runs without UX_SHOTS: the overflow check fires on a chip past its card and stays quiet inside it. */
test('LANE-143 the card chip overflow check fires on an overflowing chip (negative control)', async ({ page }) => {
    const card = (chipLeft: number) => `
        <div class="ds-card" style="position:relative;width:200px;height:80px"><span class="chip" style="position:absolute;top:4px;left:${chipLeft}px;width:60px">x suppressed</span></div>`;
    await page.setContent(card(170));
    expect((await page.evaluate(cardChipOverflows)).length, 'LANE-143: a chip past its card must be reported').toBeGreaterThan(0);
    await page.setContent(card(10));
    expect(await page.evaluate(cardChipOverflows), 'LANE-143: a chip inside its card must not be reported').toEqual([]);
});

/** Runs without UX_SHOTS: the overlap check fires on an overlap and stays quiet without one. */
test('LANE-143 the icon overlap check fires on an overlap (negative control)', async ({ page }) => {
    const tile = (badgeRight: number) => `
        <div class="kpi" style="position:relative;width:200px;height:90px;padding:14px 16px;box-sizing:border-box">
          <span class="kpi-icon" style="position:absolute;top:10px;right:10px;width:24px;height:24px"><svg width="14" height="14"></svg></span>
          <div class="kpi-label">HPS</div><div class="kpi-value">5.1</div>
          <span class="badge" style="position:absolute;top:10px;right:${badgeRight}px;width:15px;height:15px;border:1px solid #888;border-radius:50%">i</span>
        </div>`;
    await page.setContent(tile(12));
    const hit = await page.evaluate(iconOverlaps);
    expect(hit.length, `LANE-143: an icon on a corner badge must be reported, got ${JSON.stringify(hit)}`).toBeGreaterThan(0);
    await page.setContent(tile(140));
    expect(await page.evaluate(iconOverlaps), 'LANE-143: a clear icon must not be reported').toEqual([]);
});

/** Runs without UX_SHOTS: a missing REQUIRED baseline image fails; a screen the set lacks by design is an exception. */
test('LANE-143 a missing required baseline image fails (negative control)', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'lane143-base-'));
    try {
        fs.mkdirSync(path.join(root, 'L0'));
        expect(() => baselineException(root, 'L0', 'datasets', 'off-datasets-dark.png'), 'LANE-143: a missing required baseline image must fail the run').toThrow(
            /^LANE-143: baseline L0 lacks required off-datasets-dark\.png/,
        );
        expect(baselineException(root, 'L0', 'projects', 'off-projects-dark.png')).toMatch(/^projects not in L0 by design/);
        fs.writeFileSync(path.join(root, 'L0', 'off-datasets-dark.png'), '');
        expect(baselineException(root, 'L0', 'datasets', 'off-datasets-dark.png')).toBeNull();
        expect(() => baselineException(root, 'X9', 'datasets', 'off-datasets-dark.png')).toThrow(/^LANE-143: unknown baseline set/);
    } finally {
        fs.rmSync(root, { recursive: true, force: true });
    }
});

/** Runs without UX_SHOTS: an OMITTED baseline on a non-L0 set compares against L0; L0 is exempt; absent shots fail up front. */
test('LANE-143 an omitted baseline defaults to L0 and absent baseline shots fail before capture (negative control)', () => {
    expect(resolveBaseline('3', undefined), 'LANE-143: an omitted baseline on L3 must compare against L0').toBe('L0');
    expect(resolveBaseline('1', ''), 'LANE-143: an empty baseline on L1 must compare against L0').toBe('L0');
    expect(resolveBaseline('2', 'L1')).toBe('L1');
    expect(resolveBaseline('0', undefined), 'LANE-143: L0 generates the baseline and is exempt').toBeUndefined();
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'lane143-pre-'));
    try {
        expect(() => assertBaselineOnDisk(root, 'L0'), 'LANE-143: a baseline set without shots must fail before capture').toThrow(
            /^LANE-143: baseline set L0 has no shots on disk/,
        );
        fs.mkdirSync(path.join(root, 'L0'));
        for (const screen of FIVE) for (const theme of THEMES) fs.writeFileSync(path.join(root, 'L0', `off-${screen}-${theme}.png`), '');
        expect(() => assertBaselineOnDisk(root, 'L0')).not.toThrow();
    } finally {
        fs.rmSync(root, { recursive: true, force: true });
    }
});

/** Runs without UX_SHOTS: the table is consistent with itself and with the spec. */
test('LANE-143 the shot table is self-consistent', () => {
    for (const [name, spec] of Object.entries(SETS)) {
        expect(spec.shots, `LANE-143: ${name} shots = screens x themes x states`).toBe(spec.screens.length * THEMES.length * statesOf(spec).length);
        for (const screen of spec.screens) {
            expect(typeof spec.iconsOn[screen], `LANE-143: ${name} has no ON icon count for ${screen}`).toBe('number');
        }
    }
    expect(SETS['L1'].shots).toBe(20);
    expect(SETS['L2'].shots).toBe(24);
    expect(SETS['L3'].shots).toBe(20);
    expect(SETS['L3'].extra, 'LANE-143: L3 adds the on-modal and on-workspace shots').toEqual(['on-modal-dark.png', 'on-modal-light.png', 'on-workspace-dark.png', 'on-workspace-light.png']);
    const sum = (n: string) => SETS[n].screens.reduce((a, s) => a + SETS[n].iconsOn[s], 0);
    expect(sum('L0') + sum('L1'), 'LANE-143: no icon may render before L2').toBe(0);
    expect(sum('L2'), 'LANE-143: L2 renders 26 icons ON').toBe(26);
    expect(SETS['L3d'].extra, 'LANE-143: L3d adds one hovered modal CTA per theme').toEqual([...(SETS['L3'].extra ?? []), 'on-modal-hover-dark.png', 'on-modal-hover-light.png']);
    expect(STAGE === undefined || SETS[`L${STAGE}`] !== undefined, `LANE-143: unknown shot set ${SET}`).toBe(true);
    expect(statesOf(SETS['L5']), 'LANE-143: the closing round shoots ON only').toEqual(['on']);
    expect(SETS['L5'].shots, 'LANE-143: L5 = 6 screens (incl. projects) x 2 themes, ON only').toBe(12);
    expect(sum('L5'), 'LANE-143: L5 renders 26 icons').toBe(26);
    expect(resolveBaseline('5', undefined, SETS['L5'].baseline), 'LANE-143: an omitted baseline on L5 compares against L4').toBe('L4');
});

/** A set with its own row (L3d) uses it; a re-shoot without one (L3b, L3c) takes its layer's. */
const SPEC = SETS[SET] ?? SETS[`L${STAGE}`] ?? SETS['L0'];
const SCREENS = SPEC.screens;
const BASELINE = resolveBaseline(STAGE, process.env['UX_SHOTS_BASELINE'], SPEC.baseline);
const COMPARE: State = SPEC.compare ?? 'off';

test.describe.configure({ mode: 'serial' });
test.use({ viewport: { width: 1440, height: 900 } });

/**
 * The full-length twin of a capture (UAT round 1 item 3: the Server log-level
 * pills sat below the 1440x900 fold). The shell scrolls inside its own
 * containers, so `fullPage` alone would stop at the viewport: the viewport is
 * grown by the largest scroll overflow on the page, captured, and restored.
 */
async function fullCapture(page: import('@playwright/test').Page, file: string): Promise<void> {
    const extra = await page.evaluate(() => {
        let most = Math.max(0, document.documentElement.scrollHeight - window.innerHeight);
        for (const el of Array.from(document.querySelectorAll<HTMLElement>('body *'))) {
            const oy = getComputedStyle(el).overflowY;
            if ((oy === 'auto' || oy === 'scroll') && el.clientHeight > 0) most = Math.max(most, el.scrollHeight - el.clientHeight);
        }
        return Math.ceil(most);
    });
    const size = page.viewportSize()!;
    if (extra > 0) {
        await page.setViewportSize({ width: size.width, height: Math.min(size.height + extra, 8000) });
        await page.waitForTimeout(400);
    }
    await page.screenshot({ path: file, fullPage: true, animations: 'disabled' });
    if (extra > 0) {
        await page.setViewportSize(size);
        await page.waitForTimeout(200);
    }
}

async function settle(page: import('@playwright/test').Page, screen: string): Promise<void> {
    await page.waitForLoadState('networkidle');
    if (screen === 'jobs') {
        await expect(page.locator('.kpi-row app-kpi-tile'), 'LANE-143: expected 6 visible KPI tiles on jobs').toHaveCount(6);
    }
    await page.waitForTimeout(1500); // KPI tween + route paint
}

/**
 * VERIFY 1.01: the painted contrast of every visible status dot (`.sdot`), as the
 * browser rendered it -- the file-based guard measured the recipe while an
 * encapsulated component rule out-ranked it. The fill and the backdrop (the
 * ancestors' backgrounds composited root-down) are resolved through a 1x1 canvas
 * so `oklch()`/`color-mix()` computed values become sRGB the way they are painted.
 */
async function dotContrasts(page: import('@playwright/test').Page): Promise<{ cls: string; fill: number[]; back: number[]; ratio: number }[]> {
    return page.evaluate(() => {
        const ctx = document.createElement('canvas').getContext('2d', { willReadFrequently: true })!;
        const px = (): number[] => Array.from(ctx.getImageData(0, 0, 1, 1).data.slice(0, 3));
        const lum = ([r, g, b]: number[]): number => {
            const f = (c: number) => ((c /= 255) <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
            return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
        };
        return Array.from(document.querySelectorAll<HTMLElement>('.sdot'))
            .filter((el) => el.getClientRects().length > 0)
            .map((el) => {
                const chain: string[] = [];
                for (let a = el.parentElement; a; a = a.parentElement) chain.unshift(getComputedStyle(a).backgroundColor);
                ctx.clearRect(0, 0, 1, 1);
                ctx.fillStyle = '#ffffff';
                ctx.fillRect(0, 0, 1, 1);
                for (const c of chain) {
                    ctx.fillStyle = c;
                    ctx.fillRect(0, 0, 1, 1);
                }
                const back = px();
                ctx.fillStyle = getComputedStyle(el).backgroundColor;
                ctx.fillRect(0, 0, 1, 1);
                const fill = px();
                const [hi, lo] = [lum(fill), lum(back)].sort((x, y) => y - x);
                return { cls: el.className, fill, back, ratio: Math.round(((hi + 0.05) / (lo + 0.05)) * 100) / 100 };
            });
    });
}

/**
 * Verify-r2 1.01: the text-on-fill contrast of a button at rest, hovered and
 * pressed, as painted (1x1 canvas over the ancestors' backgrounds, root-down).
 * Hover = the real pointer over it; pressed = `:hover:active` forced through the
 * DevTools protocol (a held mouse released elsewhere clicks the common ancestor,
 * the modal backdrop, and closes the modal).
 */
async function statesRead(page: import('@playwright/test').Page, selector: string, label: string): Promise<string[]> {
    const btn = page.locator(selector).first();
    const read = () => btn.evaluate((el) => {
        const cv = document.createElement('canvas');
        cv.width = cv.height = 1;
        const cx = cv.getContext('2d', { willReadFrequently: true })!;
        const chain: Element[] = [];
        for (let e: Element | null = el; e; e = e.parentElement) chain.unshift(e);
        cx.fillStyle = '#fff';
        cx.fillRect(0, 0, 1, 1);
        for (const e of chain) {
            cx.fillStyle = getComputedStyle(e).backgroundColor;
            cx.fillRect(0, 0, 1, 1);
        }
        const bg = Array.from(cx.getImageData(0, 0, 1, 1).data).slice(0, 3);
        cx.fillStyle = getComputedStyle(el).color;
        cx.fillRect(0, 0, 1, 1);
        const fg = Array.from(cx.getImageData(0, 0, 1, 1).data).slice(0, 3);
        const lum = (c: number[]) => {
            const [r, g, b] = c.map((v) => (v / 255 <= 0.04045 ? v / 255 / 12.92 : ((v / 255 + 0.055) / 1.055) ** 2.4));
            return 0.2126 * r + 0.7152 * g + 0.0722 * b;
        };
        const a = lum(fg), b = lum(bg);
        return { ratio: (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05), fg: fg.join(','), bg: bg.join(',') };
    });
    const out: string[] = [];
    const row = (state: string, m: { ratio: number; fg: string; bg: string }) =>
        out.push(`| ${label} | ${state} | ${m.ratio.toFixed(2)} | ${m.fg} on ${m.bg} | ${m.ratio >= 4.5 ? 'ok' : 'FAIL'} |`);
    await page.mouse.move(1, 1);
    await page.waitForTimeout(250);
    row('rest', await read());
    await btn.hover();
    await page.waitForTimeout(250);
    row('hover', await read());
    const cdp = await page.context().newCDPSession(page);
    try {
        await cdp.send('DOM.enable');
        await cdp.send('CSS.enable');
        const { root } = await cdp.send('DOM.getDocument', { depth: 0 });
        const { nodeId } = await cdp.send('DOM.querySelector', { nodeId: root.nodeId, selector });
        expect(nodeId, `LANE-143: ${selector} not found for the pressed state`).toBeGreaterThan(0);
        await cdp.send('CSS.forcePseudoState', { nodeId, forcedPseudoClasses: ['hover', 'active'] });
        await page.waitForTimeout(150);
        row('pressed', await read());
        await cdp.send('CSS.forcePseudoState', { nodeId, forcedPseudoClasses: [] });
    } finally {
        await cdp.detach();
    }
    await page.mouse.move(1, 1);
    return out;
}

/**
 * Opens `route` in `theme` as a RETURNING V1 user: `mrln.ux` is stored as `v1`, which this
 * head must ignore and leave in place (closing round) -- the page renders V2 regardless.
 */
async function openAs(page: import('@playwright/test').Page, theme: string, route: string): Promise<void> {
    await page.goto('/');
    await page.evaluate((t) => {
        localStorage.setItem('mrln.theme', t);
        localStorage.setItem('mrln.ux', 'v1');
    }, theme);
    await page.goto(route);
    const state = await page.evaluate(() => ({ ux: document.documentElement.getAttribute('data-ux'), stored: localStorage.getItem('mrln.ux') }));
    expect(state.ux, 'LANE-143: data-ux is "v2" regardless of mrln.ux').toBe('v2');
    expect(state.stored, 'LANE-143: a stored mrln.ux is left in place').toBe('v1');
}

/**
 * The closing round's compare: this ON capture vs the baseline set's ON capture at +-NOISE.
 * Masked: clock text (as the OFF-vs-L0 compare did) and the topbar actions row grown 40 px to
 * the left -- the ONE layout change of T6 (the ux-toggle removed from a right-anchored row).
 */
async function compareToBaseline(page: import('@playwright/test').Page, file: string, screen: string): Promise<void> {
    if (!BASELINE) return;
    const exception = baselineException(SHOTS_ROOT, BASELINE, screen, file);
    if (exception !== null) {
        baselineRows.push(`| ${file} | - | exception: ${exception} | - | - |`);
        return;
    }
    const masks: Rect[] = await page.evaluate(() => {
        const out = Array.from(document.body.querySelectorAll('*'))
            .filter((el) =>
                Array.from(el.childNodes).some(
                    (n) => n.nodeType === Node.TEXT_NODE && /\b\d{1,2}:\d{2}\b/.test(n.textContent ?? ''),
                ),
            )
            .map((el) => {
                const r = el.getBoundingClientRect();
                return { x: Math.floor(r.x) - 1, y: Math.floor(r.y) - 1, w: Math.ceil(r.width) + 3, h: Math.ceil(r.height) + 3 };
            })
            .filter((r) => r.w > 3 && r.h > 3);
        const bar = document.querySelector('.topbar-actions')?.getBoundingClientRect();
        // 40 px for the retired toggle + 8 px each side for the buttons' shadow halo, which
        // paints outside their boxes (measured: 4 px past the row on on-modal-light).
        if (bar) out.push({ x: Math.floor(bar.x) - 48, y: Math.floor(bar.y) - 8, w: Math.ceil(bar.width) + 56, h: Math.ceil(bar.height) + 16 });
        return out;
    });
    const d = pixelDiff(fs.readFileSync(path.join(OUT, file)), fs.readFileSync(path.join(SHOTS_ROOT, BASELINE, file)), masks);
    const masked = masks.map((r) => `${r.w}x${r.h}@${r.x},${r.y}`).join(' ') || '-';
    baselineRows.push(`| ${file} | ${d.size} | ${d.beyond} | ${d.noise} | ${masked} |`);
    sweepRows.push(`| ${file} | ${d.size} | ${d.beyond} | ${d.noise} | ${masks.length} boxes |`);
    expect(d.beyond, `LANE-143: ${file} differs from baseline ${BASELINE} in ${d.beyond} px beyond +-${NOISE} (${d.size})`).toBe(0);
}

/**
 * Runs WITHOUT UX_SHOTS (an ordinary e2e test): the pre-paint guard in index.html sets
 * `data-ux="v2"` before the body exists and before the parse ends. The bundle is a deferred
 * module script, so a value present at those two moments came from the inline guard, never
 * from UxStore -- and nothing can paint before the body exists.
 */
test('LANE-143 data-ux is set before first paint by the inline guard', async ({ page }) => {
    await page.addInitScript(() => {
        const w = window as unknown as Record<string, unknown>;
        const obs = new MutationObserver(() => {
            if (document.body && !('__uxAtBody' in w)) {
                w['__uxAtBody'] = document.documentElement?.getAttribute('data-ux') ?? null;
                obs.disconnect();
            }
        });
        obs.observe(document, { childList: true, subtree: true });
        document.addEventListener('readystatechange', () => {
            if (document.readyState === 'interactive') w['__uxAtInteractive'] = document.documentElement.getAttribute('data-ux');
        });
    });
    await page.goto('/');
    await page.evaluate(() => localStorage.setItem('mrln.ux', 'v1'));
    await page.goto('/datasets');
    await page.waitForLoadState('networkidle');
    const r = await page.evaluate(() => {
        const w = window as unknown as Record<string, unknown>;
        return {
            atBody: w['__uxAtBody'] ?? 'unrecorded',
            atInteractive: w['__uxAtInteractive'] ?? 'unrecorded',
            bundleIsModule: Array.from(document.querySelectorAll<HTMLScriptElement>('script[src]')).some(
                (el) => /main/.test(el.getAttribute('src') ?? '') && el.type === 'module',
            ),
        };
    });
    expect(r.bundleIsModule, 'LANE-143: the app bundle must be a deferred module script, or the parse-time check proves nothing').toBe(true);
    expect(r.atBody, 'LANE-143: data-ux must be set before first paint').toBe('v2');
    expect(r.atInteractive, 'LANE-143: data-ux must be set before first paint').toBe('v2');
});

const dotRows: string[] = [];
const sweepRows: string[] = [];
const baselineRows: string[] = [];
const manifest: { file: string; px: string; bytes: number; sha256: string }[] = [];

test.describe('LANE-143 UX shots', () => {
    test.skip(!process.env['UX_SHOTS'], 'UX_SHOTS=1 required: this spec produces UAT evidence');

    test.beforeAll(() => {
        if (statesOf(SPEC).includes('off')) {
            throw new Error(`LANE-143: set ${SET} shoots the OFF state, which this head no longer renders (V2 is the only rendering); shoot L5`);
        }
        if (BASELINE) assertBaselineOnDisk(SHOTS_ROOT, BASELINE, COMPARE);
        fs.mkdirSync(path.join(OUT, 'full'), { recursive: true });
        fs.writeFileSync(path.join(OUT, 'button-contrast.md'), '| button | state | ratio | text on fill (sRGB) | >= 4.5 |\n|---|---|---|---|---|\n');
    });

    test('LANE-143 the seeded running job yields step 1200 / total 5000 / 1024x1024', () => {
        const m = latestMetrics(runningJobStepLogs);
        expect(m, 'LANE-143: latestMetrics(job.logs) returned null, expected step 1200').not.toBeNull();
        expect(m!.step).toBe(1200);
        expect(m!.total_steps).toBe(5000);
        expect(m!.resolution).toBe('1024x1024');
        expect(jobs.find((j) => j.id === 'job-e2e-running')!.logs).toEqual(runningJobStepLogs);
    });

    for (const screen of SCREENS) {
        for (const theme of THEMES) {
            test(`LANE-143 shots ${screen} ${theme}`, async ({ page }) => {
                await openAs(page, theme, `/${screen}`);
                await settle(page, screen);
                const want = SPEC.iconsOn[screen];
                const squares = await page.locator('.kpi-icon').count();
                const icons = await page.locator('.kpi-icon svg').count();
                const stray = await page.locator('.kpi app-ico').count();
                expect(squares, `LANE-143: expected ${want} visible KPI tiles/icons on ${screen} (set ${SET})`).toBe(want);
                expect(icons, `LANE-143: expected ${want} KPI icon svgs on ${screen} (set ${SET})`).toBe(want);
                expect(stray, `LANE-143: an <app-ico> in a KPI tile outside .kpi-icon on ${screen}`).toBe(want);
                if (want > 0) {
                    // every square sits inside its tile's padding box (no overflow, no collision with the label row height)
                    const outside = await page.evaluate(() =>
                        Array.from(document.querySelectorAll('.kpi-icon')).filter((el) => {
                            const r = el.getBoundingClientRect();
                            const k = el.closest('.kpi')!.getBoundingClientRect();
                            return r.right > k.right || r.top < k.top || r.bottom > k.bottom || r.width === 0;
                        }).length,
                    );
                    expect(outside, `LANE-143: ${outside} icon square(s) outside their tile on ${screen}`).toBe(0);
                    const overlaps = await page.evaluate(iconOverlaps);
                    expect(overlaps, `LANE-143: a .kpi-icon overlaps tile content on ${screen}/${theme}: ${overlaps.join('; ')}`).toEqual([]);
                }
                if (screen === 'training' && SPEC.iconsOn['training'] > 0) {
                    const mis = await page.evaluate(estimateTileMismatches);
                    expect(mis, `LANE-143: a Training rail KPI tile does not match the estimate wall's card layout on ${theme}: ${mis.join('; ')}`).toEqual([]);
                }
                if (screen === 'datasets') {
                    const over = await page.evaluate(cardChipOverflows);
                    expect(over, `LANE-143: a chip/tag extends past its .ds-card on ${screen}/${theme}: ${over.join('; ')}`).toEqual([]);
                }
                for (const d of await dotContrasts(page)) {
                    dotRows.push(`| ${screen} | ${theme} | on | ${d.cls} | rgb(${d.fill}) | rgb(${d.back}) | ${d.ratio} |`);
                    expect(
                        d.ratio,
                        `LANE-143: painted .sdot (${d.cls}) on ${screen}/${theme} measures ${d.ratio}:1 < 3:1 (rgb(${d.fill}) on rgb(${d.back}))`,
                    ).toBeGreaterThanOrEqual(3);
                }
                const file = `on-${screen}-${theme}.png`;
                await page.screenshot({ path: path.join(OUT, file), animations: 'disabled' });
                await compareToBaseline(page, file, screen);
                await fullCapture(page, path.join(OUT, 'full', file));
            });
        }
    }

    for (const theme of THEMES) {
        if (!SPEC.extra) break;
        test(`LANE-143 modal ${theme}`, async ({ page }) => {
            await openAs(page, theme, '/datasets');
            await settle(page, 'datasets');
            await page.getByRole('button', { name: /new dataset/i }).first().click();
            const dialog = page.locator('.modal').first();
            await expect(dialog, 'LANE-143: the New dataset dialog must open for the modal shot').toBeVisible();
            await page.waitForTimeout(500);
            const file = `on-modal-${theme}.png`;
            await page.screenshot({ path: path.join(OUT, file), animations: 'disabled' });
            await compareToBaseline(page, file, 'datasets');
            await fullCapture(page, path.join(OUT, 'full', file));
        });
    }

    for (const theme of THEMES) {
        if (!SPEC.extra) break;
        test(`LANE-143 workspace toolbar ${theme}`, async ({ page }) => {
            await openAs(page, theme, '/datasets');
            await settle(page, 'datasets');
            await page.getByTestId('dataset-card-alpha').click();
            const mask = page.getByTestId('ws-mass-mask-btn');
            const edit = page.getByTestId('ws-mass-edit-btn');
            await expect(mask, 'LANE-143: the workspace Mass mask button must be visible').toBeVisible();
            await page.waitForTimeout(500);
            // the three Mass buttons are coloured on purpose: caption (brand fill), mask (green), edit (violet)
            const paint = (loc: typeof mask) => loc.evaluate((el) => {
                const cs = getComputedStyle(el);
                return `${cs.color}|${cs.backgroundColor}`;
            });
            const caption = await paint(page.getByTestId('ws-mass-caption-btn'));
            const plain = await page.locator('.ws-secondary .btn:not(.primary):not(.ws-mass-mask):not(.ws-mass-edit)').first().evaluate((el) => `${getComputedStyle(el).color}|${getComputedStyle(el).backgroundColor}`).catch(() => '');
            const m = await paint(mask);
            const e = await paint(edit);
            expect(new Set([caption, m, e]).size, `LANE-143: Mass caption/mask/edit must be three distinct paints under v2: ${caption} / ${m} / ${e}`).toBe(3);
            expect(m, 'LANE-143: Mass mask must not paint the plain secondary button').not.toBe(plain);
            expect(e, 'LANE-143: Mass edit must not paint the plain secondary button').not.toBe(plain);
            const file = `on-workspace-${theme}.png`;
            await page.screenshot({ path: path.join(OUT, file), animations: 'disabled' });
            await compareToBaseline(page, file, 'datasets');
            await fullCapture(page, path.join(OUT, 'full', file));
        });
    }

    for (const theme of THEMES) {
        if (!SPEC.extra) break;
        test(`LANE-143 component buttons read hovered and pressed ${theme}`, async ({ page }) => {
            await openAs(page, theme, '/');
            const rows: string[] = [];
            const openWorkspace = async () => {
                await page.goto('/datasets');
                await settle(page, 'datasets');
                await page.getByTestId('dataset-card-alpha').click();
                await expect(page.getByTestId('ws-mass-mask-btn'), 'LANE-143: the workspace must open').toBeVisible();
                await page.waitForTimeout(500);
            };
            await openWorkspace();
            for (const id of ['ws-mass-caption-btn', 'ws-mass-mask-btn', 'ws-mass-edit-btn']) {
                rows.push(...await statesRead(page, `[data-testid="${id}"]`, `${id} (${theme})`));
            }
            // the two modal CTAs of verify-r2 1.01, enabled the way a user enables them
            for (const [mass, cta, enable] of [
                ['mask', '.modal .btn.cta.success', null],
                ['caption', '.modal .btn.cta', '.modal .mc-choice:has-text("Destructive")'],
            ] as const) {
                await openWorkspace();
                await page.getByTestId(`ws-mass-${mass}-btn`).click();
                await expect(page.locator('.modal').first(), `LANE-143: the mass-${mass} modal must open`).toBeVisible();
                if (enable) await page.locator(enable).first().click();
                const btn = page.locator(cta).first();
                await expect(btn, `LANE-143: the mass-${mass} CTA must be enabled to be hovered`).toBeEnabled({ timeout: 10_000 });
                await page.waitForTimeout(400);
                rows.push(...await statesRead(page, cta, `mass-${mass} modal CTA (${theme})`));
                const file = `on-modal-hover-${theme}.png`;
                if (mass === 'mask' && SPEC.extra?.includes(file)) {
                    await btn.hover();
                    await page.waitForTimeout(300);
                    await page.screenshot({ path: path.join(OUT, file), animations: 'disabled' });
                    await compareToBaseline(page, file, 'datasets');
                    await fullCapture(page, path.join(OUT, 'full', file));
                }
            }
            fs.appendFileSync(path.join(OUT, 'button-contrast.md'), rows.map((r) => `${r}\n`).join(''));
            const bad = rows.filter((r) => r.includes('FAIL'));
            expect(bad, `LANE-143: a component button's text no longer reads on its v2 fill:\n${bad.join('\n')}`).toEqual([]);
        });
    }

    test('LANE-143 manifest', async () => {
        const files = fs.readdirSync(OUT).filter((f) => f.endsWith('.png')).sort();
        const full = fs.readdirSync(path.join(OUT, 'full')).filter((f) => f.endsWith('.png')).sort();
        const expected = SPEC.shots + (SPEC.extra?.length ?? 0); // from the SETS row, never re-derived here
        expect(files.length, `LANE-143: expected ${expected} shots, got ${files.length}`).toBe(expected);
        expect(full, `LANE-143: every shot has its full-length twin under full/`).toEqual(files);
        for (const f of [...files, ...full.map((x) => `full/${x}`)]) {
            const buf = fs.readFileSync(path.join(OUT, f));
            manifest.push({
                file: f,
                px: `${buf.readUInt32BE(16)}x${buf.readUInt32BE(20)}`,
                bytes: buf.length,
                sha256: createHash('sha256').update(buf).digest('hex').slice(0, 16),
            });
        }
        fs.writeFileSync(path.join(OUT, 'shots-manifest.json'), JSON.stringify(manifest, null, 2));
        fs.writeFileSync(
            path.join(OUT, 'sweep.md'),
            [`ON vs ${BASELINE ?? '-'} ON (tolerance +-${NOISE} per channel; replaces the OFF-vs-ON height sweep: this head renders no OFF state)`, '',
                '| file | px | beyond | noise | masked |', '|---|---|---|---|---|', ...sweepRows,
                '',
                'Masked in every compare: the `.topbar-actions` row widened 40 px leftward plus 8 px on every side for the button shadow halo (the retired ux-toggle: 32 px button + 8 px gap; the row is right-anchored, so the elements before the theme toggle move 40 px right), and the boxes of elements whose own text holds a clock time (h:mm).',
            ].join('\n') + '\n',
        );
        fs.writeFileSync(
            path.join(OUT, 'dot-contrast.md'),
            ['| screen | theme | state | class | fill | backdrop | ratio |', '|---|---|---|---|---|---|---|', ...dotRows].join('\n') + '\n',
        );
        fs.writeFileSync(
            path.join(OUT, 'baseline-diff.md'),
            [`${COMPARE.toUpperCase()} vs ${BASELINE ?? '-'} (tolerance +-${NOISE} per channel)`, '', '| file | px | beyond | noise | masked |', '|---|---|---|---|---|', ...baselineRows].join('\n') + '\n',
        );
        expect(manifest.length).toBe(2 * expected);
    });
});
