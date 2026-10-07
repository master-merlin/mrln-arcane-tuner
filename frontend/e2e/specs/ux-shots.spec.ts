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
 *   UX_SHOTS=1 UX_SHOTS_SET=L1 [UX_SHOTS_BASELINE=L0] \
 *     npm --prefix frontend run e2e -- ux-shots --workers=1
 *
 * Writes `.agent/workdir/lane-143/shots/<set>/<off|on>-<screen>-<theme>.png`
 * (1440x900), `shots-manifest.json`, `sweep.md` and `toggle-displacement.txt`.
 * Every OFF capture and every box sweep hides `[data-testid="ux-toggle"]` (the
 * one element this branch adds to the OFF rendering) so the topbar sits where
 * `main` puts it. What a set shoots and asserts is ONE table, `SETS`, keyed by
 * the set's layer (`L2b` = a remediation re-shoot of L2): its screens, its PNG
 * count and the ON icon count of every screen (OFF is always 0). Every capture
 * also has a full-length twin under `<set>/full/` (the viewport grown to the
 * page's scroll height), so content below the 1440x900 fold is in the pack.
 */
const SET = process.env['UX_SHOTS_SET'] ?? 'L0';
const STAGE = /^L(\d+)/.exec(SET)?.[1];
/**
 * VERIFY round 2 (e85b2d8c) 1.01: the OFF-vs-main compare is not optional. Every set
 * except the baseline-generating L0 compares its OFF shots to a baseline set: the
 * named `UX_SHOTS_BASELINE`, else `L0`. L0 (stage 0) is exempt -- it IS the baseline.
 */
function resolveBaseline(stage: string | undefined, env: string | undefined): string | undefined {
    if (stage === '0') return undefined;
    return env && env.trim() ? env.trim() : 'L0';
}
const BASELINE = resolveBaseline(STAGE, process.env['UX_SHOTS_BASELINE']);
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
const STATES = ['off', 'on'] as const;
const FIVE = ['datasets', 'training', 'templates', 'server', 'jobs'];
/** Icons ON per screen once L2 lands (spec: 6 datasets, 4 server, 6 jobs, 4 projects, 6 training = 26). */
const KPI_ICONS: Record<string, number> = { datasets: 6, training: 6, templates: 0, server: 4, jobs: 6, projects: 4 };
const NO_ICONS: Record<string, number> = Object.fromEntries(Object.keys(KPI_ICONS).map((k) => [k, 0]));
type SetSpec = { screens: string[]; shots: number; iconsOn: Record<string, number>; extra?: string[] };
/** The ONE table: a round's evidence is whatever its row says, nothing derived elsewhere. */
const SETS: Record<string, SetSpec> = {
    L0: { screens: FIVE, shots: 20, iconsOn: NO_ICONS },
    L1: { screens: FIVE, shots: 20, iconsOn: NO_ICONS },
    L2: { screens: [...FIVE, 'projects'], shots: 24, iconsOn: KPI_ICONS },
    L3: { screens: FIVE, shots: 20, iconsOn: KPI_ICONS, extra: ['on-modal-dark.png', 'on-modal-light.png'] },
};
const HIDE_TOGGLE = '[data-testid="ux-toggle"]{display:none}';

/**
 * Fails BEFORE any capture when the baseline set has no shots on disk (the compare
 * would otherwise be skipped or die late): every OFF image its SETS row promises
 * must exist. Throws a `LANE-143:` message naming the first missing file.
 */
function assertBaselineOnDisk(root: string, baseline: string): void {
    const stage = /^L(\d+)/.exec(baseline)?.[1];
    const row = stage === undefined ? undefined : SETS[`L${stage}`];
    if (!row) throw new Error(`LANE-143: unknown baseline set ${baseline}`);
    for (const screen of row.screens) {
        for (const theme of THEMES) {
            const file = `off-${screen}-${theme}.png`;
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
        expect(spec.shots, `LANE-143: ${name} shots = screens x themes x states`).toBe(spec.screens.length * THEMES.length * STATES.length);
        for (const screen of spec.screens) {
            expect(typeof spec.iconsOn[screen], `LANE-143: ${name} has no ON icon count for ${screen}`).toBe('number');
        }
    }
    expect(SETS['L1'].shots).toBe(20);
    expect(SETS['L2'].shots).toBe(24);
    expect(SETS['L3'].shots).toBe(20);
    expect(SETS['L3'].extra, 'LANE-143: L3 adds the two on-modal shots').toEqual(['on-modal-dark.png', 'on-modal-light.png']);
    const sum = (n: string) => SETS[n].screens.reduce((a, s) => a + SETS[n].iconsOn[s], 0);
    expect(sum('L0') + sum('L1'), 'LANE-143: no icon may render before L2').toBe(0);
    expect(sum('L2'), 'LANE-143: L2 renders 26 icons ON').toBe(26);
    expect(STAGE === undefined || SETS[`L${STAGE}`] !== undefined, `LANE-143: unknown shot set ${SET}`).toBe(true);
});

const SPEC = SETS[`L${STAGE}`] ?? SETS['L0'];
const SCREENS = SPEC.screens;

test.describe.configure({ mode: 'serial' });
test.use({ viewport: { width: 1440, height: 900 } });

type Box = Record<string, number>;

/** Heights of every `body *` element keyed by a structural path that ignores `.kpi-icon` nodes. */
async function sweep(page: import('@playwright/test').Page): Promise<Box> {
    return page.evaluate(() => {
        const out: Record<string, number> = {};
        const key = (el: Element): string => {
            const parts: string[] = [];
            for (let n: Element | null = el; n && n !== document.body; n = n.parentElement) {
                const p = n.parentElement!;
                const sibs = Array.from(p.children).filter((c) => !c.classList.contains('kpi-icon'));
                parts.unshift(`${n.tagName.toLowerCase()}:${sibs.indexOf(n)}`);
            }
            return parts.join('>');
        };
        for (const el of Array.from(document.body.querySelectorAll('*'))) {
            if (el.closest('.kpi-icon')) continue;
            out[key(el)] = Math.round(el.getBoundingClientRect().height * 100) / 100;
        }
        return out;
    });
}

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

const dotRows: string[] = [];
const sweepRows: string[] = [];
const baselineRows: string[] = [];
const manifest: { file: string; px: string; bytes: number; sha256: string }[] = [];

test.describe('LANE-143 UX shots', () => {
    test.skip(!process.env['UX_SHOTS'], 'UX_SHOTS=1 required: this spec produces UAT evidence');

    test.beforeAll(() => {
        if (BASELINE) assertBaselineOnDisk(SHOTS_ROOT, BASELINE);
        fs.mkdirSync(path.join(OUT, 'full'), { recursive: true });
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
                const run = async (state: 'off' | 'on') => {
                    await page.goto('/');
                    await page.evaluate(
                        ([t, ux]) => {
                            localStorage.setItem('mrln.theme', t);
                            if (ux) localStorage.setItem('mrln.ux', 'v2');
                            else localStorage.removeItem('mrln.ux');
                        },
                        [theme, state === 'on'] as const,
                    );
                    await page.goto(`/${screen}`);
                    await settle(page, screen);
                    const want = state === 'on' ? SPEC.iconsOn[screen] : 0;
                    const squares = await page.locator('.kpi-icon').count();
                    const icons = await page.locator('.kpi-icon svg').count();
                    const stray = await page.locator('.kpi app-ico').count();
                    expect(squares, `LANE-143: expected ${want} visible KPI tiles/icons on ${screen} (${state}, set ${SET})`).toBe(want);
                    expect(icons, `LANE-143: expected ${want} KPI icon svgs on ${screen} (${state}, set ${SET})`).toBe(want);
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
                    for (const d of await dotContrasts(page)) {
                        dotRows.push(`| ${screen} | ${theme} | ${state} | ${d.cls} | rgb(${d.fill}) | rgb(${d.back}) | ${d.ratio} |`);
                        if (state === 'on') {
                            expect(
                                d.ratio,
                                `LANE-143: painted .sdot (${d.cls}) on ${screen}/${theme} measures ${d.ratio}:1 < 3:1 (rgb(${d.fill}) on rgb(${d.back}))`,
                            ).toBeGreaterThanOrEqual(3);
                        }
                    }
                    // OFF: the toggle leaves layout for the capture; ON: shown for the shot, hidden for the sweep.
                    const style = await page.addStyleTag({ content: HIDE_TOGGLE });
                    const heights = await sweep(page);
                    if (state === 'on') await style.evaluate((el) => el.remove());
                    const file = `${state}-${screen}-${theme}.png`;
                    await page.screenshot({ path: path.join(OUT, file), animations: 'disabled' });
                    const base = BASELINE ? path.join(SHOTS_ROOT, BASELINE, file) : '';
                    const exception = state === 'off' && BASELINE ? baselineException(SHOTS_ROOT, BASELINE, screen, file) : null;
                    if (exception !== null) {
                        // only a screen the baseline SET lacks by design (its SETS row): recorded, not compared
                        baselineRows.push(`| ${file} | - | exception: ${exception} | - | - |`);
                    } else if (state === 'off' && BASELINE) {
                        const clocks: Rect[] = await page.evaluate(() =>
                            Array.from(document.body.querySelectorAll('*'))
                                .filter((el) =>
                                    Array.from(el.childNodes).some(
                                        (n) => n.nodeType === Node.TEXT_NODE && /\b\d{1,2}:\d{2}\b/.test(n.textContent ?? ''),
                                    ),
                                )
                                .map((el) => {
                                    const r = el.getBoundingClientRect();
                                    return { x: Math.floor(r.x) - 1, y: Math.floor(r.y) - 1, w: Math.ceil(r.width) + 3, h: Math.ceil(r.height) + 3 };
                                })
                                .filter((r) => r.w > 3 && r.h > 3),
                        );
                        const d = pixelDiff(fs.readFileSync(path.join(OUT, file)), fs.readFileSync(base), clocks);
                        const skipped = clocks.map((r) => `${r.w}x${r.h}@${r.x},${r.y}`).join(' ') || '-';
                        baselineRows.push(`| ${file} | ${d.size} | ${d.beyond} | ${d.noise} | ${skipped} |`);
                        expect(
                            d.beyond,
                            `LANE-143: ${file} differs from baseline ${BASELINE} in ${d.beyond} px beyond +-${NOISE} (${d.size})`,
                        ).toBe(0);
                    }
                    await fullCapture(page, path.join(OUT, 'full', file));
                    return heights;
                };
                const off = await run('off');
                const on = await run('on');
                const changed = Object.keys(off).filter((k) => k in on && off[k] !== on[k]);
                const missing = Object.keys(off).filter((k) => !(k in on)).length + Object.keys(on).filter((k) => !(k in off)).length;
                sweepRows.push(
                    `| ${screen} | ${theme} | ${Object.keys(off).length} | ${changed.length} | ${missing} |` +
                        (changed.length ? ` ${changed.slice(0, 5).join(', ')}` : ''),
                );
                expect(changed, `LANE-143: ${screen}/${theme} height changes OFF vs ON`).toEqual([]);
            });
        }
    }

    test('LANE-143 toggle displacement is exactly the 32px button plus 8px gap', async ({ page }) => {
        await page.goto('/');
        await page.evaluate(() => {
            localStorage.setItem('mrln.theme', 'dark');
            localStorage.removeItem('mrln.ux');
        });
        await page.goto('/datasets');
        await settle(page, 'datasets');
        const boxes = () =>
            page.evaluate(() => {
                const out: Record<string, number[]> = {};
                const toggle = document.querySelector('[data-testid="ux-toggle"]')!;
                for (const el of Array.from(document.body.querySelectorAll('*'))) {
                    if (el === toggle || toggle.contains(el)) continue;
                    const p: string[] = [];
                    for (let n: Element | null = el; n && n !== document.body; n = n.parentElement) {
                        p.unshift(n.tagName.toLowerCase() + ':' + Array.from(n.parentElement!.children).indexOf(n));
                    }
                    const r = el.getBoundingClientRect();
                    out[p.join('>')] = [r.x, r.y, r.width, r.height].map((v) => Math.round(v * 100) / 100);
                }
                return out;
            });
        const themeBtn = page.locator('[data-testid="ux-toggle"] + button');
        const shown = await boxes();
        const shownTheme = (await themeBtn.boundingBox())!;
        const shownToggle = (await page.getByTestId('ux-toggle').boundingBox())!;
        await page.addStyleTag({ content: HIDE_TOGGLE });
        const hidden = await boxes();
        const hiddenTheme = (await themeBtn.boundingBox())!;
        const diffs = Object.keys(hidden).filter((k) => JSON.stringify(hidden[k]) !== JSON.stringify(shown[k]));
        const lines = [
            `toggle box: ${shownToggle.width}x${shownToggle.height}`,
            `theme toggle x shown=${shownTheme.x} hidden=${hiddenTheme.x} delta=${shownTheme.x - hiddenTheme.x}`,
            `other element boxes differing (toggle excluded, theme toggle included): ${diffs.length}`,
            ...diffs.slice(0, 20).map((k) => `  ${k}: shown ${JSON.stringify(shown[k])} hidden ${JSON.stringify(hidden[k])}`),
        ];
        fs.writeFileSync(path.join(OUT, 'toggle-displacement.txt'), lines.join('\n') + '\n');
        expect(shownToggle.width, 'LANE-143: toggle width').toBe(32);
        expect(shownToggle.height, 'LANE-143: toggle height').toBe(32);
        // The row is right-aligned (`margin-left:auto`), so showing the 32px toggle + 8px gap
        // grows the row LEFTWARD: the theme toggle (after it) stays put and every element before
        // it moves 40px left. (The spec's "theme toggle x is 40px smaller" has the direction
        // backwards; the invariant that matters is that the displacement is exactly 40px.)
        expect(hiddenTheme.x - shownTheme.x, 'LANE-143: theme toggle stays anchored right').toBe(0);
        expect(diffs.length, 'LANE-143: showing the toggle displaced nothing').toBeGreaterThan(0);
        for (const k of diffs) {
            const [sx, sy, sw, sh] = shown[k];
            const [hx, hy, hw, hh] = hidden[k];
            expect(hx - sx, `LANE-143: ${k} x displacement`).toBe(40);
            expect(sy, `LANE-143: ${k} y`).toBe(hy);
            expect(sh, `LANE-143: ${k} height`).toBe(hh);
            // only the row container itself changes width (+40)
            expect(sw - hw === 0 || sw - hw === 40, `LANE-143: ${k} width`).toBe(true);
        }
    });

    for (const theme of THEMES) {
        if (!SPEC.extra) break;
        test(`LANE-143 modal ${theme}`, async ({ page }) => {
            await page.goto('/');
            await page.evaluate((t) => {
                localStorage.setItem('mrln.theme', t);
                localStorage.setItem('mrln.ux', 'v2');
            }, theme);
            await page.goto('/datasets');
            await settle(page, 'datasets');
            await page.getByRole('button', { name: /new dataset/i }).first().click();
            const dialog = page.locator('.modal').first();
            await expect(dialog, 'LANE-143: the New dataset dialog must open for the modal shot').toBeVisible();
            await page.waitForTimeout(500);
            const file = `on-modal-${theme}.png`;
            await page.screenshot({ path: path.join(OUT, file), animations: 'disabled' });
            await fullCapture(page, path.join(OUT, 'full', file));
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
            ['| screen | theme | elements | height changes | structural diffs |', '|---|---|---|---|---|', ...sweepRows].join('\n') + '\n',
        );
        fs.writeFileSync(
            path.join(OUT, 'dot-contrast.md'),
            ['| screen | theme | state | class | fill | backdrop | ratio |', '|---|---|---|---|---|---|---|', ...dotRows].join('\n') + '\n',
        );
        if (!BASELINE) {
            fs.writeFileSync(path.join(OUT, 'baseline-diff.md'), `Set ${SET} generates the baseline: exempt from the OFF-vs-baseline compare.
`);
        } else {
            fs.writeFileSync(
                path.join(OUT, 'baseline-diff.md'),
                [`OFF vs ${BASELINE} (tolerance +-${NOISE} per channel)`, '', '| file | px | beyond | noise | clock text skipped |', '|---|---|---|---|---|', ...baselineRows].join('\n') + '\n',
            );
        }
        expect(manifest.length).toBe(2 * expected);
    });
});
