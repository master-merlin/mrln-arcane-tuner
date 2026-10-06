import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
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
 * `main` puts it. Sets named L2 or later also shoot `projects` and assert the
 * KPI icon counts; L0/L1 assert zero icons in both switch states.
 */
const SET = process.env['UX_SHOTS_SET'] ?? 'L0';
const BASELINE = process.env['UX_SHOTS_BASELINE'];
const STAGE = Number(/^L(\d+)/.exec(SET)?.[1] ?? '0');
const WITH_ICONS = STAGE >= 2;
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

const SCREENS = ['datasets', 'training', 'templates', 'server', 'jobs', ...(WITH_ICONS ? ['projects'] : [])];
const THEMES = ['dark', 'light'] as const;
const ICON_COUNTS: Record<string, number> = { jobs: 6, datasets: 6, server: 4, projects: 4, training: 6 };
const HIDE_TOGGLE = '[data-testid="ux-toggle"]{display:none}';

test.skip(!process.env['UX_SHOTS'], 'UX_SHOTS=1 required: this spec produces UAT evidence');
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

test.beforeAll(() => {
    fs.mkdirSync(OUT, { recursive: true });
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
                const icons = await page.locator('.kpi-icon svg').count();
                const stray = await page.locator('.kpi app-ico, .kpi-icon').count();
                if (state === 'off' || !WITH_ICONS) {
                    expect(stray, `LANE-143: expected 0 KPI icons on ${screen} (${state}, set ${SET})`).toBe(0);
                } else {
                    expect(icons, `LANE-143: expected ${ICON_COUNTS[screen]} visible KPI tiles/icons on ${screen}`).toBe(
                        ICON_COUNTS[screen],
                    );
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
                if (state === 'off' && BASELINE) {
                    const base = path.join(SHOTS_ROOT, BASELINE, file);
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

test('LANE-143 manifest', async () => {
    const files = fs.readdirSync(OUT).filter((f) => f.endsWith('.png')).sort();
    const expected = WITH_ICONS ? 24 : 20; // 5 (6 with projects) screens x 2 themes x OFF/ON, independent of SCREENS
    expect(files.length, `LANE-143: expected ${expected} shots, got ${files.length}`).toBe(expected);
    for (const f of files) {
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
    if (BASELINE) {
        fs.writeFileSync(
            path.join(OUT, 'baseline-diff.md'),
            [`OFF vs ${BASELINE} (tolerance +-${NOISE} per channel)`, '', '| file | px | beyond | noise | clock text skipped |', '|---|---|---|---|---|', ...baselineRows].join('\n') + '\n',
        );
    }
    expect(manifest.length).toBe(expected);
});
