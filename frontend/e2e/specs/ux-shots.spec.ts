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

const sweepRows: string[] = [];
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
                // OFF: the toggle leaves layout for the capture; ON: shown for the shot, hidden for the sweep.
                const style = await page.addStyleTag({ content: HIDE_TOGGLE });
                const heights = await sweep(page);
                if (state === 'on') await style.evaluate((el) => el.remove());
                const file = `${state}-${screen}-${theme}.png`;
                await page.screenshot({ path: path.join(OUT, file), animations: 'disabled' });
                if (state === 'off' && BASELINE) {
                    const base = path.join(SHOTS_ROOT, BASELINE, file);
                    expect(
                        fs.readFileSync(path.join(OUT, file)).equals(fs.readFileSync(base)),
                        `LANE-143: ${file} differs from baseline ${BASELINE}`,
                    ).toBe(true);
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
    expect(manifest.length).toBe(expected);
});
