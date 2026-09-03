import type uPlot from 'uplot';
import { tooltipPlugin } from './stats-tooltip';

/**
 * The plugin's hooks are plain DOM operations, so they are testable with a
 * fake uPlot instance — no canvas needed. Regression pinned here: the tooltip
 * must be anchored inside `u.over` (the plot-area overlay, a positioned
 * element whose coordinate space matches `u.cursor.left/top`), NOT `u.root`,
 * which is unpositioned — anchoring there made the tooltip render at the
 * page's top-left corner instead of beside the cursor.
 */
function fakeU(cursor: Partial<uPlot.Cursor> = {}): uPlot {
    const root = document.createElement('div');
    const over = document.createElement('div');
    root.appendChild(over);
    return { root, over, cursor } as unknown as uPlot;
}

function hook<K extends 'init' | 'setCursor' | 'destroy'>(
    plugin: uPlot.Plugin, name: K,
): (u: uPlot) => void {
    return plugin.hooks[name] as (u: uPlot) => void;
}

describe('tooltipPlugin', () => {
    it('anchors the tooltip inside u.over so cursor coordinates line up', () => {
        const plugin = tooltipPlugin(() => 'text');
        const u = fakeU();
        hook(plugin, 'init')(u);
        expect(u.over.querySelector('.stats-tooltip')).toBeTruthy();
        expect(u.root.children.length).toBe(1); // only u.over — nothing appended to root
    });

    it('positions at cursor + offset and shows the formatted text', () => {
        const plugin = tooltipPlugin(() => 'week of X: 3 completed');
        const u = fakeU({ idx: 2, left: 40, top: 20 });
        Object.defineProperty(u.over, 'clientWidth', { value: 900 });
        Object.defineProperty(u.over, 'clientHeight', { value: 150 });
        hook(plugin, 'init')(u);
        hook(plugin, 'setCursor')(u);
        const el = u.over.querySelector('.stats-tooltip') as HTMLDivElement;
        expect(el.style.display).toBe('block');
        expect(el.textContent).toBe('week of X: 3 completed');
        expect(el.style.left).toBe('52px'); // 40 + 12
        expect(el.style.top).toBe('28px');  // 20 + 8
    });

    it('flips left of the cursor when it would overflow the right edge', () => {
        const plugin = tooltipPlugin(() => 'week of 2026-07-06: 22 completed');
        const u = fakeU({ idx: 19, left: 180, top: 10 });
        Object.defineProperty(u.over, 'clientWidth', { value: 200 });
        Object.defineProperty(u.over, 'clientHeight', { value: 150 });
        hook(plugin, 'init')(u);
        const el = u.over.querySelector('.stats-tooltip') as HTMLDivElement;
        Object.defineProperty(el, 'offsetWidth', { value: 100 });
        hook(plugin, 'setCursor')(u);
        expect(el.style.left).toBe('68px'); // 180 - 12 - 100
    });

    it('flips above the cursor when it would overflow the bottom edge', () => {
        const plugin = tooltipPlugin(() => 'x');
        const u = fakeU({ idx: 3, left: 10, top: 140 });
        Object.defineProperty(u.over, 'clientWidth', { value: 900 });
        Object.defineProperty(u.over, 'clientHeight', { value: 150 });
        hook(plugin, 'init')(u);
        const el = u.over.querySelector('.stats-tooltip') as HTMLDivElement;
        Object.defineProperty(el, 'offsetHeight', { value: 24 });
        hook(plugin, 'setCursor')(u);
        expect(el.style.top).toBe('108px'); // 140 - 8 - 24
    });

    it('hides when the cursor leaves the plot or the formatter returns null', () => {
        const plugin = tooltipPlugin(() => null);
        const u = fakeU({ idx: 1, left: -10, top: 0 });
        hook(plugin, 'init')(u);
        hook(plugin, 'setCursor')(u);
        const el = u.over.querySelector('.stats-tooltip') as HTMLDivElement;
        expect(el.style.display).toBe('none');
    });

    it('removes its element on destroy', () => {
        const plugin = tooltipPlugin(() => 'x');
        const u = fakeU();
        hook(plugin, 'init')(u);
        hook(plugin, 'destroy')(u);
        expect(u.over.querySelector('.stats-tooltip')).toBeNull();
    });
});
