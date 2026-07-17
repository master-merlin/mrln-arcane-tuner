import type uPlot from 'uplot';

/** Returns the tooltip line for the hovered index, or null to hide. */
export type TooltipFormatter = (u: uPlot, idx: number) => string | null;

/**
 * Minimal cursor-follow tooltip: one absolutely-positioned div inside the
 * uPlot plot-area overlay (`u.over`). It must live in `u.over` — that is the
 * positioned element whose coordinate space `u.cursor.left/top` are relative
 * to; anchored anywhere less specific, the absolute coords resolve against
 * the page and the tooltip renders at the viewport's top-left. Self-styled
 * (inline, design tokens via CSS vars) so no global or ::ng-deep CSS is
 * needed.
 */
export function tooltipPlugin(format: TooltipFormatter): uPlot.Plugin {
    let el: HTMLDivElement | null = null;
    return {
        hooks: {
            init(u: uPlot) {
                el = document.createElement('div');
                el.className = 'stats-tooltip'; // identification only — styling is inline
                Object.assign(el.style, {
                    position: 'absolute', display: 'none', pointerEvents: 'none',
                    zIndex: '10', whiteSpace: 'nowrap', padding: '4px 8px',
                    fontSize: '11px', borderRadius: '4px',
                    background: 'var(--color-surface-high)',
                    border: '1px solid var(--color-border-subtle)',
                    color: 'var(--color-text-secondary)',
                });
                u.over.appendChild(el);
            },
            setCursor(u: uPlot) {
                if (!el) return;
                const { idx, left, top } = u.cursor;
                const text = idx == null ? null : format(u, idx);
                if (text == null || left == null || left < 0) {
                    el.style.display = 'none';
                    return;
                }
                el.textContent = text;
                el.style.display = 'block';
                // Flip to the other side of the cursor when the tooltip would
                // spill past the plot area (guards: jsdom / pre-layout report
                // zero sizes — keep the simple placement there).
                const areaW = u.over.clientWidth;
                const areaH = u.over.clientHeight;
                let x = left + 12;
                if (areaW > 0 && el.offsetWidth > 0 && x + el.offsetWidth > areaW) {
                    x = Math.max(0, left - 12 - el.offsetWidth);
                }
                const cursorTop = top ?? 0;
                let y = cursorTop + 8;
                if (areaH > 0 && el.offsetHeight > 0 && y + el.offsetHeight > areaH) {
                    y = Math.max(0, cursorTop - 8 - el.offsetHeight);
                }
                el.style.left = `${x}px`;
                el.style.top = `${y}px`;
            },
            destroy() {
                el?.remove();
                el = null;
            },
        },
    };
}
