import type uPlot from 'uplot';

/** Returns the tooltip line for the hovered index, or null to hide. */
export type TooltipFormatter = (u: uPlot, idx: number) => string | null;

/**
 * Minimal cursor-follow tooltip: one absolutely-positioned div inside the
 * uPlot root. Self-styled (inline, design tokens via CSS vars) so no global
 * or ::ng-deep CSS is needed.
 */
export function tooltipPlugin(format: TooltipFormatter): uPlot.Plugin {
    let el: HTMLDivElement | null = null;
    return {
        hooks: {
            init(u: uPlot) {
                el = document.createElement('div');
                Object.assign(el.style, {
                    position: 'absolute', display: 'none', pointerEvents: 'none',
                    zIndex: '10', whiteSpace: 'nowrap', padding: '4px 8px',
                    fontSize: '11px', borderRadius: '4px',
                    background: 'var(--color-surface-high)',
                    border: '1px solid var(--color-border-subtle)',
                    color: 'var(--color-text-secondary)',
                });
                u.root.appendChild(el);
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
                el.style.left = `${left + 12}px`;
                el.style.top = `${(top ?? 0) + 8}px`;
            },
            destroy() {
                el?.remove();
                el = null;
            },
        },
    };
}
