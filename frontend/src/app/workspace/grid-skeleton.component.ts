import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

/**
 * Placeholder tiles for the browse grid while a dataset's first `/pairs`
 * is in flight (LANE-58).
 *
 * The workspace body used to paint `var(--color-base)` with NO children
 * until `/pairs` answered — MEASURED at 2.2 s on a 33-item dataset whose
 * caption sidecars were cold, and the user reads that as "a black screen,
 * then the images". The backend now reads those sidecars concurrently, but a
 * first paint that depends on a round trip is a first paint that can blank;
 * this component is what the user sees instead: the same tile boxes the grid
 * will draw, each with the grid's own loader dots, so the screen says
 * "loading" rather than nothing.
 *
 * Deliberately dumb: no data, no requests, no state. `slots` is the number
 * of boxes to draw (the workspace derives it from the dataset row's media
 * count, capped to what could plausibly be on screen), `columns` the density
 * the real grid is about to use, so the swap to real tiles does not reflow.
 */
@Component({
    selector: 'app-grid-skeleton',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="grid-skeleton" data-testid="grid-skeleton" role="status" aria-busy="true"
             aria-label="Loading dataset"
             [style.grid-template-columns]="'repeat(' + columns() + ', minmax(0, 1fr))'">
            @for (i of indices(); track i) {
                <div class="tile bg-surface-mid/50 border border-surface-mid rounded-theme-xl overflow-hidden flex flex-col h-[480px]"
                     data-testid="grid-skeleton-tile">
                    <div class="h-80 bg-media-backdrop relative overflow-hidden flex-shrink-0">
                        <span class="grid-thumb-loader" aria-hidden="true">
                            <span></span><span></span><span></span>
                        </span>
                    </div>
                </div>
            }
        </div>
    `,
    styles: [`
        :host { display: block; min-height: 0; overflow: hidden; padding: 16px; }
        .grid-skeleton { display: grid; gap: 8px; }
        /* Mirrors the grid's loader dots (viewer-grid-view) — both are
           component-scoped, so the rule has to live twice. */
        .grid-thumb-loader {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            pointer-events: none;
        }
        .grid-thumb-loader > span {
            width: 8px;
            height: 8px;
            background: var(--color-text-subtle);
            border-radius: 50%;
            animation: grid-skeleton-bounce 1.1s ease-in-out infinite;
        }
        .grid-thumb-loader > span:nth-child(2) { animation-delay: 0.15s; }
        .grid-thumb-loader > span:nth-child(3) { animation-delay: 0.30s; }
        @keyframes grid-skeleton-bounce {
            0%, 80%, 100% { transform: translateY(0); opacity: 0.35; }
            40% { transform: translateY(-7px); opacity: 1; }
        }
    `],
})
export class GridSkeletonComponent {
    /** Placeholder tiles to draw. Non-finite or negative values draw none. */
    slots = input<number>(0);
    /** Grid columns — the density the real grid will paint at. */
    columns = input<number>(5);

    protected indices = computed(() => {
        const n = Math.floor(this.slots());
        if (!Number.isFinite(n) || n <= 0) return [];
        return Array.from({ length: n }, (_, i) => i);
    });
}
