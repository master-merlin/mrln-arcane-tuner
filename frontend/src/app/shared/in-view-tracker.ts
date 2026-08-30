import { signal, type Signal } from '@angular/core';

/**
 * Tracks which indexed children of a scroll container are on (or near)
 * screen, so a repeating list can restrict *per-item work that never
 * stops on its own* — a CSS `animation: … infinite`, a poll, a video —
 * to the items a user can actually see.
 *
 * Why this exists, measured rather than assumed: a 263-item workspace
 * held **1503 simultaneously running CSS animations** (789 filmstrip
 * spinner dots + 714 grid loader dots) and rendered at 35.5 ms / 28 fps
 * *standing still* — the idle frame cost equalled the scrolling frame
 * cost, so it was never a scroll problem. Three cheaper levers were
 * measured and did nothing: `content-visibility: auto` on the item,
 * `display: none` on the spinner, and hiding the images. Only the
 * animation count moved it (789 → 3 running took the frame to 17.8 ms).
 *
 * Gating on a load event alone is NOT sufficient and the measurement is
 * the reason we know: the items carry `loading="lazy"`, so an off-screen
 * image never starts loading, never fires `load`, and its spinner spins
 * forever. 34 of 263 images had loaded; 687 animations would have
 * remained. The bound has to come from visibility, not from readiness.
 *
 * `IntersectionObserver` is feature-detected. Where it is absent (jsdom,
 * any SSR pass) `has()` answers `true` for every index, which is exactly
 * the pre-existing behaviour — a missing observer degrades to "show it",
 * never to "hide it".
 */
export interface InViewTracker {
    /** Indices currently intersecting the root (plus `rootMargin`). */
    readonly indices: Signal<ReadonlySet<number>>;
    /**
     * True when `index` is on screen — or when no observer is available,
     * so callers behave exactly as they did before this module existed.
     */
    has(index: number): boolean;
    /**
     * (Re-)observe the current children of `root` matching `selector`.
     * Safe to call after every render; the observer instance is reused,
     * so repeated calls never accumulate observers.
     */
    refresh(root: HTMLElement | null | undefined): void;
    /** Release the observer. After this, `refresh` is a no-op. */
    destroy(): void;
}

export interface InViewTrackerOptions {
    /**
     * Element selector for the observed children. Each match must carry
     * the index in `data-index`; a child without it is ignored rather
     * than silently mapped to 0.
     */
    selector: string;
    /**
     * Margin grown around the root before an item counts as visible.
     * Keeps an item's work started slightly before it scrolls in.
     */
    rootMargin?: string;
}

/**
 * Build a tracker. Creating one has no side effect of its own — no
 * observer exists until the first `refresh` with a live root.
 */
export function createInViewTracker(options: InViewTrackerOptions): InViewTracker {
    const { selector, rootMargin = '300px' } = options;
    const indices = signal<ReadonlySet<number>>(new Set<number>());
    const supported = typeof IntersectionObserver !== 'undefined';

    let observer: IntersectionObserver | null = null;
    let destroyed = false;

    const onEntries = (entries: IntersectionObserverEntry[]): void => {
        // A callback can still be delivered after teardown (the observer
        // queues entries on the task queue). Dropping them here is what
        // makes destroy() observable: no signal write, so no change
        // detection is scheduled for a component that is already gone.
        if (destroyed) return;
        const next = new Set(indices());
        let changed = false;
        for (const entry of entries) {
            const raw = (entry.target as HTMLElement).dataset['index'];
            if (raw == null) continue;
            const index = Number.parseInt(raw, 10);
            if (!Number.isFinite(index)) continue;
            if (entry.isIntersecting) {
                if (!next.has(index)) { next.add(index); changed = true; }
            } else if (next.delete(index)) {
                changed = true;
            }
        }
        if (changed) indices.set(next);
    };

    return {
        indices: indices.asReadonly(),
        has(index: number): boolean {
            if (!supported || destroyed) return true;
            return indices().has(index);
        },
        refresh(root: HTMLElement | null | undefined): void {
            if (!supported || destroyed || !root) return;
            observer ??= new IntersectionObserver(onEntries, { root, rootMargin });
            observer.disconnect();
            for (const el of root.querySelectorAll<HTMLElement>(selector)) {
                observer.observe(el);
            }
        },
        destroy(): void {
            destroyed = true;
            observer?.disconnect();
            observer = null;
        },
    };
}
