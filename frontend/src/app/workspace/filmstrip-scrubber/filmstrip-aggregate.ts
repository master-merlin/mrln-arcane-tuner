/**
 * Filmstrip aggregation — collapses a long image list into ≤ threshold
 * cells so the bottom-of-workspace scrubber stays readable. Each cell
 * covers a contiguous slice of the source list. A cell is considered
 * "ready" for a flag (harmonized / captioned / masked) only if **every**
 * image in its range carries that flag — a pessimistic colour avoids
 * misleading the user into thinking a partially-ready group is done.
 *
 * Pure function; UI lives in `filmstrip-scrubber.component.ts`.
 */

export interface CellState {
    harmonized: boolean;
    captioned: boolean;
    masked: boolean;
}

export interface StripCell {
    /** Inclusive start index in the source `images` array. */
    startIndex: number;
    /** Number of images this cell covers. */
    count: number;
    /** Aggregated readiness (AND over the slice). */
    state: CellState;
}

interface ImageReadiness {
    harmonized?: boolean;
    captioned?: boolean;
    masked?: boolean;
}

export function aggregateFilmstrip(
    images: ReadonlyArray<ImageReadiness>,
    threshold: number,
): StripCell[] {
    if (images.length === 0) return [];

    const cellCount = Math.min(images.length, Math.max(1, threshold));
    const baseSize = Math.floor(images.length / cellCount);
    const remainder = images.length % cellCount;

    const out: StripCell[] = [];
    let cursor = 0;
    for (let i = 0; i < cellCount; i++) {
        const size = baseSize + (i < remainder ? 1 : 0);
        if (size === 0) continue;
        const slice = images.slice(cursor, cursor + size);
        out.push({
            startIndex: cursor,
            count: size,
            state: {
                harmonized: slice.every(s => !!s.harmonized),
                captioned: slice.every(s => !!s.captioned),
                masked: slice.every(s => !!s.masked),
            },
        });
        cursor += size;
    }
    return out;
}
