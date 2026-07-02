/**
 * Human-readable file size — "1.5 GB", "149.6 MB", "0 B". Non-byte units
 * always carry one decimal so `149,600,000` reads as "149.6 MB" rather than
 * rounding to "150 MB"; raw bytes stay as integers.
 *
 * Shared by the Datasets and Jobs screens (was a character-identical
 * private copy in each). See `workspace/shared/media-meta.ts`'s
 * `formatBytes` for a deliberately-different variant used by the media
 * panel — do not consolidate that one here (see its doc comment).
 */
export function formatBytes(n: number): string {
    if (!Number.isFinite(n) || n <= 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0;
    while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(i > 0 ? 1 : 0)} ${units[i]}`;
}
