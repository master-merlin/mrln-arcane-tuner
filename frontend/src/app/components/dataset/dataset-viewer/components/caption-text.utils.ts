// caption-text.utils.ts
/** Split a caption into trimmed, non-empty comma-separated tags. */
function splitTags(caption: string): string[] {
    return caption
        .split(',')
        .map(t => t.replace(/\s+/g, ' ').trim())
        .filter(t => t.length > 0);
}

/** Collapse whitespace, enforce ", " separators, drop empty segments. */
export function normalizeCommaSpacing(caption: string): string {
    return splitTags(caption).join(', ');
}

/** Remove case-insensitive duplicate tags (keep first occurrence) and normalize spacing. */
export function dedupeTags(caption: string): string {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const tag of splitTags(caption)) {
        const key = tag.toLowerCase();
        if (!seen.has(key)) {
            seen.add(key);
            out.push(tag);
        }
    }
    return out.join(', ');
}
