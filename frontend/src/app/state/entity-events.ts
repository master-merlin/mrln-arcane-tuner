export type EntityChangedMessage =
    | { entity: string; op: 'created' | 'updated' | 'deleted'; id: string; payload: unknown }
    | { entity: string; op: 'bulk_deleted'; payload: { ids: string[] } };

export function isBulkDeletedPayload(p: unknown): p is { ids: string[] } {
    return typeof p === 'object' && p !== null && Array.isArray((p as { ids?: unknown }).ids);
}
