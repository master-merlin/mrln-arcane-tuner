export type EntityOp = 'created' | 'updated' | 'deleted' | 'bulk_deleted';

export interface EntityChangedMessage {
    entity: string;
    op: EntityOp;
    id: string;
    payload: unknown;
}

export function isBulkDeletedPayload(p: unknown): p is { ids: string[] } {
    return typeof p === 'object' && p !== null && Array.isArray((p as { ids?: unknown }).ids);
}
