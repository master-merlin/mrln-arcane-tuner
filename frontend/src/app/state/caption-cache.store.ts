import { Injectable, signal } from '@angular/core';

/**
 * Heavy per-image caption text. Intentionally NOT in MediaItemStore
 * (large, and not broadcast over WS). Shared between the dataset
 * workspace (which renders captions) and the mass-operation modals
 * (which generate them) so a write in a modal repaints the workspace
 * grid live via the workspace `pairs` computed.
 */
export interface CaptionRow {
    caption_content?: string;
    masked_caption_content?: string;
}

/** App-wide cache keyed by `dataset name → (media_file → CaptionRow)`. */
@Injectable({ providedIn: 'root' })
export class CaptionCacheStore {
    private _byDataset = signal<Record<string, Map<string, CaptionRow>>>({});

    /** Reactive view consumed by the workspace `pairs` computed. */
    readonly byDataset = this._byDataset.asReadonly();

    /** Current caption map for a dataset (empty map if none loaded). */
    get(dataset: string): Map<string, CaptionRow> {
        return this._byDataset()[dataset] ?? new Map();
    }

    /** Replace the whole map for a dataset (initial `/pairs` load). */
    seed(dataset: string, rows: Map<string, CaptionRow>): void {
        this._byDataset.update(m => ({ ...m, [dataset]: new Map(rows) }));
    }

    /** Merge one caption field for one image, preserving the other field. */
    setCaption(dataset: string, mediaFile: string, content: string, isMasked: boolean): void {
        this._byDataset.update(m => {
            const map = new Map(m[dataset] ?? []);
            const prev = map.get(mediaFile) ?? {};
            map.set(mediaFile, isMasked
                ? { ...prev, masked_caption_content: content }
                : { ...prev, caption_content: content });
            return { ...m, [dataset]: map };
        });
    }

    /** Overwrite a whole row (optimistic-rollback restore). */
    setRow(dataset: string, mediaFile: string, row: CaptionRow): void {
        this._byDataset.update(m => {
            const map = new Map(m[dataset] ?? []);
            map.set(mediaFile, row);
            return { ...m, [dataset]: map };
        });
    }

    /** Drop a single row (pair deleted). */
    remove(dataset: string, mediaFile: string): void {
        this._byDataset.update(m => {
            const map = new Map(m[dataset] ?? []);
            map.delete(mediaFile);
            return { ...m, [dataset]: map };
        });
    }

    /** Drop the whole dataset entry (forced refresh). */
    clear(dataset: string): void {
        this._byDataset.update(m => {
            const next = { ...m };
            delete next[dataset];
            return next;
        });
    }
}
