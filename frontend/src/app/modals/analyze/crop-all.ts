/** One image queued for batch crop. `targetWidth/Height` come from the
 *  analysis pass (the harmonized crop target for that image). */
export interface CropAllItem {
    path: string;
    targetWidth: number;
    targetHeight: number;
}

export interface CropAllProgress {
    current: number;   // 1-based count of items attempted
    total: number;
    path: string;      // the item just attempted
}

export interface CropAllOptions {
    /** 9-position anchor applied to every crop. */
    origin: string;
    /** Performs one crop. Rejects on failure (caller HTTP error). */
    crop: (item: CropAllItem, origin: string) => Promise<void>;
    onProgress: (p: CropAllProgress) => void;
    /** Polled before each item — return true to stop the batch. */
    isCancelled: () => boolean;
}

export interface CropAllResult {
    ok: number;
    failed: number;
    cancelled: boolean;
}

/**
 * Sequentially crop a batch of images via the single-image endpoint.
 * Sequential (not parallel) on purpose: each crop rewrites a file on
 * disk server-side, and the backend serialises dataset metadata writes —
 * parallel requests would contend. A single failure is tallied, not
 * thrown, so one bad image doesn't abort the whole run.
 */
export async function runCropAll(
    items: CropAllItem[],
    opts: CropAllOptions,
): Promise<CropAllResult> {
    let ok = 0;
    let failed = 0;
    for (let i = 0; i < items.length; i++) {
        if (opts.isCancelled()) return { ok, failed, cancelled: true };
        const item = items[i];
        try {
            await opts.crop(item, opts.origin);
            ok++;
        } catch {
            failed++;
        }
        opts.onProgress({ current: i + 1, total: items.length, path: item.path });
    }
    return { ok, failed, cancelled: false };
}
