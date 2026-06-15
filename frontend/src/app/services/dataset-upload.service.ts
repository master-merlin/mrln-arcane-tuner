import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { DatasetService } from './dataset';
import { DatasetStore } from '../state/dataset.store';
import { DatasetSyncService } from '../state/dataset-sync.service';
import { MediaItemStore } from '../state/media-item.store';
import { ToastService } from './toast';

/** Caption file extensions (mirrors the backend `CAPTION_EXTS`). Anything
 *  else uploaded is treated as a media file for the optimistic count. */
const CAPTION_EXTS = ['.txt', '.caption'];

/** Image extensions eligible to seed an optimistic card preview. */
const IMAGE_EXTS = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.avif'];

/** Lower-cased extension including the leading dot, or '' when none. */
function extOf(filename: string): string {
    const dot = filename.lastIndexOf('.');
    return dot >= 0 ? filename.slice(dot).toLowerCase() : '';
}

/** Filename without its extension (the pairing "stem"). */
function stemOf(filename: string): string {
    const base = filename.split(/[\\/]/).pop() ?? filename;
    const dot = base.lastIndexOf('.');
    return dot >= 0 ? base.slice(0, dot) : base;
}

/** Outcome of a control upload: which target stems auto-paired, and the
 *  files whose stem matched no target (handed to the manual-assignment tray). */
export interface ControlUploadResult {
    matched: string[];
    unmatched: File[];
}

/**
 * The single upload authority for the app.
 *
 * Both the datasets-screen cards and the in-workspace grid drop onto an edit
 * dataset, but a dropped image can be a TARGET ("after", dataset root) or a
 * CONTROL ("before", `control/` slot). This service owns both paths so the
 * card, the grid drop zone, the pair-role-chooser, and the Pairs manager all
 * upload identically:
 *
 * - {@link uploadTargets} reproduces the original card-drop contract — caption
 *   files don't inflate the image count, the first image seeds an optimistic
 *   preview, and the follow-up is the BACKGROUNDED safe rescan whose
 *   `dataset.invalidated` broadcast refreshes counts/preview authoritatively.
 * - {@link uploadControls} auto-matches each file's stem against existing
 *   target stems and uploads the matches into the chosen control slot; the
 *   leftovers come back for a manual-assignment UI. Matching is
 *   case-insensitive but the upload always carries the REAL target stem so the
 *   backend's case-sensitive stem-pairing convention holds.
 */
@Injectable({ providedIn: 'root' })
export class DatasetUploadService {
    private api = inject(DatasetService);
    private datasets = inject(DatasetStore);
    private sync = inject(DatasetSyncService);
    private mediaItems = inject(MediaItemStore);
    private toast = inject(ToastService);

    /**
     * Upload files into the dataset root as training targets. Uploads run in
     * parallel; once they settle we apply one classified optimistic count
     * update (captions vs images) plus a preview from the first image, then
     * kick off the backgrounded safe rescan.
     */
    uploadTargets(name: string, files: FileList | File[]): void {
        const list = Array.from(files ?? []);
        if (!name || list.length === 0) return;

        let completed = 0;
        let failed = 0;
        let media = 0;
        let caption = 0;
        let firstImage: string | undefined;

        const settle = () => {
            if (completed + failed < list.length) return;
            this.datasets.applyOptimisticUpload(name, { media, caption }, firstImage);
            this.finalizeTargets(name, completed, failed);
        };

        for (const file of list) {
            this.api.uploadFile(name, file).subscribe({
                next: (res) => {
                    completed++;
                    const fname = res?.filename ?? file.name;
                    const ext = extOf(fname);
                    if (CAPTION_EXTS.includes(ext)) {
                        caption++;
                    } else {
                        media++;
                        if (!firstImage && IMAGE_EXTS.includes(ext)) firstImage = fname;
                    }
                    settle();
                },
                error: (err: unknown) => {
                    failed++;
                    console.error('[dataset-upload] target upload failed', file.name, err);
                    settle();
                },
            });
        }
    }

    private finalizeTargets(name: string, ok: number, failed: number): void {
        if (ok > 0) {
            this.toast.success(
                `${ok} file${ok === 1 ? '' : 's'} uploaded — rescanning in the background.`,
            );
            this.api.rescanDataset(name, 'safe').subscribe({
                error: (err: unknown) => {
                    const e = err as { error?: { detail?: string }; message?: string };
                    this.toast.error('Rescan after upload failed: ' + (e.error?.detail || e.message));
                },
            });
        }
        if (failed > 0) {
            this.toast.error(`${failed} file${failed === 1 ? '' : 's'} failed to upload.`);
        }
    }

    /**
     * Upload control ("before") images into `slot` (1..3), auto-pairing each
     * by filename stem to an existing target. Returns the matched target stems
     * and the unmatched files so the caller can drive a manual tray. After any
     * matched upload the dataset is re-synced so the grid badges update.
     */
    async uploadControls(
        name: string, files: FileList | File[], slot: number,
    ): Promise<ControlUploadResult> {
        const list = Array.from(files ?? []);
        const matched: string[] = [];
        const unmatched: File[] = [];
        if (!name || list.length === 0) return { matched, unmatched };

        // Map lowercased target stem → the REAL (case-preserving) stem, so a
        // case-insensitive filename match still uploads with the exact stem the
        // backend pairs against.
        const realByLower = new Map<string, string>();
        for (const row of this.mediaItems.byDataset(name)()) {
            const s = stemOf(row.media_file);
            realByLower.set(s.toLowerCase(), s);
        }

        const uploads: Promise<unknown>[] = [];
        for (const file of list) {
            const real = realByLower.get(stemOf(file.name).toLowerCase());
            if (real) {
                matched.push(real);
                uploads.push(
                    firstValueFrom(this.api.uploadControlFile(name, file, slot, real)),
                );
            } else {
                unmatched.push(file);
            }
        }

        if (uploads.length) {
            const results = await Promise.allSettled(uploads);
            const ok = results.filter(r => r.status === 'fulfilled').length;
            const bad = results.length - ok;
            await this.sync.refreshDataset(name);
            if (ok > 0) this.toast.success(`Paired ${ok} control image${ok === 1 ? '' : 's'}.`);
            if (bad > 0) this.toast.error(`${bad} control upload${bad === 1 ? '' : 's'} failed.`);
        }

        return { matched, unmatched };
    }
}
