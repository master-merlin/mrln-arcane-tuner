import { Injectable } from '@angular/core';
import { WebSocketService } from '../services/websocket.service';
import { ToastService } from '../services/toast';

interface MaskApplySummary {
    dataset_name: string;
    applied: number;
    skipped: number;
    missing_masks_count: number;
}

/** Toasts the result of a backend mask-apply task. A root listener (not modal-
 *  scoped) so the summary survives the mask modal returning to its tabs or
 *  being closed. Mirrors `caption-write.listener.ts`. */
@Injectable({ providedIn: 'root' })
export class MaskApplySummaryListener {
    constructor(ws: WebSocketService, toast: ToastService) {
        ws.on<MaskApplySummary>('mask.apply_summary').subscribe(e => {
            if (e.missing_masks_count > 0) {
                toast.warning(
                    `Applied ${e.applied} mask(s); ${e.missing_masks_count} image(s) had no mask.`);
            } else {
                toast.success(
                    `Applied ${e.applied} mask(s)${e.skipped ? ` (${e.skipped} skipped)` : ''}.`);
            }
        });
    }
}
