import { Injectable } from '@angular/core';
import { WebSocketService } from '../services/websocket.service';
import { ToastService } from '../services/toast';

interface HarmonizeSummary {
    dataset_name: string;
    processed: number;
    converted: number;
    renamed: number;
}

/** Toasts the result of a backend harmonize task. Root listener (survives the
 *  Analyze modal closing). Mirrors `mask-apply-summary.listener.ts`. */
@Injectable({ providedIn: 'root' })
export class HarmonizeSummaryListener {
    constructor(ws: WebSocketService, toast: ToastService) {
        ws.on<HarmonizeSummary>('harmonize.summary').subscribe(e => {
            toast.success(
                `Harmonized "${e.dataset_name}" — ${e.processed} processed, ` +
                `${e.converted} converted, ${e.renamed} renamed.`);
        });
    }
}
