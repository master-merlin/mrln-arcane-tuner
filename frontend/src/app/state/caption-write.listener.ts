import { Injectable } from '@angular/core';
import { WebSocketService } from '../services/websocket.service';
import { CaptionCacheStore } from './caption-cache.store';
import { MediaItemStore } from './media-item.store';

interface CaptionWritten {
    dataset_name: string;
    media_file: string;
    caption: string;
    target: 'original' | 'masked';
}

/** Applies backend `caption.written` events to the caption + media stores so the
 *  workspace grid repaints captions live during a backend caption batch —
 *  replacing the per-image writes the mass-caption modal used to do. */
@Injectable({ providedIn: 'root' })
export class CaptionWriteListener {
    constructor(ws: WebSocketService, captions: CaptionCacheStore, media: MediaItemStore) {
        ws.on<CaptionWritten>('caption.written').subscribe(e => {
            const masked = e.target === 'masked';
            captions.setCaption(e.dataset_name, e.media_file, e.caption, masked);
            if (masked) {
                media.markMaskedCaptioned(e.dataset_name, e.media_file);
            } else {
                media.stampCaption(
                    e.dataset_name, e.media_file,
                    e.media_file.substring(0, e.media_file.lastIndexOf('.')) + '.txt',
                );
            }
        });
    }
}
