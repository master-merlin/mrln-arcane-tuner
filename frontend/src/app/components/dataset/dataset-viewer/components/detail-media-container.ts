import { Component, effect, input, output, signal } from '@angular/core';
import { PanZoomDirective } from '../../../../workspace/shared/pan-zoom.directive';
import type { DatasetPair } from '../../../../services/dataset';

@Component({
    selector: 'app-detail-media-container',
    host: { class: 'flex-1 flex flex-col overflow-hidden bg-base' },
    imports: [PanZoomDirective],
    template: `
    <div class="w-full h-full flex flex-col relative min-h-0 items-center justify-center">
        @if (currentPair(); as pair) {
            <div class="relative w-full flex-1 min-h-0 p-4 flex items-center justify-center overflow-hidden"
                 appPanZoom [zoom]="zoom()" (zoomChange)="zoomChange.emit($event)">
                @if (pair.media_type === 'video') {
                    <video [src]="getMediaUrl(pair.media_file)" controls
                           class="max-w-full max-h-full object-contain rounded-theme-lg"></video>
                } @else {
                    <img [src]="getDisplayUrl(pair)"
                         (error)="onOverlayError(pair)"
                         class="max-w-full max-h-full object-contain rounded-theme-lg"
                         alt="Dataset Image">
                }
            </div>
        }

        <button (click)="prevRequested.emit()" class="absolute left-4 top-1/2 -translate-y-1/2 bg-base/60 hover:bg-overlay text-white p-3 rounded-full opacity-0 group-hover:opacity-100 transition-all transform hover:scale-110">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>
        </button>
        <button (click)="nextRequested.emit()" class="absolute right-4 top-1/2 -translate-y-1/2 bg-base/60 hover:bg-overlay text-white p-3 rounded-full opacity-0 group-hover:opacity-100 transition-all transform hover:scale-110">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
        </button>
    </div>
    `,
    styles: []
})
export class DetailMediaContainerComponent {
    currentPair = input.required<DatasetPair>();
    datasetName = input.required<string>();
    mediaBaseUrl = input.required<string>();
    lastUpdateTime = input<number>(0);
    showMasked = input<boolean>(false);
    showOverlay = input<boolean>(true);
    apiUrl = input<string>('');
    /** Zoom factor (1 = 100%) applied to the media via a CSS scale transform. */
    zoom = input<number>(1);

    prevRequested = output<void>();
    nextRequested = output<void>();
    deleteRequested = output<void>();
    /** Wheel-zoom (from the pan/zoom directive) bubbled up so the host keeps
     *  the footer's zoom readout + controls in sync. */
    zoomChange = output<number>();

    /**
     * `media_file` keys whose overlay URL returned an error. Once a
     * pair lands here, `getDisplayUrl` skips the overlay path and
     * falls back to the parent media URL.
     *
     * Cleared on every ``lastUpdateTime`` (mediaRev) bump — a re-save
     * or bake might have produced a now-valid overlay PNG for a key
     * that was previously 404'ing, so the set is dropped to allow
     * a fresh fetch attempt. If the URL still 404s, ``onOverlayError``
     * will add it back.
     */
    protected failedOverlays = signal<Set<string>>(new Set());

    constructor() {
        // Drop previously-failed overlays whenever the cache-bust counter
        // bumps — those URLs may now be valid (overlay re-rendered).
        effect(() => {
            this.lastUpdateTime();    // track the signal
            this.failedOverlays.set(new Set());
        });
    }

    protected onOverlayError(pair: DatasetPair): void {
        const mf = pair?.media_file;
        if (!mf) return;
        this.failedOverlays.update(s => {
            if (s.has(mf)) return s;
            const next = new Set(s);
            next.add(mf);
            return next;
        });
    }

    getMediaUrl(relativePath: string): string {
        return `${this.mediaBaseUrl()}/${encodeURIComponent(this.datasetName())}/${encodeURIComponent(relativePath)}?t=${this.lastUpdateTime()}`;
    }

    getOverlayUrl(imagePath: string): string {
        return `${this.apiUrl()}/datasets/${encodeURIComponent(this.datasetName())}/overlay/${encodeURIComponent(imagePath)}?t=${this.lastUpdateTime()}`;
    }

    getDisplayUrl(pair: DatasetPair): string {
        if (this.showMasked() && pair.metadata?.has_masked) {
            const stem = pair.media_file.substring(0, pair.media_file.lastIndexOf('.'));
            return this.getMediaUrl('masked/' + stem + '.jpg');
        }
        if (
            this.showOverlay() && pair.metadata?.has_overlay
            && !this.failedOverlays().has(pair.media_file)
        ) {
            return this.getOverlayUrl(pair.media_file);
        }
        return this.getMediaUrl(pair.media_file);
    }
}
