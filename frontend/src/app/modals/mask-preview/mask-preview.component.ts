import {
    ChangeDetectionStrategy,
    Component,
    OnInit,
    computed,
    inject,
    signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { IcoComponent } from '../../icons/ico.component';
import { OverlayStore } from '../../state/overlay.store';
import { MediaItemStore } from '../../state/media-item.store';
import { ToastService } from '../../services/toast';
import { RuntimeConfigService } from '../../services/runtime-config.service';

interface MaskPreviewData {
    datasetId?: string;
    datasetName?: string;
    pair?: any;
    /** Open mode — 'preview' (composite) or 'mask' (raw alpha). */
    mode?: 'preview' | 'mask';
}

/**
 * Mask preview modal — toggles between composite preview (mask applied
 * over the source image, with adjustable mix) and raw alpha channel.
 *
 * Ports the workflow from the orphan
 * [viewer-mask-preview-modal](../../components/dataset/dataset-viewer/components/viewer-mask-preview-modal.ts)
 * and the design shell from `modals-more.jsx → MaskPreviewModal`. The
 * `Bake mask` button emits a `mask-baked` data update on the modal via
 * close, leaving rebake wiring to the caller (e.g. the DetailsMode mask
 * sidebar).
 *
 * TODO(frontend): wire `Bake mask` to the dataset mask-bake endpoint
 * once the new workspace exposes a callback channel for that.
 */
@Component({
    selector: 'app-modal-mask-preview',
    standalone: true,
    imports: [FormsModule, IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head mp-head">
            <div>
                <div class="eyebrow">{{ mode() === 'preview' ? 'MASK COMPOSITE PREVIEW' : 'RAW ALPHA CHANNEL' }}</div>
                <div class="modal-title mono">{{ data.pair?.media_file ?? '—' }}</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body mp-body">
            <div class="mp-toggle">
                <button class="chip"
                        type="button"
                        [class.solid]="mode() === 'preview'"
                        (click)="mode.set('preview')">composite</button>
                <button class="chip"
                        type="button"
                        [class.solid]="mode() === 'mask'"
                        (click)="mode.set('mask')">raw alpha</button>
            </div>

            <div class="mp-viewport" [class.mask]="mode() === 'mask'">
                @if (imageUrl(); as url) {
                    <!-- Bouncing-dots overlay shown while the backend is
                         re-rendering the composite (e.g. on slider drag) —
                         hidden once the current URL reports a load event. -->
                    @if (isImageLoading()) {
                        <span class="mp-loader" aria-hidden="true">
                            <span></span><span></span><span></span>
                        </span>
                    }
                    <img [src]="url" class="mp-image" alt=""
                         (load)="onImageLoaded($event)"/>
                } @else {
                    <div class="muted mp-empty-text">No image to preview.</div>
                }
                <div class="mp-corner">{{ mode() === 'preview' ? 'BACKEND COMPOSITE' : 'SOURCE ALPHA' }}</div>
            </div>
        </div>

        <div class="modal-foot mp-foot">
            @if (mode() === 'preview') {
                <div class="mp-slider">
                    <div class="mp-slider-head">
                        <span class="eyebrow">COMPOSITE ALPHA MIX</span>
                        <span class="mono brand">{{ (opacity() * 100).toFixed(0) }}%</span>
                    </div>
                    <input type="range" min="0" max="1" step="0.05"
                           [(ngModel)]="opacity" class="mp-range">
                </div>
                <button class="btn cta" type="button"
                        [disabled]="!canBake() || isBaking()"
                        [title]="bakeTooltip()"
                        (click)="onBake()">
                    @if (isBaking()) {
                        <span class="mp-spinner" aria-hidden="true">
                            <span></span><span></span><span></span>
                        </span>
                        Baking…
                    } @else {
                        <app-ico name="Check" [size]="12"/> Bake mask
                    }
                </button>
            } @else {
                <div class="mp-meta">
                    @if (data.pair?.metadata?.mask_info; as info) {
                        <span><span class="muted">Resolution </span><span class="mono">{{ info.width }}×{{ info.height }}</span></span>
                        <span><span class="muted">Size </span><span class="mono">{{ ((info.size_bytes ?? 0) / 1024).toFixed(1) }} KB</span></span>
                    }
                </div>
                <span class="muted mp-hint">ESC closes</span>
            }
        </div>
    `,
    styles: [`
        .modal-title { font-size: 13px; font-weight: 600; margin-top: 2px; }
        .mp-body { padding: 14px 18px 18px; }
        .mp-toggle { display: flex; gap: 6px; justify-content: center; margin-bottom: 14px; }
        .chip {
            padding: 4px 12px; border-radius: 999px;
            font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            color: var(--color-text-secondary);
            cursor: pointer;
        }
        .chip.solid {
            background: var(--color-brand); color: white; border-color: var(--color-brand);
        }
        .mp-viewport {
            display: flex; align-items: center; justify-content: center;
            padding: 14px;
            background: var(--color-base);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-xl);
            min-height: 380px;
            position: relative;
        }
        .mp-viewport.mask { background: #0a0a0a; }
        .mp-image {
            max-width: 90%;
            max-height: 60vh;
            border-radius: var(--radius-theme-lg);
            box-shadow: var(--shadow-lg);
            object-fit: contain;
        }
        .mp-empty-text { padding: 40px; }
        .mp-corner {
            position: absolute; bottom: 16px; left: 22px;
            background: oklch(0.08 0.01 265 / 0.75);
            backdrop-filter: blur(6px);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 10px; font-weight: 700;
            letter-spacing: 0.16em; text-transform: uppercase;
            color: var(--color-text-secondary);
        }

        .mp-foot { display: flex; align-items: center; gap: 14px; }
        .mp-slider { flex: 1; }
        .mp-slider-head { display: flex; justify-content: space-between; margin-bottom: 4px; }
        .eyebrow {
            font-size: 10px; font-weight: 700; letter-spacing: 0.14em;
            text-transform: uppercase; color: var(--color-text-muted);
        }
        .mono.brand { color: var(--color-brand); font-weight: 700; }
        .mp-range { width: 100%; accent-color: var(--color-brand); }
        .muted { color: var(--color-text-muted); }
        .mp-meta { flex: 1; display: flex; gap: 14px; font-size: 11px; }
        .mp-hint { font-size: 11px; font-style: italic; }
        .btn.cta {
            display: inline-flex; align-items: center; gap: 8px;
            background: var(--color-brand);
            color: white;
            font-weight: 700;
            padding: 9px 16px;
            border-radius: var(--radius-theme-lg);
            cursor: pointer;
            border: none;
        }
        .btn.cta:disabled { opacity: 0.5; cursor: not-allowed; }

        /* Three-dot loader inside the preview viewport. Sits centered on
           top of the still-displayed previous image so the user has
           confidence the slider drag is triggering a backend render. */
        .mp-loader {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            pointer-events: none;
            z-index: 2;
        }
        .mp-loader > span {
            width: 10px;
            height: 10px;
            background: var(--color-brand);
            border-radius: 50%;
            box-shadow: 0 1px 4px oklch(0 0 0 / 0.45);
            animation: mp-bounce 1.1s ease-in-out infinite;
        }
        .mp-loader > span:nth-child(2) { animation-delay: 0.15s; }
        .mp-loader > span:nth-child(3) { animation-delay: 0.30s; }
        @keyframes mp-bounce {
            0%, 80%, 100% { transform: translateY(0); opacity: 0.45; }
            40%           { transform: translateY(-8px); opacity: 1; }
        }

        /* Inline spinner used inside the Bake button while the bake POST
           is in flight. Same dots, smaller scale. */
        .mp-spinner {
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .mp-spinner > span {
            width: 5px;
            height: 5px;
            background: currentColor;
            border-radius: 50%;
            animation: mp-bounce 1.1s ease-in-out infinite;
        }
        .mp-spinner > span:nth-child(2) { animation-delay: 0.15s; }
        .mp-spinner > span:nth-child(3) { animation-delay: 0.30s; }
    `],
})
export class MaskPreviewModalComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    private rtc = inject(RuntimeConfigService);
    private mediaItems = inject(MediaItemStore);
    private toast = inject(ToastService);

    protected data: MaskPreviewData = (this.overlay.topModal()?.data as MaskPreviewData) ?? {};

    protected mode = signal<'preview' | 'mask'>('preview');
    protected opacity = signal<number>(0.65);
    /** Truthy while the bake POST is in flight — disables the Bake button
     *  and swaps the label/icon for an inline spinner. */
    protected isBaking = signal<boolean>(false);

    /**
     * URLs that have successfully reported `load`. Each composite-preview
     * URL is unique per opacity step (the slider stamps it into the query
     * string), so a fresh URL means the loader should surface until the
     * backend produces and the browser paints the new variant.
     */
    private loadedUrls = signal<Set<string>>(new Set());

    protected isImageLoading = computed<boolean>(() => {
        const u = this.imageUrl();
        return !!u && !this.loadedUrls().has(u);
    });

    /** Bake is only meaningful when a raw mask file exists for the pair. */
    protected canBake = computed<boolean>(() =>
        !!this.data.pair?.metadata?.has_mask && !!this.data.datasetName,
    );

    protected bakeTooltip = computed<string>(() => {
        if (this.isBaking()) return 'Baking…';
        if (!this.canBake()) return 'No mask available to bake';
        return `Bake masked image at ${(this.opacity() * 100).toFixed(0)}% mix`;
    });

    ngOnInit(): void {
        if (this.data.mode) this.mode.set(this.data.mode);
    }

    protected imageUrl = computed<string | null>(() => {
        const pair = this.data.pair;
        const name = this.data.datasetName;
        if (!pair || !name) return null;

        if (this.mode() === 'mask') {
            if (!pair.metadata?.has_mask) return null;
            const dot = pair.media_file.lastIndexOf('.');
            const stem = dot > 0 ? pair.media_file.substring(0, dot) : pair.media_file;
            const maskPath = `masks/${stem}.png`;
            return `${this.rtc.mediaBaseUrl}/${encodeURIComponent(name)}/${encodeURIComponent(maskPath)}`;
        }

        return `${this.rtc.apiUrl}/datasets/${encodeURIComponent(name)}/masking/preview` +
               `?image_rel_path=${encodeURIComponent(pair.media_file)}&opacity=${this.opacity()}`;
    });

    protected onImageLoaded(event: Event): void {
        const target = event.target as HTMLImageElement | null;
        const url = target?.currentSrc || target?.src;
        if (!url) return;
        this.loadedUrls.update(s => {
            if (s.has(url)) return s;
            const next = new Set(s);
            next.add(url);
            return next;
        });
    }

    /**
     * Persist the current opacity as a baked masked image. Routes through
     * MediaItemStore.applyMask so the local row's `has_masked` flag flips
     * on success — that's what unlocks the workspace's masked-view toggle
     * for this pair.
     */
    protected async onBake(): Promise<void> {
        if (!this.canBake() || this.isBaking()) return;
        const name = this.data.datasetName!;
        const mediaFile = this.data.pair?.media_file;
        if (!mediaFile) return;

        this.isBaking.set(true);
        try {
            const result = await this.mediaItems.applyMask(name, mediaFile, this.opacity());
            if (result.ok) {
                this.toast.success('Masked image baked.');
                this.overlay.closeModal();
            }
        } finally {
            this.isBaking.set(false);
        }
    }
}
