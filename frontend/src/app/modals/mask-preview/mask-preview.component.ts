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
                    <img [src]="url" class="mp-image" alt=""/>
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
                <button class="btn cta" type="button" (click)="bakeMask()">
                    <app-ico name="Check" [size]="12"/> Bake mask
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
        }
    `],
})
export class MaskPreviewModalComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    private rtc = inject(RuntimeConfigService);

    protected data: MaskPreviewData = (this.overlay.topModal()?.data as MaskPreviewData) ?? {};

    protected mode = signal<'preview' | 'mask'>('preview');
    protected opacity = signal<number>(0.65);

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

    protected bakeMask(): void {
        // TODO(frontend): wire to the dataset mask-bake endpoint once the
        // workspace exposes a callback channel back to the details mask
        // sidebar. For now this closes the modal so the user can re-run
        // the mask-apply flow from the mass-mask modal.
        this.overlay.closeModal();
    }
}
