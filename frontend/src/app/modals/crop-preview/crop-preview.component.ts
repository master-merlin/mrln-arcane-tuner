import {
    ChangeDetectionStrategy,
    Component,
    OnInit,
    computed,
    inject,
    signal,
} from '@angular/core';
import { IcoComponent } from '../../icons/ico.component';
import { OverlayStore } from '../../state/overlay.store';
import { RuntimeConfigService } from '../../services/runtime-config.service';

interface CropPreviewData {
    datasetId?: string;
    datasetName?: string;
    /** Path of the source image (relative to dataset root). */
    path?: string;
    /** Source dimensions. */
    width?: number;
    height?: number;
    /** Target bucket dimensions. */
    target_width?: number;
    target_height?: number;
}

const RATIOS: ReadonlyArray<readonly [string, number]> = [
    ['16:9', 16 / 9],
    ['4:3', 4 / 3],
    ['1:1', 1],
    ['9:16', 9 / 16],
    ['3:2', 3 / 2],
];

/**
 * Crop preview modal — visualizes a centered crop window on the source
 * image with aspect ratio toggles and rule-of-thirds guides.
 *
 * Ports the workflow from the orphan
 * [viewer-crop-preview-modal](../../components/dataset/dataset-viewer/components/viewer-crop-preview-modal.ts)
 * and the design shell from `modals-more.jsx → CropPreviewModal`.
 *
 * Scope-limited port: the orphan modal supports interactive drag-to-move
 * and corner-handle resize of the crop window. That is **not yet** in
 * this PR — the crop here is centered, sized by aspect ratio, and the
 * Save button is a placeholder. Editing/persisting crops will land with
 * the EditMode buildout in the orphan-cleanup PR.
 *
 * TODO(frontend): wire interactive crop edit + save-crop endpoint.
 */
@Component({
    selector: 'app-modal-crop-preview',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head cp-head">
            <div>
                <div class="eyebrow warning">CROP PREVIEW</div>
                <div class="modal-title mono">{{ data.path ?? '—' }}</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body cp-body">
            <div class="cp-toggles">
                @for (r of ratios; track r[0]) {
                    <button class="chip"
                            type="button"
                            [class.solid]="aspect() === r[0]"
                            (click)="aspect.set(r[0])">{{ r[0] }}</button>
                }
                <span class="cp-divider"></span>
                <button class="chip" type="button" (click)="autoFit()"><app-ico name="Sparkles" [size]="10"/> Auto-fit</button>
                <button class="chip" type="button" (click)="reset()"><app-ico name="RefreshCw" [size]="10"/> Reset</button>
            </div>

            <div class="cp-stage">
                <div class="cp-canvas">
                    @if (imageUrl(); as url) {
                        <img [src]="url" class="cp-image" alt=""/>
                    }
                    <div class="cp-dim"></div>
                    <div class="cp-window" [style.width.%]="cropPct().width" [style.height.%]="cropPct().height"
                         [style.left.%]="cropPct().left" [style.top.%]="cropPct().top">
                        <span class="cp-handle tl"></span>
                        <span class="cp-handle tr"></span>
                        <span class="cp-handle bl"></span>
                        <span class="cp-handle br"></span>
                        <div class="cp-thirds">
                            <span class="cp-line v" style="left:33.33%"></span>
                            <span class="cp-line v" style="left:66.66%"></span>
                            <span class="cp-line h" style="top:33.33%"></span>
                            <span class="cp-line h" style="top:66.66%"></span>
                        </div>
                        <div class="cp-badge">{{ aspect() }}</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="modal-foot cp-foot">
            <div class="cp-dims">
                <span class="muted">Source </span>
                <span class="mono">{{ data.width ?? '—' }}×{{ data.height ?? '—' }}</span>
                <app-ico name="ChevronRight" [size]="11" class="cp-arrow"/>
                <span class="muted">Output </span>
                <span class="mono warning">{{ outputSize().w }}×{{ outputSize().h }}</span>
            </div>
            <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
            <button class="btn cta warning" type="button" (click)="saveCrop()">
                <app-ico name="Check" [size]="12"/> Save crop
            </button>
        </div>
    `,
    styles: [`
        .modal-title { font-size: 13px; font-weight: 600; margin-top: 2px; }
        .eyebrow {
            font-size: 10px; font-weight: 700; letter-spacing: 0.14em;
            text-transform: uppercase; color: var(--color-text-muted);
        }
        .eyebrow.warning { color: var(--color-warning); }
        .muted { color: var(--color-text-muted); }
        .warning { color: var(--color-warning); }

        .cp-body { padding: 14px 18px; display: flex; flex-direction: column; gap: 14px; }
        .cp-toggles { display: flex; gap: 6px; justify-content: center; align-items: center; }
        .cp-divider { width: 1px; background: var(--color-border-subtle); margin: 0 4px; align-self: stretch; }
        .chip {
            display: inline-flex; align-items: center; gap: 4px;
            padding: 4px 11px; border-radius: 999px;
            font-size: 11px; font-weight: 600;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            color: var(--color-text-secondary);
            cursor: pointer;
        }
        .chip.solid {
            background: var(--color-warning);
            color: oklch(0.18 0.05 75);
            border-color: var(--color-warning);
        }

        .cp-stage {
            position: relative;
            border-radius: var(--radius-theme-xl);
            overflow: hidden;
            background: var(--color-base);
            border: 1px solid var(--color-border-subtle);
            display: flex; align-items: center; justify-content: center;
            padding: 24px;
            min-height: 360px;
        }
        .cp-canvas {
            position: relative;
            width: 560px; max-width: 100%;
            aspect-ratio: 16/9;
            border-radius: var(--radius-theme-md);
            overflow: hidden;
            background: var(--color-surface-low);
        }
        .cp-image {
            position: absolute; inset: 0;
            width: 100%; height: 100%;
            object-fit: cover;
        }
        .cp-dim { position: absolute; inset: 0; background: oklch(0.08 0.01 265 / 0.55); }
        .cp-window {
            position: absolute;
            border: 1.5px dashed var(--color-warning);
            box-shadow:
                0 0 0 9999px oklch(0.08 0.01 265 / 0.55),
                0 0 24px oklch(0.75 0.16 75 / 0.4);
        }
        .cp-handle {
            position: absolute;
            width: 10px; height: 10px;
            background: var(--color-warning);
            border-radius: 2px;
        }
        .cp-handle.tl { top: -5px; left: -5px; }
        .cp-handle.tr { top: -5px; right: -5px; }
        .cp-handle.bl { bottom: -5px; left: -5px; }
        .cp-handle.br { bottom: -5px; right: -5px; }
        .cp-thirds { position: absolute; inset: 0; pointer-events: none; opacity: 0.5; }
        .cp-line {
            position: absolute;
            background: oklch(0.75 0.16 75 / 0.5);
        }
        .cp-line.v { top: 0; bottom: 0; width: 1px; }
        .cp-line.h { left: 0; right: 0; height: 1px; }
        .cp-badge {
            position: absolute; top: 6px; left: 6px;
            background: var(--color-warning);
            color: oklch(0.18 0.05 75);
            font-weight: 800;
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 10px;
            letter-spacing: 0.10em;
            text-transform: uppercase;
        }

        .cp-foot { display: flex; align-items: center; gap: 10px; }
        .cp-dims {
            display: flex; align-items: center; gap: 6px;
            margin-right: auto;
            font-size: 11.5px;
        }
        .cp-arrow { color: var(--color-text-subtle); }
        .btn.cta.warning {
            display: inline-flex; align-items: center; gap: 8px;
            background: var(--color-warning);
            color: oklch(0.18 0.05 75);
            font-weight: 800;
            padding: 9px 16px;
            border-radius: var(--radius-theme-lg);
        }
    `],
})
export class CropPreviewModalComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    private rtc = inject(RuntimeConfigService);

    protected data: CropPreviewData = (this.overlay.topModal()?.data as CropPreviewData) ?? {};

    protected readonly ratios = RATIOS;
    protected aspect = signal<string>('16:9');

    ngOnInit(): void {
        const sw = this.data.width ?? 0;
        const sh = this.data.height ?? 0;
        if (sw > 0 && sh > 0) {
            const ar = sw / sh;
            // Pick the closest match from the ratio palette.
            let best: string = '16:9';
            let bestDiff = Infinity;
            for (const [label, val] of RATIOS) {
                const diff = Math.abs(ar - val);
                if (diff < bestDiff) { bestDiff = diff; best = label; }
            }
            this.aspect.set(best);
        }
    }

    protected imageUrl = computed<string | null>(() => {
        const name = this.data.datasetName;
        const path = this.data.path;
        if (!name || !path) return null;
        return `${this.rtc.mediaBaseUrl}/${encodeURIComponent(name)}/${encodeURIComponent(path)}`;
    });

    /** Centered crop window sized by aspect ratio, projected over a 16/9 canvas. */
    protected cropPct = computed(() => {
        const ratio = RATIOS.find(r => r[0] === this.aspect())?.[1] ?? 16 / 9;
        // Canvas is fixed 16:9. Compute window size so it has `ratio` AR and fits.
        const canvasAR = 16 / 9;
        let w = 0.85;
        let h = (w * canvasAR) / ratio; // height in canvas-aspect units
        if (h > 0.85) {
            h = 0.85;
            w = (h * ratio) / canvasAR;
        }
        const left = (1 - w) / 2;
        const top = (1 - h) / 2;
        return { width: w * 100, height: h * 100, left: left * 100, top: top * 100 };
    });

    protected outputSize = computed<{ w: number; h: number }>(() => {
        const sw = this.data.target_width ?? this.data.width ?? 0;
        const sh = this.data.target_height ?? this.data.height ?? 0;
        return { w: sw, h: sh };
    });

    protected autoFit(): void {
        const sw = this.data.width ?? 0;
        const sh = this.data.height ?? 0;
        if (sw > 0 && sh > 0) {
            const ar = sw / sh;
            let best: string = '16:9';
            let bestDiff = Infinity;
            for (const [label, val] of RATIOS) {
                const diff = Math.abs(ar - val);
                if (diff < bestDiff) { bestDiff = diff; best = label; }
            }
            this.aspect.set(best);
        }
    }

    protected reset(): void {
        this.aspect.set('16:9');
    }

    protected saveCrop(): void {
        // TODO(frontend): wire to a backend save-crop endpoint. The
        // existing orphan modal supported interactive resize + persistence,
        // but the persistence path is bundled with EditMode and will land
        // alongside the editor-body extraction in the cleanup PR.
        this.overlay.closeModal();
    }
}
