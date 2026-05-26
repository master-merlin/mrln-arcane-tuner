import { ChangeDetectionStrategy, Component, computed, inject, input, output } from '@angular/core';
import { IcoComponent } from '../../../../icons/ico.component';
import { OverlayStore } from '../../../../state/overlay.store';
import { RuntimeConfigService } from '../../../../services/runtime-config.service';
import { CanvasFooterComponent, CanvasMeta } from '../../../shared/canvas-footer.component';
import { PipelineEditorState } from '../pipeline-editor.state';

@Component({
    selector: 'app-edit-canvas',
    standalone: true,
    imports: [IcoComponent, CanvasFooterComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="stage hover-host">
            <!-- prev / next (hover-revealed) -->
            <button type="button" class="nav-btn left hover-show" (click)="prev.emit()" title="Previous">
                <app-ico name="ChevronLeft" [size]="18"/>
            </button>
            <button type="button" class="nav-btn right hover-show" (click)="next.emit()" title="Next">
                <app-ico name="ChevronRight" [size]="18"/>
            </button>

            <div class="image-stage">
                <img class="layer base"
                     [src]="sourceUrl()"
                     [alt]="mediaFile()"
                     loading="eager" decoding="sync"/>
                @if (overlayUrl(); as url) {
                    <img class="layer overlay"
                         [src]="url"
                         alt=""
                         aria-hidden="true"
                         loading="eager" decoding="sync"/>
                }
                <div class="file-label">
                    <app-ico name="Image" [size]="11"/>
                    <span class="mono">{{ mediaFile() }}</span>
                </div>
            </div>
        </div>

        <app-canvas-footer [meta]="meta()"/>
    `,
    styles: [`
        :host { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
        .stage {
            flex: 1; position: relative;
            display: flex; align-items: center; justify-content: center;
            padding: 14px 16px; min-height: 0;
            background: var(--color-base);
        }
        .image-stage {
            position: relative;
            max-width: 100%; max-height: 100%;
            display: inline-flex;
            border-radius: var(--radius-theme-lg);
            box-shadow: var(--shadow-lg, 0 8px 24px rgba(0,0,0,0.25));
            overflow: hidden;
        }
        .layer {
            display: block;
            max-width: 100%; max-height: 100%;
            object-fit: contain;
        }
        .layer.overlay {
            position: absolute; inset: 0;
            pointer-events: none;
        }
        .file-label {
            position: absolute; top: 14px; left: 14px;
            display: inline-flex; align-items: center; gap: 6px;
            padding: 3px 8px;
            font-family: var(--font-mono); font-size: 11px;
            background: oklch(0.10 0.01 265 / 0.7);
            color: var(--color-text-secondary);
            border-radius: 4px;
            backdrop-filter: blur(6px);
        }
        .nav-btn {
            position: absolute; top: 50%; transform: translateY(-50%);
            z-index: 5;
            width: 40px; height: 40px;
            border-radius: 999px;
            background: oklch(0.10 0.01 265 / 0.65);
            color: #fff;
            border: 1px solid oklch(0.95 0 0 / 0.10);
            backdrop-filter: blur(6px);
            box-shadow: 0 2px 10px oklch(0 0 0 / 0.4);
            cursor: pointer;
            display: flex; align-items: center; justify-content: center;
            opacity: 0; transition: opacity 120ms;
        }
        .nav-btn.left  { left: 18px; }
        .nav-btn.right { right: 18px; }
        .hover-host:hover .hover-show { opacity: 1; }
    `],
})
export class EditCanvasComponent {
    datasetName = input.required<string>();
    mediaFile = input.required<string>();
    hasOverlay = input<boolean>(false);

    prev = output<void>();
    next = output<void>();

    private overlay = inject(OverlayStore);
    private rtc = inject(RuntimeConfigService);
    private state = inject(PipelineEditorState);

    // Source = the dataset's media URL.
    protected sourceUrl = computed(() =>
        `${this.rtc.mediaBaseUrl}/${this.datasetName()}/${encodeURIComponent(this.mediaFile())}`,
    );

    // Overlay URL — prefers the live preview signal; falls back to saved overlay.
    protected overlayUrl = computed<string | null>(() => {
        const preview = this.state.previewOverlay();
        if (preview) {
            return `${this.rtc.mediaBaseUrl}/${preview.url}?h=${preview.hash}`;
        }
        // Fall back to the saved overlay if no preview yet.
        if (!this.hasOverlay()) return null;
        const id = `${this.datasetName()}/${this.mediaFile()}`;
        const ov = (this.overlay.entities() ?? []).find((o: any) => o.id === id);
        if (!ov?.overlay_file) return null;
        const hash = ov.hash ? `?h=${ov.hash}` : '';
        return `${this.rtc.mediaBaseUrl}/${ov.overlay_file}${hash}`;
    });

    protected meta = computed<CanvasMeta>(() => ({
        res: null, ar: null, orientation: null, size: null,
        hpsLabel: null, hpsTone: null, hasOverlay: this.hasOverlay(),
    }));
}
