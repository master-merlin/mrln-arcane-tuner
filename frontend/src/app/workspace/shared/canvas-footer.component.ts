import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';
import { IcoComponent } from '../../icons/ico.component';

export interface CanvasMeta {
    file: string | null;
    res: string | null;
    ar: string | null;
    orientation: string | null;
    size: string | null;
    hpsLabel: string | null;
    hpsTone: 'success' | 'warning' | 'danger' | null;
    hasOverlay: boolean;
}

/**
 * Shared canvas footer (zoom group + centered metadata strip + optional
 * action group). Mounted at the bottom of the canvas in Details mode
 * AND Edit mode so they don't drift.
 */
@Component({
    selector: 'app-canvas-footer',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="canvas-footer">
            <div class="footer-side footer-side-left">
            <div class="zoom-group">
                <button type="button" class="icon-btn" title="Zoom out"
                        [disabled]="zoomOutDisabled()" (click)="onZoomOut()">
                    <app-ico name="ZoomOut" [size]="13"/>
                </button>
                <button type="button" class="mono zoom-val" title="Reset zoom to 100%"
                        (click)="onZoomReset()">{{ zoomPct() }}%</button>
                <button type="button" class="icon-btn" title="Zoom in"
                        [disabled]="zoomInDisabled()" (click)="onZoomIn()">
                    <app-ico name="ZoomIn" [size]="13"/>
                </button>
                <span class="footer-divider"></span>
                <button type="button" class="icon-btn"
                        [title]="fullscreen() ? 'Exit fullscreen' : 'Fullscreen'"
                        (click)="toggleFullscreen.emit()">
                    <app-ico [name]="fullscreen() ? 'Minimize' : 'Maximize'" [size]="12"/>
                </button>
            </div>
            </div>

            <div class="meta-strip">
                @if (meta(); as m) {
                    @if (m.file) {
                        <span class="meta-item meta-file" [title]="m.file">
                            <app-ico name="FileImage" [size]="12"/>
                            <span class="mono">{{ m.file }}</span>
                        </span>
                    }
                    @if (m.res) {
                        <span class="meta-item">
                            <app-ico name="Image" [size]="12"/>
                            <span class="mono">{{ m.res }}</span>
                        </span>
                    }
                    @if (m.ar) {
                        <span class="meta-item">
                            <span class="muted">AR</span>
                            <span class="mono">{{ m.ar }}</span>
                        </span>
                    }
                    @if (m.orientation) {
                        <span class="chip solid orientation">{{ m.orientation }}</span>
                    }
                    @if (m.size) {
                        <span class="meta-item">
                            <app-ico name="HardDrive" [size]="12"/>
                            <span class="mono">{{ m.size }}</span>
                        </span>
                    }
                    @if (m.hpsLabel && m.hpsTone) {
                        <span [class]="'tag ' + m.hpsTone">{{ m.hpsLabel }}</span>
                    }
                    @if (m.hasOverlay) {
                        @if (showOverlay() === null) {
                            <span class="tag violet" title="Adjustment overlay applied">
                                <app-ico name="Layers" [size]="11"/>
                                OVR
                            </span>
                        } @else {
                            <button type="button"
                                    class="tag violet ovr-toggle"
                                    [class.active]="showOverlay()"
                                    (click)="toggleOverlay.emit()"
                                    [title]="showOverlay() ? 'Hide overlay (showing edited version) — click to show original' : 'Show overlay (showing original) — click to show edited version'">
                                <app-ico name="Layers" [size]="11"/>
                                OVR
                            </button>
                        }
                    }
                }
            </div>

            <div class="action-group">
                <ng-content/>
            </div>
        </div>
    `,
    styles: [`
        :host { display: block; flex-shrink: 0; }
        .canvas-footer {
            height: 52px;
            padding: 0 18px;
            display: flex;
            align-items: center;
            gap: 12px;
            background: var(--color-surface-low);
            border-top: 1px solid var(--color-border-subtle);
        }
        /* Equal-width side regions keep the meta strip centered on the bar
           regardless of how many (if any) action buttons the right side has,
           so the strip doesn't jump between Details (with actions) and Edit. */
        .footer-side { flex: 1 1 0; display: flex; align-items: center; min-width: 0; }
        .footer-side-left { justify-content: flex-start; }
        .zoom-group {
            display: flex; align-items: center; gap: 2px;
            padding: 3px 4px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
        }
        .zoom-group .icon-btn { width: 24px; height: 24px; }
        .zoom-group .icon-btn:disabled { opacity: 0.45; cursor: default; }
        .zoom-val {
            font-size: 11.5px; padding: 0 6px; min-width: 46px; text-align: center;
            color: var(--color-text-muted);
            background: none; border: none; cursor: pointer;
            border-radius: var(--radius-theme-sm); transition: color 120ms, background 120ms;
        }
        .zoom-val:hover { color: var(--color-text-primary); background: var(--color-surface-high); }
        .footer-divider { width: 1px; height: 16px; background: var(--color-border-subtle); margin: 0 2px; }
        .meta-strip {
            flex: 0 1 auto; min-width: 0;
            display: flex; justify-content: center; align-items: center;
            gap: 22px; flex-wrap: wrap;
            font-size: 11.5px;
            color: var(--color-text-muted);
        }
        .meta-strip .meta-item { display: inline-flex; align-items: center; gap: 6px; }
        .meta-strip .meta-file .mono {
            max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        }
        .meta-strip .meta-item .muted { color: var(--color-text-subtle); }
        .meta-strip .chip.solid.orientation {
            font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em;
        }
        .action-group {
            flex: 1 1 0; min-width: 0;
            display: flex; align-items: center; justify-content: flex-end; gap: 6px;
        }
        .ovr-toggle {
            cursor: pointer;
            border: none;
            font: inherit;
            opacity: 0.55;
            transition: opacity 120ms;
        }
        .ovr-toggle:hover { opacity: 0.85; }
        .ovr-toggle.active { opacity: 1; }
    `],
})
export class CanvasFooterComponent {
    meta = input<CanvasMeta | null>(null);
    /** When non-null, the OVR pill renders as a clickable toggle reflecting
     *  this state (active when true). When null (default), it renders as
     *  the existing static indicator — Edit mode uses that variant since
     *  it IS the overlay editor. */
    showOverlay = input<boolean | null>(null);
    /** Fired when the user clicks the OVR toggle. Parent wires this to
     *  whatever owns the showOverlay signal (workspace's toggleOverlayView). */
    toggleOverlay = output<void>();

    // ── Zoom ─────────────────────────────────────────────────────────────
    /** Current zoom factor (1 = 100%). Owned by the host canvas so it can
     *  reset on image change; the footer is the stateless control surface. */
    zoom = input<number>(1);
    /** Emits the new clamped zoom factor on every +/-/reset action. */
    zoomChange = output<number>();

    private readonly MIN = 0.25;
    private readonly MAX = 5;
    private readonly STEP = 1.25;

    protected zoomPct = computed(() => Math.round(this.zoom() * 100));
    protected zoomInDisabled = computed(() => this.zoom() >= this.MAX - 1e-6);
    protected zoomOutDisabled = computed(() => this.zoom() <= this.MIN + 1e-6);

    protected onZoomIn(): void {
        this.zoomChange.emit(Math.min(this.MAX, +(this.zoom() * this.STEP).toFixed(4)));
    }
    protected onZoomOut(): void {
        this.zoomChange.emit(Math.max(this.MIN, +(this.zoom() / this.STEP).toFixed(4)));
    }
    protected onZoomReset(): void {
        this.zoomChange.emit(1);
    }

    // ── Fullscreen ───────────────────────────────────────────────────────
    /** Reflects whether the host canvas is currently fullscreen (drives the
     *  Maximize/Minimize icon + tooltip). */
    fullscreen = input<boolean>(false);
    /** Fired when the user clicks the fullscreen button; the host requests/
     *  exits fullscreen on whichever element wraps the canvas + this footer. */
    toggleFullscreen = output<void>();
}
