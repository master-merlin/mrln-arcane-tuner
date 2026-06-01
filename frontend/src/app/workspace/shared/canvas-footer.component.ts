import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';
import { IcoComponent } from '../../icons/ico.component';

export interface CanvasMeta {
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
            <div class="zoom-group">
                <button type="button" class="icon-btn" title="Zoom out" disabled>
                    <app-ico name="ZoomOut" [size]="13"/>
                </button>
                <span class="mono zoom-val">100%</span>
                <button type="button" class="icon-btn" title="Zoom in" disabled>
                    <app-ico name="ZoomIn" [size]="13"/>
                </button>
                <span class="footer-divider"></span>
                <button type="button" class="icon-btn" title="Fullscreen" disabled>
                    <app-ico name="Maximize" [size]="12"/>
                </button>
            </div>

            <div class="meta-strip">
                @if (meta(); as m) {
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
        .zoom-group {
            display: flex; align-items: center; gap: 2px;
            padding: 3px 4px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
        }
        .zoom-group .icon-btn { width: 24px; height: 24px; }
        .zoom-group .icon-btn:disabled { opacity: 0.45; cursor: default; }
        .zoom-val { font-size: 11.5px; padding: 0 6px; min-width: 46px; text-align: center; color: var(--color-text-muted); }
        .footer-divider { width: 1px; height: 16px; background: var(--color-border-subtle); margin: 0 2px; }
        .meta-strip {
            flex: 1;
            display: flex; justify-content: center; align-items: center;
            gap: 22px; flex-wrap: wrap;
            font-size: 11.5px;
            color: var(--color-text-muted);
        }
        .meta-strip .meta-item { display: inline-flex; align-items: center; gap: 6px; }
        .meta-strip .meta-item .muted { color: var(--color-text-subtle); }
        .meta-strip .chip.solid.orientation {
            font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em;
        }
        .action-group {
            display: flex; align-items: center; gap: 6px; flex-shrink: 0;
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
}
