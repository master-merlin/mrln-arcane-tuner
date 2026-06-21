/**
 * bbox-overlay.ts
 *
 * Overlay component that draws bounding boxes over an image and supports
 * selection, rubber-band draw mode, MOVE (drag interior), and RESIZE
 * (drag corner handles on the selected box).
 *
 * Coordinate convention (mirrors backend ideogram4 schema):
 *   All bboxes are y-first, normalized 0–1000:
 *   [y_min, x_min, y_max, x_max]
 */

import {
    ChangeDetectionStrategy,
    Component,
    ElementRef,
    ViewChild,
    input,
    output,
    signal,
} from '@angular/core';

import { BBOX_MAX } from './ideogram-format';

// ---------------------------------------------------------------------------
// Pure coordinate helpers (exported for unit tests)
// ---------------------------------------------------------------------------

/**
 * Convert a displayed-pixel drag rect to y-first normalized [0, BBOX_MAX] coords.
 *
 * @param pxLeft   Left pixel coordinate (relative to the image's left edge)
 * @param pxTop    Top pixel coordinate (relative to the image's top edge)
 * @param pxRight  Right pixel coordinate
 * @param pxBottom Bottom pixel coordinate
 * @param imgW     Rendered image width in pixels
 * @param imgH     Rendered image height in pixels
 * @returns [y_min, x_min, y_max, x_max] clamped to [0, BBOX_MAX], integers
 */
export function pxToNorm(
    pxLeft: number,
    pxTop: number,
    pxRight: number,
    pxBottom: number,
    imgW: number,
    imgH: number,
): number[] {
    const clamp = (v: number) => Math.max(0, Math.min(BBOX_MAX, Math.round(v)));
    const x_min = clamp((pxLeft / imgW) * BBOX_MAX);
    const x_max = clamp((pxRight / imgW) * BBOX_MAX);
    const y_min = clamp((pxTop / imgH) * BBOX_MAX);
    const y_max = clamp((pxBottom / imgH) * BBOX_MAX);
    return [y_min, x_min, y_max, x_max];
}

/**
 * Convert a y-first normalized bbox [y_min, x_min, y_max, x_max] (0–BBOX_MAX) to
 * a pixel rect expressed as { left, top, width, height } suitable for CSS positioning.
 *
 * @param bbox  [y_min, x_min, y_max, x_max] normalized 0–BBOX_MAX
 * @param imgW  Rendered image width in pixels
 * @param imgH  Rendered image height in pixels
 */
export function normToPx(
    bbox: number[],
    imgW: number,
    imgH: number,
): { left: number; top: number; width: number; height: number } {
    const [y_min, x_min, y_max, x_max] = bbox;
    const left = (x_min / BBOX_MAX) * imgW;
    const top = (y_min / BBOX_MAX) * imgH;
    const right = (x_max / BBOX_MAX) * imgW;
    const bottom = (y_max / BBOX_MAX) * imgH;
    return {
        left,
        top,
        width: right - left,
        height: bottom - top,
    };
}

/**
 * Translate a y-first bbox by (dyNorm, dxNorm) in normalized units.
 * The box retains its original width/height and is clamped so it stays within [0, BBOX_MAX].
 *
 * @param bbox    [y_min, x_min, y_max, x_max] normalized 0–BBOX_MAX
 * @param dyNorm  Delta in y (normalized), positive = down
 * @param dxNorm  Delta in x (normalized), positive = right
 * @returns new [y_min, x_min, y_max, x_max] clamped so both corners stay in [0, BBOX_MAX]
 */
export function movedBbox(bbox: number[], dyNorm: number, dxNorm: number): number[] {
    const [y_min, x_min, y_max, x_max] = bbox;
    const h = y_max - y_min;
    const w = x_max - x_min;

    // Translate then clamp the leading edge and adjust the trailing edge accordingly
    let ny_min = Math.max(0, Math.min(BBOX_MAX - h, y_min + dyNorm));
    let nx_min = Math.max(0, Math.min(BBOX_MAX - w, x_min + dxNorm));

    // Round to integers (same convention as pxToNorm)
    ny_min = Math.round(ny_min);
    nx_min = Math.round(nx_min);

    return [ny_min, nx_min, Math.min(BBOX_MAX, ny_min + h), Math.min(BBOX_MAX, nx_min + w)];
}

/**
 * Resize a y-first bbox by dragging one corner.
 * The opposite corner stays fixed; the dragged corner moves to (yNorm, xNorm).
 * Output is sorted so y_min < y_max and x_min < x_max (cross-over flip handled),
 * and all values are clamped to [0, BBOX_MAX].
 *
 * @param bbox    original [y_min, x_min, y_max, x_max]
 * @param corner  which corner is being dragged: 'tl'|'tr'|'bl'|'br'
 * @param yNorm   new y of the dragged corner (normalized 0–BBOX_MAX)
 * @param xNorm   new x of the dragged corner (normalized 0–BBOX_MAX)
 * @returns new [y_min, x_min, y_max, x_max] sorted and clamped
 */
export function resizedBbox(
    bbox: number[],
    corner: 'tl' | 'tr' | 'bl' | 'br',
    yNorm: number,
    xNorm: number,
): number[] {
    const clamp = (v: number) => Math.max(0, Math.min(BBOX_MAX, Math.round(v)));
    const cy = clamp(yNorm);
    const cx = clamp(xNorm);

    const [y_min, x_min, y_max, x_max] = bbox;

    // Fixed corner depends on which handle is being dragged
    let ay: number, ax: number, by: number, bx: number;
    switch (corner) {
        case 'tl': { ay = cy; ax = cx; by = y_max; bx = x_max; break; }
        case 'tr': { ay = cy; ax = x_min; by = y_max; bx = cx; break; }
        case 'bl': { ay = y_min; ax = cx; by = cy; bx = x_max; break; }
        case 'br': { ay = y_min; ax = x_min; by = cy; bx = cx; break; }
    }

    // Sort so y_min < y_max and x_min < x_max (cross-over flip)
    return [Math.min(ay, by), Math.min(ax, bx), Math.max(ay, by), Math.max(ax, bx)];
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export interface BboxItem {
    id: string;
    bbox: number[]; // [y_min, x_min, y_max, x_max] 0–1000
}

/** What kind of drag is in progress. */
type DragMode =
    | { kind: 'none' }
    | { kind: 'draw' }
    | { kind: 'move'; id: string; origBbox: number[]; startY: number; startX: number; imgW: number; imgH: number }
    | { kind: 'resize'; id: string; origBbox: number[]; corner: 'tl' | 'tr' | 'bl' | 'br'; imgW: number; imgH: number };

interface DrawState {
    startX: number; // image-relative X at pointerdown
    startY: number; // image-relative Y at pointerdown
    curX: number;
    curY: number;
    active: boolean;
    imgW: number;  // rendered image width captured at pointerdown
    imgH: number;  // rendered image height captured at pointerdown
}

/** Size of corner resize handles in pixels (visual + hit area). */
const HANDLE_PX = 8;

@Component({
    selector: 'app-bbox-overlay',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    styles: [`
        :host { display: block; position: relative; }
        .bbox-box {
            position: absolute;
            border: 2px solid rgba(99, 179, 237, 0.8);
            box-sizing: border-box;
            cursor: pointer;
            border-radius: 2px;
            pointer-events: auto;
            transition: border-color 0.1s;
        }
        .bbox-box.bbox-selected {
            border-color: #f6e05e;
            border-width: 2px;
            box-shadow: 0 0 0 1px rgba(246, 224, 94, 0.4);
            cursor: move;
        }
        .bbox-box:hover {
            border-color: rgba(144, 205, 244, 1);
        }
        .bbox-rubber {
            position: absolute;
            border: 1.5px dashed rgba(99, 179, 237, 0.9);
            box-sizing: border-box;
            pointer-events: none;
            background: rgba(99, 179, 237, 0.08);
        }
        .bbox-image {
            display: block;
            width: 100%;
            height: 100%;
            object-fit: contain;
            pointer-events: none;
            user-select: none;
        }
        .bbox-container {
            position: relative;
            display: inline-block;
            width: 100%;
        }
        .bbox-handle {
            position: absolute;
            width: ${HANDLE_PX}px;
            height: ${HANDLE_PX}px;
            background: #f6e05e;
            border: 1px solid rgba(0,0,0,0.4);
            box-sizing: border-box;
            border-radius: 2px;
            pointer-events: auto;
            z-index: 10;
        }
        .bbox-handle-tl { top: -${HANDLE_PX / 2}px; left: -${HANDLE_PX / 2}px; cursor: nwse-resize; }
        .bbox-handle-tr { top: -${HANDLE_PX / 2}px; right: -${HANDLE_PX / 2}px; cursor: nesw-resize; }
        .bbox-handle-bl { bottom: -${HANDLE_PX / 2}px; left: -${HANDLE_PX / 2}px; cursor: nesw-resize; }
        .bbox-handle-br { bottom: -${HANDLE_PX / 2}px; right: -${HANDLE_PX / 2}px; cursor: nwse-resize; }
    `],
    template: `
        <div
            class="bbox-container"
            data-testid="bbox-container"
            [style.cursor]="drawEnabled() ? 'crosshair' : 'default'"
            (pointerdown)="onPointerDown($event)"
            (pointermove)="onPointerMove($event)"
            (pointerup)="onPointerUp($event)"
            (pointerleave)="onPointerLeave($event)"
        >
            @if (imageUrl()) {
                <img
                    #imgEl
                    class="bbox-image"
                    [src]="imageUrl()"
                    alt=""
                    draggable="false"
                />
            }

            @for (box of boxes(); track box.id) {
                <div
                    class="bbox-box"
                    [class.bbox-selected]="box.id === selectedId()"
                    [style]="boxStyle(box)"
                    data-testid="bbox-box"
                    [attr.data-bbox-id]="box.id"
                    (click)="onBoxClick($event, box.id)"
                    (pointerdown)="onBoxPointerDown($event, box.id, box.bbox)"
                >
                    @if (box.id === selectedId()) {
                        <div
                            class="bbox-handle bbox-handle-tl"
                            data-testid="bbox-handle-tl"
                            (pointerdown)="onHandlePointerDown($event, box.id, box.bbox, 'tl')"
                        ></div>
                        <div
                            class="bbox-handle bbox-handle-tr"
                            data-testid="bbox-handle-tr"
                            (pointerdown)="onHandlePointerDown($event, box.id, box.bbox, 'tr')"
                        ></div>
                        <div
                            class="bbox-handle bbox-handle-bl"
                            data-testid="bbox-handle-bl"
                            (pointerdown)="onHandlePointerDown($event, box.id, box.bbox, 'bl')"
                        ></div>
                        <div
                            class="bbox-handle bbox-handle-br"
                            data-testid="bbox-handle-br"
                            (pointerdown)="onHandlePointerDown($event, box.id, box.bbox, 'br')"
                        ></div>
                    }
                </div>
            }

            @if (draw().active) {
                <div
                    class="bbox-rubber"
                    [style]="rubberStyle()"
                    data-testid="bbox-rubber"
                ></div>
            }
        </div>
    `,
})
export class BboxOverlayComponent {
    // Inputs
    readonly imageUrl = input<string>('');
    readonly boxes = input<BboxItem[]>([]);
    readonly selectedId = input<string | null>(null);
    readonly drawEnabled = input<boolean>(false);

    // Outputs
    readonly boxAdded = output<number[]>();
    readonly boxSelected = output<string>();
    readonly boxChanged = output<{ id: string; bbox: number[] }>();

    @ViewChild('imgEl') private imgEl?: ElementRef<HTMLImageElement>;

    // Internal draw state (rubber-band)
    protected readonly draw = signal<DrawState>({
        startX: 0,
        startY: 0,
        curX: 0,
        curY: 0,
        active: false,
        imgW: 0,
        imgH: 0,
    });

    /**
     * Live-preview override for the box currently being dragged.
     * Only one box can be dragged at a time, so a single slot is sufficient.
     * Set during pointermove (move/resize); cleared on pointerup or drag-abort.
     * boxStyle() prefers this over the input bbox when the id matches.
     * NEVER mutates the input boxes() array.
     */
    private readonly _liveBbox = signal<{ id: string; bbox: number[] } | null>(null);

    // Current drag mode (move / resize / draw / none)
    private dragMode: DragMode = { kind: 'none' };

    // Track whether a meaningful drag happened (to distinguish click-to-select from drag-to-move)
    private dragMoved = false;

    /**
     * Compute CSS style for a normalized bbox div.
     * Falls back to covering the full area if the container rect is unavailable
     * (e.g. in jsdom tests without layout).
     *
     * During a drag, prefers the _liveBbox override for the dragged box so the
     * visual updates in real-time without touching the input boxes() array.
     */
    protected boxStyle(box: BboxItem): string {
        // Prefer the internal live-drag override over the (immutable) input bbox.
        const live = this._liveBbox();
        const bbox = (live && live.id === box.id) ? live.bbox : box.bbox;

        const imgRect = this.imgEl?.nativeElement?.getBoundingClientRect();
        const W = imgRect?.width ?? 0;
        const H = imgRect?.height ?? 0;
        if (W === 0 || H === 0) {
            // Fallback: render as percentages so boxes are visually present even
            // without a real layout engine.
            const [y_min, x_min, y_max, x_max] = bbox;
            return [
                `left:${(x_min / BBOX_MAX) * 100}%`,
                `top:${(y_min / BBOX_MAX) * 100}%`,
                `width:${((x_max - x_min) / BBOX_MAX) * 100}%`,
                `height:${((y_max - y_min) / BBOX_MAX) * 100}%`,
            ].join(';');
        }
        const rect = normToPx(bbox, W, H);
        return [
            `left:${rect.left}px`,
            `top:${rect.top}px`,
            `width:${rect.width}px`,
            `height:${rect.height}px`,
        ].join(';');
    }

    /** CSS style for the rubber-band rectangle during draw. */
    protected rubberStyle(): string {
        const d = this.draw();
        const left = Math.min(d.startX, d.curX);
        const top = Math.min(d.startY, d.curY);
        const width = Math.abs(d.curX - d.startX);
        const height = Math.abs(d.curY - d.startY);
        return `left:${left}px;top:${top}px;width:${width}px;height:${height}px`;
    }

    /**
     * Pointer relative to the image element's top-left.
     * Using the image rect keeps pointer coords in the same space as boxStyle(),
     * which also measures from the image element — avoiding any offset under
     * letterboxing (object-fit:contain with aspect-ratio mismatch).
     *
     * Falls back to the container element when the image is not yet rendered
     * (e.g. no imageUrl set), so the guard in finalizeDraw (W===0||H===0) still
     * fires correctly in jsdom tests.
     */
    private pointerOffset(
        e: PointerEvent,
    ): { x: number; y: number; imgW: number; imgH: number } | null {
        const imgEl = this.imgEl?.nativeElement;
        const ref: Element = imgEl ?? (e.currentTarget as HTMLElement);
        if (!ref) return null;
        const rect = ref.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
            imgW: rect.width,
            imgH: rect.height,
        };
    }

    // ---------------------------------------------------------------------------
    // Box + handle pointer events (move / resize)
    // ---------------------------------------------------------------------------

    /**
     * Called when the user presses a pointer button down on an existing box interior.
     * Begins a potential move drag.
     */
    protected onBoxPointerDown(e: PointerEvent, id: string, bbox: number[]): void {
        // Prevent the container's onPointerDown from also starting a draw
        e.stopPropagation();
        const offset = this.pointerOffset(e);
        if (!offset) return;

        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        this.dragMoved = false;
        this.dragMode = {
            kind: 'move',
            id,
            origBbox: [...bbox],
            startY: offset.y,
            startX: offset.x,
            imgW: offset.imgW,
            imgH: offset.imgH,
        };
    }

    /**
     * Called when the user presses a pointer button down on a corner resize handle.
     * Begins a resize drag.
     */
    protected onHandlePointerDown(
        e: PointerEvent,
        id: string,
        bbox: number[],
        corner: 'tl' | 'tr' | 'bl' | 'br',
    ): void {
        e.stopPropagation();
        const offset = this.pointerOffset(e);
        if (!offset) return;

        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        this.dragMoved = false;
        this.dragMode = {
            kind: 'resize',
            id,
            origBbox: [...bbox],
            corner,
            imgW: offset.imgW,
            imgH: offset.imgH,
        };
    }

    // ---------------------------------------------------------------------------
    // Container pointer events (draw + move/resize forwarding)
    // ---------------------------------------------------------------------------

    protected onPointerDown(e: PointerEvent): void {
        // Move/resize start from onBoxPointerDown / onHandlePointerDown — don't
        // accidentally start a draw when the mode is already set.
        if (this.dragMode.kind !== 'none') return;

        if (!this.drawEnabled()) return;
        const offset = this.pointerOffset(e);
        if (!offset) return;
        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        this.dragMode = { kind: 'draw' };
        this.draw.set({
            startX: offset.x,
            startY: offset.y,
            curX: offset.x,
            curY: offset.y,
            active: true,
            imgW: offset.imgW,
            imgH: offset.imgH,
        });
    }

    protected onPointerMove(e: PointerEvent): void {
        const mode = this.dragMode;

        if (mode.kind === 'draw') {
            const d = this.draw();
            if (!d.active) return;
            const offset = this.pointerOffset(e);
            if (!offset) return;
            this.draw.set({ ...d, curX: offset.x, curY: offset.y });
            return;
        }

        if (mode.kind === 'move') {
            const offset = this.pointerOffset(e);
            if (!offset || mode.imgW === 0 || mode.imgH === 0) return;
            this.dragMoved = true;
            const dyNorm = ((offset.y - mode.startY) / mode.imgH) * BBOX_MAX;
            const dxNorm = ((offset.x - mode.startX) / mode.imgW) * BBOX_MAX;
            const newBbox = movedBbox(mode.origBbox, dyNorm, dxNorm);
            // Update the live bbox on the box so boxStyle re-renders it
            this._updateLiveBbox(mode.id, newBbox);
            return;
        }

        if (mode.kind === 'resize') {
            const offset = this.pointerOffset(e);
            if (!offset || mode.imgW === 0 || mode.imgH === 0) return;
            this.dragMoved = true;
            const yNorm = (offset.y / mode.imgH) * BBOX_MAX;
            const xNorm = (offset.x / mode.imgW) * BBOX_MAX;
            const newBbox = resizedBbox(mode.origBbox, mode.corner, yNorm, xNorm);
            this._updateLiveBbox(mode.id, newBbox);
            return;
        }
    }

    protected onPointerUp(e: PointerEvent): void {
        const mode = this.dragMode;

        if (mode.kind === 'draw') {
            const d = this.draw();
            if (d.active) {
                const offset = this.pointerOffset(e);
                if (offset) {
                    this.finalizeDraw(d, offset.x, offset.y);
                }
                this.draw.set({ ...d, active: false });
            }
            this.dragMode = { kind: 'none' };
            return;
        }

        if (mode.kind === 'move') {
            if (this.dragMoved) {
                const offset = this.pointerOffset(e);
                if (offset && mode.imgW > 0 && mode.imgH > 0) {
                    const dyNorm = ((offset.y - mode.startY) / mode.imgH) * BBOX_MAX;
                    const dxNorm = ((offset.x - mode.startX) / mode.imgW) * BBOX_MAX;
                    const newBbox = movedBbox(mode.origBbox, dyNorm, dxNorm);
                    this.boxChanged.emit({ id: mode.id, bbox: newBbox });
                }
            }
            this._liveBbox.set(null); // clear override after emit (or if aborted by imgW/H guard)
            this.dragMode = { kind: 'none' };
            this.dragMoved = false;
            return;
        }

        if (mode.kind === 'resize') {
            if (this.dragMoved) {
                const offset = this.pointerOffset(e);
                if (offset && mode.imgW > 0 && mode.imgH > 0) {
                    const yNorm = (offset.y / mode.imgH) * BBOX_MAX;
                    const xNorm = (offset.x / mode.imgW) * BBOX_MAX;
                    const newBbox = resizedBbox(mode.origBbox, mode.corner, yNorm, xNorm);
                    this.boxChanged.emit({ id: mode.id, bbox: newBbox });
                }
            }
            this._liveBbox.set(null); // clear override after emit (or if aborted by imgW/H guard)
            this.dragMode = { kind: 'none' };
            this.dragMoved = false;
            return;
        }
    }

    protected onPointerLeave(e: PointerEvent): void {
        const mode = this.dragMode;
        if (mode.kind === 'draw') {
            const d = this.draw();
            if (d.active) {
                this.draw.set({ ...d, active: false });
            }
            this.dragMode = { kind: 'none' };
            this.dragMoved = false;
        }
        // For move/resize, pointer capture keeps events flowing even outside the
        // container, so we do NOT cancel on pointerleave — the drag continues.
    }

    private finalizeDraw(
        d: DrawState,
        endX: number,
        endY: number,
    ): void {
        if (!this.drawEnabled()) return;
        const pxLeft = Math.min(d.startX, endX);
        const pxTop = Math.min(d.startY, endY);
        const pxRight = Math.max(d.startX, endX);
        const pxBottom = Math.max(d.startY, endY);

        // Use the image dimensions captured at pointerdown (same origin as boxStyle)
        const W = d.imgW;
        const H = d.imgH;

        if (W === 0 || H === 0 || (pxRight - pxLeft < 2 && pxBottom - pxTop < 2)) {
            // Degenerate — ignore
            return;
        }

        const norm = pxToNorm(pxLeft, pxTop, pxRight, pxBottom, W, H);
        this.boxAdded.emit(norm);
    }

    protected onBoxClick(e: MouseEvent, id: string): void {
        e.stopPropagation();
        // Only emit select if this was a pure click (no drag movement)
        if (!this.dragMoved) {
            this.boxSelected.emit(id);
        }
    }

    /**
     * Store a live bbox override for the box currently being dragged.
     * Setting the _liveBbox signal is sufficient to trigger OnPush change
     * detection; boxStyle() reads _liveBbox() and prefers it over the input
     * bbox when the id matches. The input boxes() array is NEVER mutated.
     */
    private _updateLiveBbox(id: string, newBbox: number[]): void {
        this._liveBbox.set({ id, bbox: newBbox });
    }
}
