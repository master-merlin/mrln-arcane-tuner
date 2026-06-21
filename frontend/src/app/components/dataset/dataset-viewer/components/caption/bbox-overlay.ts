/**
 * bbox-overlay.ts
 *
 * Overlay component that draws bounding boxes over an image and supports
 * selection and rubber-band draw mode.
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

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export interface BboxItem {
    id: string;
    bbox: number[]; // [y_min, x_min, y_max, x_max] 0–1000
}

interface DrawState {
    startX: number; // image-relative X at pointerdown
    startY: number; // image-relative Y at pointerdown
    curX: number;
    curY: number;
    active: boolean;
    imgW: number;  // rendered image width captured at pointerdown
    imgH: number;  // rendered image height captured at pointerdown
}

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
                ></div>
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
    // TODO (v2): emit on drag-resize; declared now for stable public contract
    readonly boxChanged = output<{ id: string; bbox: number[] }>();

    @ViewChild('imgEl') private imgEl?: ElementRef<HTMLImageElement>;

    // Internal draw state
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
     * Compute CSS style for a normalized bbox div.
     * Falls back to covering the full area if the container rect is unavailable
     * (e.g. in jsdom tests without layout).
     */
    protected boxStyle(box: BboxItem): string {
        const imgRect = this.imgEl?.nativeElement?.getBoundingClientRect();
        const W = imgRect?.width ?? 0;
        const H = imgRect?.height ?? 0;
        if (W === 0 || H === 0) {
            // Fallback: render as percentages so boxes are visually present even
            // without a real layout engine.
            const [y_min, x_min, y_max, x_max] = box.bbox;
            return [
                `left:${(x_min / BBOX_MAX) * 100}%`,
                `top:${(y_min / BBOX_MAX) * 100}%`,
                `width:${((x_max - x_min) / BBOX_MAX) * 100}%`,
                `height:${((y_max - y_min) / BBOX_MAX) * 100}%`,
            ].join(';');
        }
        const rect = normToPx(box.bbox, W, H);
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

    protected onPointerDown(e: PointerEvent): void {
        if (!this.drawEnabled()) return;
        const offset = this.pointerOffset(e);
        if (!offset) return;
        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
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
        const d = this.draw();
        if (!d.active) return;
        const offset = this.pointerOffset(e);
        if (!offset) return;
        this.draw.set({ ...d, curX: offset.x, curY: offset.y });
    }

    protected onPointerUp(e: PointerEvent): void {
        const d = this.draw();
        if (!d.active) return;
        const offset = this.pointerOffset(e);
        if (offset) {
            this.finalizeDraw(d, offset.x, offset.y);
        }
        this.draw.set({ ...d, active: false });
    }

    protected onPointerLeave(e: PointerEvent): void {
        const d = this.draw();
        if (!d.active) return;
        // Cancel rubber-band; user released outside the container
        this.draw.set({ ...d, active: false });
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
        this.boxSelected.emit(id);
    }

}
