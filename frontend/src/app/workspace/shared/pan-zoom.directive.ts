import {
    Directive,
    ElementRef,
    HostListener,
    effect,
    inject,
    input,
    model,
    signal,
} from '@angular/core';

/** Zoom bounds — kept in sync with the footer's own clamp. */
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 5;
/** Multiplicative step per wheel notch. */
const WHEEL_STEP = 1.12;

/**
 * Pan + scroll-wheel zoom for a canvas viewport. Attach to the VIEWPORT
 * element (the area that clips); it transforms a TARGET child element with
 * `translate(...) scale(...)`.
 *
 * - `zoom` is a two-way model so the footer's +/-/reset controls and the
 *   wheel share one source of truth. The host owns the signal (and resets it
 *   to 1 on image change); this directive only nudges it.
 * - Wheel zooms toward the cursor (the point under the pointer stays put).
 * - Drag pans, but only while zoomed in (>1); pan is clamped so the image
 *   can't be dragged entirely out of view, and is cleared whenever zoom
 *   returns to 1 (footer reset, zoom-out, or the host resetting on nav).
 *
 * The target defaults to the host's first element child; pass an explicit
 * element via `appPanZoomTarget` when the viewport has multiple children
 * (e.g. the Edit stage, which also holds nav/AB overlays).
 */
@Directive({
    selector: '[appPanZoom]',
    standalone: true,
})
export class PanZoomDirective {
    /** Two-way zoom factor (1 = 100%); shared with the footer controls. */
    zoom = model<number>(1);
    /** Element the transform is applied to. Defaults to host.firstElementChild. */
    target = input<ElementRef<HTMLElement> | HTMLElement | null>(null, { alias: 'appPanZoomTarget' });

    private hostRef = inject(ElementRef<HTMLElement>);

    private offsetX = signal(0);
    private offsetY = signal(0);

    private dragging = false;
    private pointerId: number | null = null;
    private lastX = 0;
    private lastY = 0;

    private targetEl(): HTMLElement | null {
        const t = this.target();
        if (t instanceof ElementRef) return t.nativeElement;
        if (t instanceof HTMLElement) return t;
        return this.hostRef.nativeElement.firstElementChild as HTMLElement | null;
    }

    constructor() {
        // Clear pan whenever we're back at 1:1 (reset button, zoom-out, or the
        // host resetting zoom on image change). Reads zoom only → no self-loop.
        effect(() => {
            if (this.zoom() <= 1) {
                this.offsetX.set(0);
                this.offsetY.set(0);
            }
        });

        // Apply the transform on any zoom/offset change.
        effect(() => {
            const el = this.targetEl();
            if (!el) return;
            const z = this.zoom();
            const x = this.offsetX();
            const y = this.offsetY();
            el.style.transformOrigin = 'center center';
            el.style.transform = `translate(${x}px, ${y}px) scale(${z})`;
            el.style.willChange = z > 1 ? 'transform' : '';
        });

        // Host cursor reflects pan availability (grabbing handled imperatively).
        effect(() => {
            this.hostRef.nativeElement.style.cursor = this.zoom() > 1 ? 'grab' : '';
        });
    }

    // ── Wheel zoom (cursor-anchored) ──────────────────────────────────────
    @HostListener('wheel', ['$event'])
    onWheel(e: WheelEvent): void {
        e.preventDefault();
        const cur = this.zoom();
        const factor = e.deltaY < 0 ? WHEEL_STEP : 1 / WHEEL_STEP;
        const next = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, +(cur * factor).toFixed(4)));
        if (next === cur) return;

        if (next > 1) {
            // Keep the content point under the cursor fixed. With transform-origin
            // at the (viewport-centered) element center, a screen point S maps to
            // local d = (S − center − offset)/z; solve offset' so S stays put.
            const rect = this.hostRef.nativeElement.getBoundingClientRect();
            const ux = e.clientX - (rect.left + rect.width / 2);
            const uy = e.clientY - (rect.top + rect.height / 2);
            this.offsetX.set(ux - (next * (ux - this.offsetX())) / cur);
            this.offsetY.set(uy - (next * (uy - this.offsetY())) / cur);
        }
        this.zoom.set(next);
        this.clampOffset(next);
    }

    // ── Drag to pan (only while zoomed in) ────────────────────────────────
    @HostListener('pointerdown', ['$event'])
    onPointerDown(e: PointerEvent): void {
        if (this.zoom() <= 1 || e.button !== 0) return;
        // Don't hijack gestures that begin on a control.
        if ((e.target as HTMLElement).closest('button, a, input, textarea, select, .ab-divider, [data-no-pan]')) {
            return;
        }
        this.dragging = true;
        this.pointerId = e.pointerId;
        this.lastX = e.clientX;
        this.lastY = e.clientY;
        this.hostRef.nativeElement.setPointerCapture?.(e.pointerId);
        this.hostRef.nativeElement.style.cursor = 'grabbing';
        e.preventDefault();
    }

    @HostListener('pointermove', ['$event'])
    onPointerMove(e: PointerEvent): void {
        if (!this.dragging || e.pointerId !== this.pointerId) return;
        this.offsetX.update(v => v + (e.clientX - this.lastX));
        this.offsetY.update(v => v + (e.clientY - this.lastY));
        this.lastX = e.clientX;
        this.lastY = e.clientY;
        this.clampOffset(this.zoom());
    }

    @HostListener('pointerup', ['$event'])
    @HostListener('pointercancel', ['$event'])
    onPointerUp(e: PointerEvent): void {
        if (e.pointerId !== this.pointerId) return;
        this.dragging = false;
        this.pointerId = null;
        this.hostRef.nativeElement.releasePointerCapture?.(e.pointerId);
        this.hostRef.nativeElement.style.cursor = this.zoom() > 1 ? 'grab' : '';
    }

    /** Clamp pan so the scaled image edge can't move past the viewport edge. */
    private clampOffset(z: number): void {
        const el = this.targetEl();
        if (!el) return;
        const host = this.hostRef.nativeElement;
        const maxX = Math.max(0, (el.offsetWidth * z - host.clientWidth) / 2);
        const maxY = Math.max(0, (el.offsetHeight * z - host.clientHeight) / 2);
        this.offsetX.update(v => Math.min(maxX, Math.max(-maxX, v)));
        this.offsetY.update(v => Math.min(maxY, Math.max(-maxY, v)));
    }
}
