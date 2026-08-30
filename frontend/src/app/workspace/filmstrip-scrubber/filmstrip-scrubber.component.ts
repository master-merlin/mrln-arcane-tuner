import {
    ChangeDetectionStrategy,
    Component,
    computed,
    DestroyRef,
    effect,
    ElementRef,
    HostListener,
    inject,
    input,
    output,
    signal,
    viewChild,
} from '@angular/core';
import { aggregateFilmstrip, type StripCell } from './filmstrip-aggregate';
import { OverlayStore } from '../../state/overlay.store';
import { createInViewTracker } from '../../shared/in-view-tracker';

interface Img {
    harmonized?: boolean;
    captioned?: boolean;
    masked?: boolean;
    /** Direct URL to the media file. When the strip isn't aggregated
     *  (one cell per image) we render this as a tiny thumbnail so the
     *  user can scrub by visual recognition rather than positional
     *  bars alone. */
    thumbnailUrl?: string;
    mediaType?: string;
}

/**
 * Bottom-of-workspace filmstrip scrubber.
 *
 * Renders one cell per image (or one per aggregated range when the list
 * exceeds `threshold`). Single-image cells show the actual thumbnail;
 * aggregated cells fall back to color-coded readiness bars.
 *
 * The orange "viewport" indicator marks the active region; the active
 * cell additionally gets a brand-tinted border so the user can locate
 * "where am I" at a glance. Clicking a cell seeks to its start;
 * ArrowLeft / ArrowRight also seek when no input element has focus.
 *
 * On `activeIndex` change the strip scrolls the active cell into view
 * — keeping the filmstrip in sync with browse selection, details
 * prev/next, and editor navigation.
 */
@Component({
    selector: 'app-filmstrip-scrubber',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="strip" #host (mousedown)="onMouseDown($event)">
            @for (c of cells(); track c.startIndex) {
                <span class="cell"
                      [style.flex]="c.count === 1 && !!thumbAt(c.startIndex) ? '0 0 72px' : c.count"
                      [class.h]="c.state.harmonized"
                      [class.c]="c.state.captioned"
                      [class.m]="c.state.masked"
                      [class.active]="isActive(c)"
                      [class.thumb]="c.count === 1 && !!thumbAt(c.startIndex)"
                      [attr.title]="cellTitle(c)"
                      [attr.data-index]="c.startIndex">
                    @if (c.count === 1 && thumbAt(c.startIndex); as url) {
                        @if (mediaTypeAt(c.startIndex) === 'video') {
                            <span class="thumb-video">▶</span>
                        } @else if (isFailed(c.startIndex)) {
                            <!-- Thumbnail generation failed (corrupt source,
                                 codec issue, etc.). Quiet broken-image
                                 glyph so the strip doesn't dance forever. -->
                            <span class="thumb-error" aria-hidden="true" title="Thumbnail unavailable">
                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><line x1="1" y1="1" x2="23" y2="23"/><path d="M21 21H5a2 2 0 0 1-2-2V5"/><path d="M21 15V5a2 2 0 0 0-2-2H9"/></svg>
                            </span>
                        } @else {
                            <!-- Loading dots sit behind the img. They are
                                 REMOVED once the image loads, and never
                                 rendered for a cell that is off screen:
                                 lazy loading means an off-screen image
                                 never starts loading, so a spinner there
                                 would bounce forever with nothing behind
                                 it. Measured: 263 cells kept 789 CSS
                                 animations running and cost 18 ms of every
                                 frame — while idle. Covering the dots with
                                 an opaque image, which is what this used to
                                 do, does not stop them. See isPending. -->
                            @if (isPending(c.startIndex)) {
                                <span class="thumb-spinner" aria-hidden="true">
                                    <span></span><span></span><span></span>
                                </span>
                            }
                            <img [src]="url"
                                 alt=""
                                 loading="lazy"
                                 decoding="async"
                                 (load)="onImgLoad(c.startIndex)"
                                 (error)="onImgError(c.startIndex)"/>
                        }
                        <!-- The thumbnail covers the cell's own readiness
                             colour, which is what the legend explains. Repeat
                             it as a bar along the bottom edge so a strip of
                             thumbnails still answers the legend's question.
                             Same class combination as the aggregated cells,
                             so the two renderings can never drift apart. -->
                        <span class="status-bar" aria-hidden="true"></span>
                    }
                </span>
            }
        </div>
    `,
    styles: [`
        :host { display: block; }
        .strip {
            position: relative;
            display: flex;
            height: 56px;
            gap: 2px;
            cursor: pointer;
            user-select: none;
            background: var(--color-base);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
            padding: 3px;
            overflow-x: auto;
            overflow-y: hidden;
            scrollbar-width: thin;
        }
        .strip::-webkit-scrollbar { height: 6px; }
        .strip::-webkit-scrollbar-thumb {
            background: var(--color-surface-high);
            border-radius: 3px;
        }
        .cell {
            position: relative;
            background: var(--color-danger);
            opacity: 0.55;
            border-radius: 2px;
            min-width: 4px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        /* Aggregated colour-bar tones */
        .cell.c { background: var(--color-warning); opacity: 0.75; }
        .cell.c.m { background: var(--color-success); opacity: 0.9; }
        .cell.h { opacity: 1; }

        /* Thumbnail cells override the colour-bar treatment entirely —
           the readiness colour comes back as .status-bar below. */
        .cell.thumb {
            background: var(--color-surface-mid);
            opacity: 1;
            min-width: 72px;
            flex: 0 0 72px !important;
        }
        /* Readiness bar for thumbnail cells. Mirrors the aggregated cells'
           mapping exactly: base = missing, .c = captioned only, .c.m =
           masked, and un-harmonized is dimmed rather than recoloured
           (the legend names three colours, not four). z-index clears the
           thumbnail image, which sits at z-index 1. */
        .status-bar { display: none; }
        .cell.thumb .status-bar {
            display: block;
            position: absolute;
            left: 0;
            right: 0;
            bottom: 0;
            height: 4px;
            z-index: 2;
            background: var(--color-danger);
            opacity: 0.55;
            pointer-events: none;
        }
        .cell.thumb.c .status-bar { background: var(--color-warning); opacity: 0.75; }
        .cell.thumb.c.m .status-bar { background: var(--color-success); opacity: 0.9; }
        .cell.thumb.h .status-bar { opacity: 1; }
        .cell.thumb img {
            position: relative;
            z-index: 1;
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        .cell.thumb .thumb-spinner {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            pointer-events: none;
            z-index: 0;
        }
        .cell.thumb .thumb-spinner > span {
            width: 4px;
            height: 4px;
            background: var(--color-text-subtle);
            border-radius: 50%;
            animation: thumb-bounce 1.1s ease-in-out infinite;
        }
        .cell.thumb .thumb-spinner > span:nth-child(2) { animation-delay: 0.15s; }
        .cell.thumb .thumb-spinner > span:nth-child(3) { animation-delay: 0.30s; }
        @keyframes thumb-bounce {
            0%, 80%, 100% { transform: translateY(0); opacity: 0.35; }
            40% { transform: translateY(-5px); opacity: 1; }
        }
        .cell.thumb .thumb-error {
            color: var(--color-text-subtle);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .cell.thumb-video, .cell .thumb-video {
            color: var(--color-text-muted);
            font-size: 14px;
        }
        .cell.active {
            outline: 2px solid var(--color-brand);
            outline-offset: -2px;
            box-shadow: 0 0 0 4px oklch(0.68 0.13 55 / 0.20);
            z-index: 2;
        }
    `],
})
export class FilmstripScrubberComponent {
    images = input.required<ReadonlyArray<Img>>();
    activeIndex = input.required<number>();
    threshold = input<number>(140);

    seek = output<number>();

    private overlay = inject(OverlayStore);
    private host = viewChild.required<ElementRef<HTMLDivElement>>('host');

    /** Indices whose thumbnail `<img>` errored — replaces the perpetual
     *  spinner with a quiet broken-image glyph. */
    private failed = signal<Set<number>>(new Set());

    /** Indices whose thumbnail `<img>` reported `load`. */
    private loaded = signal<Set<number>>(new Set());

    /** Bounds the spinner to cells a user can see — see in-view-tracker. */
    private inView = createInViewTracker({ selector: '.cell[data-index]' });

    /**
     * When every image carries a `thumbnailUrl`, the strip should show
     * one cell per image regardless of count — the actual thumbnails
     * ARE the overview, so aggregating into colour-coded ranges would
     * just hide most of them. Falls back to threshold-based aggregation
     * only when thumbnails are unavailable (legacy callers / video-only
     * lists). The horizontal scroll on `.strip` handles long lists, and
     * `loading="lazy"` keeps off-screen tiles from fetching upfront.
     */
    protected cells = computed<StripCell[]>(() => {
        const imgs = this.images();
        const allHaveThumbs = imgs.length > 0 && imgs.every(i => !!i.thumbnailUrl);
        if (allHaveThumbs) {
            return imgs.map((img, i) => ({
                startIndex: i,
                count: 1,
                state: {
                    harmonized: !!img.harmonized,
                    captioned: !!img.captioned,
                    masked: !!img.masked,
                },
            }));
        }
        return aggregateFilmstrip(imgs, this.threshold());
    });

    protected isFailed(idx: number): boolean { return this.failed().has(idx); }

    /**
     * Show the loading dots only where they describe something that is
     * really happening: the cell is on screen (so its lazy image has been
     * asked for) and that image has not reported `load` yet.
     */
    protected isPending(idx: number): boolean {
        return !this.loaded().has(idx) && this.inView.has(idx);
    }

    protected onImgLoad(idx: number): void {
        this.loaded.update(s => {
            if (s.has(idx)) return s;
            const next = new Set(s);
            next.add(idx);
            return next;
        });
    }

    protected onImgError(idx: number): void {
        this.failed.update(s => {
            if (s.has(idx)) return s;
            const next = new Set(s);
            next.add(idx);
            return next;
        });
    }

    constructor() {
        // Auto-scroll the active cell into view when the upstream cursor
        // jumps (browse click, details prev/next, filmstrip click). Read
        // both inputs to register dependencies, then defer the DOM read
        // to a microtask so the *new* cell DOM is in place.
        effect(() => {
            const idx = this.activeIndex();
            this.cells(); // re-run when cells regenerate
            queueMicrotask(() => this.scrollActiveIntoView(idx));
        });
        // Re-observe after the cell list changes. Same deferral as above:
        // the new cell DOM only exists on the next microtask.
        effect(() => {
            this.cells();
            queueMicrotask(() => this.inView.refresh(this.host()?.nativeElement));
        });
        inject(DestroyRef).onDestroy(() => this.inView.destroy());
    }

    protected isActive(c: StripCell): boolean {
        const i = this.activeIndex();
        return i >= c.startIndex && i < c.startIndex + c.count;
    }

    protected thumbAt(index: number): string | undefined {
        return this.images()[index]?.thumbnailUrl;
    }

    protected mediaTypeAt(index: number): string | undefined {
        return this.images()[index]?.mediaType;
    }

    protected cellTitle(c: StripCell): string {
        return c.count > 1
            ? `${c.startIndex + 1}–${c.startIndex + c.count}`
            : String(c.startIndex + 1);
    }

    protected onMouseDown(e: MouseEvent): void {
        const total = this.images().length;
        if (total === 0) return;
        const target = e.target as HTMLElement | null;
        // Prefer the cell's data-index so single-image clicks land on
        // exactly that image (clicking on a thumbnail's img child still
        // bubbles to the .cell via composedPath / closest).
        const cellEl = target?.closest?.('.cell') as HTMLElement | null;
        const dataIdx = cellEl?.getAttribute('data-index');
        if (dataIdx != null) {
            const idx = parseInt(dataIdx, 10);
            if (Number.isFinite(idx)) {
                this.seek.emit(Math.max(0, Math.min(total - 1, idx)));
                return;
            }
        }
        // Aggregated strip or click between cells — fall back to %-based seek.
        const rect = this.host().nativeElement.getBoundingClientRect();
        const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const idx = Math.min(total - 1, Math.floor(pct * total));
        this.seek.emit(idx);
    }

    private scrollActiveIntoView(idx: number): void {
        const host = this.host().nativeElement;
        const cellEl = host.querySelector<HTMLElement>(`.cell[data-index="${idx}"]`);
        if (!cellEl) return;
        const cellLeft = cellEl.offsetLeft;
        const cellRight = cellLeft + cellEl.offsetWidth;
        const viewLeft = host.scrollLeft;
        const viewRight = viewLeft + host.clientWidth;
        if (cellLeft < viewLeft) {
            host.scrollTo({ left: cellLeft - 12, behavior: 'smooth' });
        } else if (cellRight > viewRight) {
            host.scrollTo({ left: cellRight - host.clientWidth + 12, behavior: 'smooth' });
        }
    }

    @HostListener('document:keydown', ['$event'])
    protected onKey(e: KeyboardEvent): void {
        // Suppress when an input is focused (existing) OR when any modal is
        // open above the workspace — arrow keys must not move the underlying
        // cursor while crop / mask-preview / analyze modals are showing.
        const t = e.target as HTMLElement | null;
        if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
        if (this.overlay.modalStack().length > 0) return;
        const total = this.images().length;
        if (total === 0) return;
        if (e.key === 'ArrowLeft') {
            this.seek.emit(Math.max(0, this.activeIndex() - 1));
            e.preventDefault();
        } else if (e.key === 'ArrowRight') {
            this.seek.emit(Math.min(total - 1, this.activeIndex() + 1));
            e.preventDefault();
        }
    }
}
