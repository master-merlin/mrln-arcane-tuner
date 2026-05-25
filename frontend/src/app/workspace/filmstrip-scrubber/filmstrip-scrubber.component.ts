import {
    ChangeDetectionStrategy,
    Component,
    computed,
    ElementRef,
    HostListener,
    input,
    output,
    viewChild,
} from '@angular/core';
import { aggregateFilmstrip, type StripCell } from './filmstrip-aggregate';

interface Img {
    harmonized?: boolean;
    captioned?: boolean;
    masked?: boolean;
}

/**
 * Bottom-of-workspace filmstrip scrubber.
 *
 * Renders one cell per image (or one per aggregated range when the list
 * exceeds `threshold`). The orange "viewport" indicator marks the active
 * region. Clicking anywhere on the strip seeks; ArrowLeft / ArrowRight
 * also seek when no input element has focus.
 */
@Component({
    selector: 'app-filmstrip-scrubber',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="strip" #host (mousedown)="onMouseDown($event)">
            @for (c of cells(); track c.startIndex) {
                <span class="cell"
                      [style.flex]="c.count"
                      [class.h]="c.state.harmonized"
                      [class.c]="c.state.captioned"
                      [class.m]="c.state.masked"
                      [attr.title]="c.startIndex + 1 + (c.count > 1 ? '–' + (c.startIndex + c.count) : '')"></span>
            }
            @if (images().length > 0) {
                <div class="viewport"
                     [style.left.%]="viewportLeftPct()"
                     [style.width.%]="viewportWidthPct()"></div>
            }
        </div>
    `,
    styles: [`
        :host { display: block; }
        .strip {
            position: relative;
            display: flex;
            height: 28px;
            gap: 1px;
            cursor: pointer;
            user-select: none;
            background: var(--color-base);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
            padding: 3px;
            overflow: hidden;
        }
        .cell {
            background: var(--color-danger);
            opacity: 0.55;
            border-radius: 1.5px;
            min-width: 2px;
        }
        /* Captioned-only → warning (yellow). */
        .cell.c { background: var(--color-warning); opacity: 0.75; }
        /* Both captioned + masked → success (green). Wins over .c. */
        .cell.c.m { background: var(--color-success); opacity: 0.9; }
        /* Harmonization adds saturation. */
        .cell.h { opacity: 1; }
        .viewport {
            position: absolute;
            top: -2px;
            bottom: -2px;
            border: 2px solid var(--color-brand);
            border-radius: var(--radius-theme-md);
            background: oklch(0.68 0.13 55 / 0.08);
            box-shadow:
                0 0 0 4px oklch(0.68 0.13 55 / 0.10),
                0 4px 16px oklch(0 0 0 / 0.4);
            pointer-events: none;
        }
    `],
})
export class FilmstripScrubberComponent {
    images = input.required<ReadonlyArray<Img>>();
    activeIndex = input.required<number>();
    /** How many images the viewport "shows" — drives the orange window width. */
    viewportSize = input<number>(20);
    threshold = input<number>(140);

    seek = output<number>();

    private host = viewChild.required<ElementRef<HTMLDivElement>>('host');

    protected cells = computed<StripCell[]>(() =>
        aggregateFilmstrip(this.images(), this.threshold()),
    );

    protected viewportLeftPct = computed(() => {
        const total = Math.max(1, this.images().length);
        return (this.activeIndex() / total) * 100;
    });

    protected viewportWidthPct = computed(() => {
        const total = Math.max(1, this.images().length);
        return Math.max(2, (this.viewportSize() / total) * 100);
    });

    protected onMouseDown(e: MouseEvent): void {
        const total = this.images().length;
        if (total === 0) return;
        const rect = this.host().nativeElement.getBoundingClientRect();
        const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const idx = Math.min(total - 1, Math.floor(pct * total));
        this.seek.emit(idx);
    }

    @HostListener('document:keydown', ['$event'])
    protected onKey(e: KeyboardEvent): void {
        const t = e.target as HTMLElement | null;
        if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
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
