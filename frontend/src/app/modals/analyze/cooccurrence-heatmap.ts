// cooccurrence-heatmap.ts
import {
    ChangeDetectionStrategy, Component, ElementRef, computed, effect, input, signal, viewChild,
} from '@angular/core';
import type { Cooccurrence } from '../../services/dataset';

export interface HeatCell { row: number; col: number; value: number; intensity: number; }

/** Live tooltip shown while hovering a cell. */
interface HeatHover { x: number; y: number; flipX: boolean; a: string; b: string; value: number; }

/** Pixel geometry of the last render — used to map cursor → cell on hover. */
interface HeatLayout { padL: number; padT: number; cell: number; n: number; width: number; }

/** Flatten a co-occurrence matrix into cells with intensity normalized to the max value. */
export function cooccurrenceCells(labels: string[], matrix: number[][]): HeatCell[] {
    if (!labels.length) return [];
    let max = 0;
    for (const r of matrix) for (const v of r) if (v > max) max = v;
    const cells: HeatCell[] = [];
    for (let row = 0; row < matrix.length; row++) {
        for (let col = 0; col < matrix[row].length; col++) {
            const value = matrix[row][col];
            cells.push({ row, col, value, intensity: max > 0 ? value / max : 0 });
        }
    }
    return cells;
}

@Component({
    selector: 'app-cooccurrence-heatmap',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="heat-wrap" (mousemove)="onMove($event)" (mouseleave)="hover.set(null)">
            <canvas #cv class="w-full" [style.height.px]="size()"></canvas>
            @if (hover(); as h) {
                <div class="heat-tip" [class.flip]="h.flipX" [style.left.px]="h.x" [style.top.px]="h.y">
                    <div class="heat-tip-terms">
                        {{ h.a }}@if (h.a !== h.b) {<span class="heat-tip-x">×</span>{{ h.b }}}
                    </div>
                    <div class="heat-tip-count">
                        {{ h.a === h.b ? 'appears in' : 'together in' }}
                        <b>{{ h.value }}</b> image{{ h.value === 1 ? '' : 's' }}
                    </div>
                </div>
            }
        </div>
    `,
    styles: [`
        .heat-wrap { position: relative; }
        .heat-tip {
            position: absolute; z-index: 5; pointer-events: none;
            transform: translate(12px, 12px);
            background: var(--color-surface-high);
            border: 1px solid var(--color-border-default);
            border-radius: var(--radius-theme-md);
            padding: 6px 9px; box-shadow: var(--shadow-sm);
            max-width: 240px; white-space: nowrap;
        }
        .heat-tip.flip { transform: translate(calc(-100% - 12px), 12px); }
        .heat-tip-terms {
            font-size: 11.5px; font-weight: 600; color: var(--color-text-primary);
            overflow: hidden; text-overflow: ellipsis;
        }
        .heat-tip-x { color: var(--color-text-muted); margin: 0 5px; font-weight: 400; }
        .heat-tip-count { font-size: 10.5px; color: var(--color-text-muted); margin-top: 1px; }
        .heat-tip-count b { color: var(--color-brand-light); font-variant-numeric: tabular-nums; }
    `],
})
export class CooccurrenceHeatmapComponent {
    data = input<Cooccurrence | null>(null);
    private canvas = viewChild<ElementRef<HTMLCanvasElement>>('cv');

    protected size = computed(() => Math.max(140, (this.data()?.labels.length ?? 0) * 16 + 28));
    protected hover = signal<HeatHover | null>(null);

    /** Geometry captured at the last render, so hover can map pixels → cell. */
    private layout: HeatLayout | null = null;

    constructor() {
        effect(() => {
            const d = this.data();
            const el = this.canvas()?.nativeElement;
            this.hover.set(null);
            if (d && el) requestAnimationFrame(() => this.render(el, d));
        });
    }

    /** Map the cursor to a grid cell and surface the two terms + their count. */
    protected onMove(ev: MouseEvent): void {
        const lay = this.layout;
        const d = this.data();
        if (!lay || !d) return;
        const target = ev.currentTarget as HTMLElement;
        const rect = target.getBoundingClientRect();
        const x = ev.clientX - rect.left;
        const y = ev.clientY - rect.top;
        const col = Math.floor((x - lay.padL) / lay.cell);
        const row = Math.floor((y - lay.padT) / lay.cell);
        if (x < lay.padL || y < lay.padT || row < 0 || col < 0 || row >= lay.n || col >= lay.n) {
            this.hover.set(null);
            return;
        }
        this.hover.set({
            x, y,
            flipX: x > lay.width - 170,
            a: d.labels[row],
            b: d.labels[col],
            value: d.matrix[row]?.[col] ?? 0,
        });
    }

    private render(canvas: HTMLCanvasElement, d: Cooccurrence): void {
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, rect.width, rect.height);

        const labels = d.labels;
        const n = labels.length;
        if (!n) { this.layout = null; return; }
        // Separate left/top padding: the left gutter holds the row labels, the
        // top only needs a small breathing margin (no column labels drawn), so
        // the grid hugs the card header instead of leaving dead space above it.
        const padL = 76;
        const padT = 10;
        const grid = Math.min(rect.width - padL - 12, rect.height - padT - 12);
        const cell = grid / n;
        // Stash geometry so the hover handler maps the cursor to a cell using
        // the exact pad/cell the cells were drawn with.
        this.layout = { padL, padT, cell, n, width: rect.width };
        const cells = cooccurrenceCells(labels, d.matrix);
        for (const c of cells) {
            ctx.fillStyle = `rgba(99, 102, 241, ${c.intensity.toFixed(3)})`;
            ctx.fillRect(padL + c.col * cell, padT + c.row * cell, cell - 1, cell - 1);
        }
        ctx.fillStyle = '#9ca3af';
        ctx.font = '9px monospace';
        for (let i = 0; i < n; i++) {
            ctx.fillText(labels[i].slice(0, 10), 2, padT + i * cell + cell / 2 + 3);
        }
    }
}
