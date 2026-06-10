// cooccurrence-heatmap.ts
import {
    ChangeDetectionStrategy, Component, ElementRef, computed, effect, input, viewChild,
} from '@angular/core';
import type { Cooccurrence } from '../../services/dataset';

export interface HeatCell { row: number; col: number; value: number; intensity: number; }

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
    template: `<canvas #cv class="w-full" [style.height.px]="size()"></canvas>`,
})
export class CooccurrenceHeatmapComponent {
    data = input<Cooccurrence | null>(null);
    private canvas = viewChild<ElementRef<HTMLCanvasElement>>('cv');

    protected size = computed(() => Math.max(160, (this.data()?.labels.length ?? 0) * 16 + 80));

    constructor() {
        effect(() => {
            const d = this.data();
            const el = this.canvas()?.nativeElement;
            if (d && el) requestAnimationFrame(() => this.render(el, d));
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
        if (!n) return;
        const pad = 76;
        const grid = Math.min(rect.width - pad, rect.height - pad);
        const cell = grid / n;
        const cells = cooccurrenceCells(labels, d.matrix);
        for (const c of cells) {
            ctx.fillStyle = `rgba(99, 102, 241, ${c.intensity.toFixed(3)})`;
            ctx.fillRect(pad + c.col * cell, pad + c.row * cell, cell - 1, cell - 1);
        }
        ctx.fillStyle = '#9ca3af';
        ctx.font = '9px monospace';
        for (let i = 0; i < n; i++) {
            ctx.fillText(labels[i].slice(0, 10), 2, pad + i * cell + cell / 2 + 3);
        }
    }
}
