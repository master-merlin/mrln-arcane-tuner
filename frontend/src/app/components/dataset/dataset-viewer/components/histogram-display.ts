import { Component, input, ElementRef, ViewChild, effect, ChangeDetectionStrategy } from '@angular/core';
import { HistogramData } from '../../../../services/dataset';

@Component({
    selector: 'app-histogram-display',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [],
    template: `
    <div class="flex flex-col gap-2">
        <div class="flex items-center justify-end px-1">
            <div class="flex gap-1">
                @for (ch of channels; track ch.key) {
                    <button
                        (click)="toggleChannel(ch.key)"
                        [class]="'w-6 h-6 rounded-full text-[10px] font-bold flex items-center justify-center transition-all ' +
                            (activeChannels[ch.key] ? ch.activeClass : 'bg-surface-mid/50 text-text-disabled hover:bg-surface-mid/80')"
                        [attr.data-testid]="'histogram-toggle-' + ch.key">
                        {{ ch.label }}
                    </button>
                }
            </div>
        </div>
        <div class="relative rounded-theme-lg overflow-hidden border border-surface-high/30" style="background: #111;">
            <canvas #histCanvas class="w-full" style="height: 160px;" data-testid="histogram-canvas"></canvas>
        </div>
    </div>
    `,
    styles: []
})
export class HistogramDisplayComponent {
    @ViewChild('histCanvas') canvasRef!: ElementRef<HTMLCanvasElement>;

    data = input<HistogramData | null>(null);

    channels = [
        { key: 'r' as const, label: 'R', color: 'rgba(255, 60, 60, 0.70)', strokeColor: 'rgba(255, 60, 60, 0.9)', activeClass: 'bg-red-500/80 text-white' },
        { key: 'g' as const, label: 'G', color: 'rgba(60, 220, 60, 0.65)', strokeColor: 'rgba(60, 220, 60, 0.85)', activeClass: 'bg-green-500/80 text-white' },
        { key: 'b' as const, label: 'B', color: 'rgba(60, 100, 255, 0.70)', strokeColor: 'rgba(60, 100, 255, 0.9)', activeClass: 'bg-blue-500/80 text-white' },
        { key: 'luminance' as const, label: 'L', color: 'rgba(255, 255, 255, 0.40)', strokeColor: 'rgba(255, 255, 255, 0.6)', activeClass: 'bg-white/80 text-black' },
    ];

    activeChannels: Record<string, boolean> = { r: true, g: true, b: true, luminance: true };

    constructor() {
        effect(() => {
            const d = this.data();
            if (d) {
                requestAnimationFrame(() => this.render(d));
            }
        });
    }

    toggleChannel(key: string): void {
        this.activeChannels[key] = !this.activeChannels[key];
        const d = this.data();
        if (d) this.render(d);
    }

    private render(data: HistogramData): void {
        const canvas = this.canvasRef?.nativeElement;
        if (!canvas) return;

        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

        const w = rect.width;
        const h = rect.height;

        ctx.clearRect(0, 0, w, h);

        // Draw grid
        this.renderGrid(ctx, w, h);

        // Find global max for normalization (skip top 0.1% to avoid spike domination)
        let globalMax = 1;
        for (const ch of this.channels) {
            if (!this.activeChannels[ch.key]) continue;
            const arr = data[ch.key];
            if (arr) {
                // Use 99.5th percentile instead of absolute max for better visibility
                const sorted = [...arr].sort((a, b) => b - a);
                const pMax = sorted[Math.floor(sorted.length * 0.005)] || sorted[0];
                if (pMax > globalMax) globalMax = pMax;
            }
        }

        // Draw each active channel as filled area + stroke
        const drawOrder: (keyof HistogramData)[] = ['luminance', 'b', 'g', 'r'];
        for (const key of drawOrder) {
            if (!this.activeChannels[key]) continue;
            const arr = data[key];
            const chDef = this.channels.find(c => c.key === key);
            if (!arr || !chDef) continue;

            // Filled area
            ctx.fillStyle = chDef.color;
            ctx.beginPath();
            ctx.moveTo(0, h);

            for (let i = 0; i < 256; i++) {
                const x = (i / 255) * w;
                const barH = Math.min(arr[i] / globalMax, 1.0) * (h - 4);
                ctx.lineTo(x, h - barH);
            }

            ctx.lineTo(w, h);
            ctx.closePath();
            ctx.fill();

            // Stroke outline for sharper definition
            ctx.strokeStyle = chDef.strokeColor;
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = 0; i < 256; i++) {
                const x = (i / 255) * w;
                const barH = Math.min(arr[i] / globalMax, 1.0) * (h - 4);
                if (i === 0) ctx.moveTo(x, h - barH);
                else ctx.lineTo(x, h - barH);
            }
            ctx.stroke();
        }
    }

    private renderGrid(ctx: CanvasRenderingContext2D, w: number, h: number): void {
        const divisions = 4;
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        ctx.lineWidth = 1;

        for (let i = 1; i < divisions; i++) {
            const x = (i / divisions) * w;
            const y = (i / divisions) * h;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        }

        // Zone labels
        ctx.fillStyle = 'rgba(255, 255, 255, 0.12)';
        ctx.font = '9px monospace';
        ctx.fillText('Shadows', 4, h - 4);
        ctx.fillText('Midtones', w * 0.35, h - 4);
        ctx.fillText('Highlights', w * 0.72, h - 4);
    }
}
