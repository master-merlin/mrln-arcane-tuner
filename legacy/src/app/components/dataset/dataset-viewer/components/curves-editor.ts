import { Component, input, output, signal, computed, ElementRef, ViewChild, effect, AfterViewInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CurvePoint, HistogramData } from '../../../../services/dataset';

type ChannelKey = 'master' | 'r' | 'g' | 'b';

interface CurvePreset {
    name: string;
    points: CurvePoint[];
}

const IDENTITY: CurvePoint[] = [{ x: 0, y: 0 }, { x: 255, y: 255 }];

const PRESETS: CurvePreset[] = [
    { name: 'Linear', points: [{ x: 0, y: 0 }, { x: 255, y: 255 }] },
    { name: 'S-Curve', points: [{ x: 0, y: 0 }, { x: 64, y: 48 }, { x: 192, y: 208 }, { x: 255, y: 255 }] },
    { name: 'High Key', points: [{ x: 0, y: 40 }, { x: 128, y: 148 }, { x: 255, y: 255 }] },
    { name: 'Low Key', points: [{ x: 0, y: 0 }, { x: 128, y: 108 }, { x: 255, y: 215 }] },
    { name: 'Cross Process', points: [{ x: 0, y: 20 }, { x: 64, y: 80 }, { x: 192, y: 220 }, { x: 255, y: 240 }] },
];

@Component({
    selector: 'app-curves-editor',
    standalone: true,
    imports: [FormsModule],
    template: `
    <div class="flex flex-col gap-3">
        <!-- Channel Selector -->
        <div class="flex items-center gap-1">
            @for (ch of channelDefs; track ch.key) {
                <button
                    (click)="selectChannel(ch.key)"
                    [class]="'w-6 h-6 rounded-full text-[10px] font-bold flex items-center justify-center transition-all ' +
                        (activeChannel() === ch.key ? ch.activeClass : 'bg-surface-mid/50 text-text-disabled hover:bg-surface-mid/80')"
                    [attr.data-testid]="'curves-channel-' + ch.key">
                    {{ ch.label }}
                </button>
            }
        </div>

        <!-- Curves Canvas -->
        <div class="relative rounded-theme-lg overflow-hidden border border-surface-high/30" style="aspect-ratio: 1;">
            <canvas #curvesCanvas
                class="w-full h-full cursor-crosshair"
                (mousedown)="onMouseDown($event)"
                (dblclick)="onDblClick($event)"
                data-testid="curves-canvas">
            </canvas>
        </div>

        <!-- Input/Output Readout -->
        <div class="flex items-center justify-between px-1 text-[11px] font-mono text-text-muted">
            <span>In: {{ hoverInput() }}  Out: {{ hoverOutput() }}</span>
            <label class="flex items-center gap-1.5 cursor-pointer">
                <input type="checkbox" [checked]="showGrid()" (change)="showGrid.set(!showGrid())"
                    class="w-3.5 h-3.5 rounded accent-brand" data-testid="curves-grid-toggle">
                <span>Fine Grid</span>
            </label>
        </div>

        <!-- Presets & Reset -->
        <div class="flex flex-col gap-1.5">
            <div class="flex items-center gap-2">
                <select [ngModel]="selectedPreset()"
                    (ngModelChange)="onPresetChange($event)"
                    class="flex-1 bg-surface-low border border-surface-high/30 rounded-theme-lg px-2 py-1.5 text-xs text-text outline-none focus:border-brand transition-colors"
                    data-testid="curves-preset-select">
                    <option value="">Preset…</option>
                    @for (preset of presets; track preset.name) {
                        <option [value]="preset.name">{{ preset.name }}</option>
                    }
                </select>
            </div>
            <button (click)="resetChannel()"
                class="w-full px-2 py-1.5 text-[10px] rounded-theme-lg bg-surface-mid/40 hover:bg-surface-mid/80 text-text-muted hover:text-text transition-all"
                data-testid="curves-reset">
                ⟲ Reset Channel
            </button>
        </div>
    </div>
    `,
    styles: []
})
export class CurvesEditorComponent implements AfterViewInit {
    @ViewChild('curvesCanvas') canvasRef!: ElementRef<HTMLCanvasElement>;

    // Inputs: current curve data per channel
    masterCurve = input<CurvePoint[]>([...IDENTITY]);
    rCurve = input<CurvePoint[]>([...IDENTITY]);
    gCurve = input<CurvePoint[]>([...IDENTITY]);
    bCurve = input<CurvePoint[]>([...IDENTITY]);
    histogramData = input<HistogramData | null>(null);

    // Outputs
    curveChanged = output<{ channel: ChannelKey; points: CurvePoint[] }>();

    activeChannel = signal<ChannelKey>('master');
    showGrid = signal(false);
    selectedPreset = signal<string>('');
    hoverInput = signal(0);
    hoverOutput = signal(0);

    presets = PRESETS;
    channelDefs = [
        { key: 'master' as ChannelKey, label: 'M', color: '#fff', activeClass: 'bg-white/90 text-black' },
        { key: 'r' as ChannelKey, label: 'R', color: '#ff5050', activeClass: 'bg-red-500/80 text-white' },
        { key: 'g' as ChannelKey, label: 'G', color: '#50ff50', activeClass: 'bg-green-500/80 text-white' },
        { key: 'b' as ChannelKey, label: 'B', color: '#5078ff', activeClass: 'bg-blue-500/80 text-white' },
    ];

    private draggingIdx = -1;
    private draggingPoints: CurvePoint[] | null = null;
    private draggedOutside = false;
    private pointsWithoutDragged: CurvePoint[] | null = null;
    private canvasSize = 0;

    // Bound handlers for document-level listeners during drag
    private boundMouseMove = (e: MouseEvent) => this.onMouseMove(e);
    private boundMouseUp = () => this.onMouseUp();

    constructor() {
        // Re-render when any input changes
        effect(() => {
            this.masterCurve();
            this.rCurve();
            this.gCurve();
            this.bCurve();
            this.histogramData();
            this.activeChannel();
            this.showGrid();
            requestAnimationFrame(() => this.render());
        });
    }

    ngAfterViewInit(): void {
        requestAnimationFrame(() => this.render());
    }

    selectChannel(key: ChannelKey): void {
        this.activeChannel.set(key);
    }

    applyPreset(preset: CurvePreset): void {
        const ch = this.activeChannel();
        this.curveChanged.emit({ channel: ch, points: preset.points.map(p => ({ ...p })) });
    }

    onPresetChange(name: string): void {
        const preset = this.presets.find(p => p.name === name);
        if (preset) {
            this.applyPreset(preset);
        }
        // Reset dropdown to placeholder after applying
        this.selectedPreset.set('');
    }

    resetChannel(): void {
        this.selectedPreset.set('');
        this.curveChanged.emit({ channel: this.activeChannel(), points: [...IDENTITY] });
    }

    // ── Mouse Handlers ──────────────────────────────────────────────────

    onMouseDown(event: MouseEvent): void {
        const { x, y } = this.eventToValue(event);
        const points = this.getActivePoints();

        // Check if clicking near an existing point
        const hitIdx = this.findNearPoint(points, x, y);
        if (hitIdx >= 0) {
            this.draggingIdx = hitIdx;
            this.draggingPoints = [...points];
        } else {
            // Insert new point — store the snapshot locally
            const newPoints = [...points, { x, y }].sort((a, b) => a.x - b.x);
            this.draggingIdx = newPoints.findIndex(p => p.x === x && p.y === y);
            this.draggingPoints = newPoints;
            this.curveChanged.emit({ channel: this.activeChannel(), points: newPoints });
        }

        // Attach document-level listeners so drag works outside the canvas
        document.addEventListener('mousemove', this.boundMouseMove);
        document.addEventListener('mouseup', this.boundMouseUp);
        event.preventDefault();
    }

    onMouseMove(event: MouseEvent): void {
        const { x, y } = this.eventToValue(event);
        this.hoverInput.set(x);

        if (this.draggingIdx >= 0 && this.draggingPoints) {
            const isEndpoint = this.draggingIdx === 0 || this.draggingIdx === this.draggingPoints.length - 1;
            const outside = !isEndpoint && this.isOutsideCanvas(event);

            if (outside) {
                // Dragged outside — show curve without this point (removal preview)
                if (!this.draggedOutside) {
                    this.draggedOutside = true;
                    this.pointsWithoutDragged = this.draggingPoints.filter((_, i) => i !== this.draggingIdx);
                    this.curveChanged.emit({ channel: this.activeChannel(), points: this.pointsWithoutDragged });
                }
            } else {
                // Inside canvas — move the point normally
                this.draggedOutside = false;
                this.pointsWithoutDragged = null;

                const points = [...this.draggingPoints];
                const pt = points[this.draggingIdx];
                if (!pt) { this.draggingIdx = -1; this.draggingPoints = null; return; }

                if (this.draggingIdx === 0) {
                    points[0] = { x: 0, y };
                } else if (this.draggingIdx === points.length - 1) {
                    points[points.length - 1] = { x: 255, y };
                } else {
                    const minX = points[this.draggingIdx - 1].x + 1;
                    const maxX = points[this.draggingIdx + 1].x - 1;
                    points[this.draggingIdx] = { x: Math.max(minX, Math.min(maxX, x)), y };
                }
                this.draggingPoints = points;
                this.curveChanged.emit({ channel: this.activeChannel(), points });
            }
        }

        // Compute output at hover position from current curve
        const activePts = this.pointsWithoutDragged || this.draggingPoints || this.getActivePoints();
        const lut = this.buildLUT(activePts);
        this.hoverOutput.set(lut[Math.max(0, Math.min(255, x))]);
    }

    onMouseUp(): void {
        if (this.draggedOutside && this.pointsWithoutDragged) {
            // Finalize removal — curve already shows the state without the point
            this.curveChanged.emit({ channel: this.activeChannel(), points: this.pointsWithoutDragged });
        }
        this.draggingIdx = -1;
        this.draggingPoints = null;
        this.draggedOutside = false;
        this.pointsWithoutDragged = null;

        // Remove document-level listeners
        document.removeEventListener('mousemove', this.boundMouseMove);
        document.removeEventListener('mouseup', this.boundMouseUp);
    }

    onDblClick(event: MouseEvent): void {
        const { x, y } = this.eventToValue(event);
        const points = this.getActivePoints();
        const hitIdx = this.findNearPoint(points, x, y);

        // Don't remove first/last
        if (hitIdx > 0 && hitIdx < points.length - 1) {
            const newPoints = points.filter((_, i) => i !== hitIdx);
            this.curveChanged.emit({ channel: this.activeChannel(), points: newPoints });
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private getActivePoints(): CurvePoint[] {
        switch (this.activeChannel()) {
            case 'master': return this.masterCurve();
            case 'r': return this.rCurve();
            case 'g': return this.gCurve();
            case 'b': return this.bCurve();
        }
    }

    private eventToValue(event: MouseEvent): { x: number; y: number } {
        const canvas = this.canvasRef?.nativeElement;
        if (!canvas) return { x: 0, y: 0 };
        const rect = canvas.getBoundingClientRect();
        const px = event.clientX - rect.left;
        const py = event.clientY - rect.top;
        const x = Math.round((px / rect.width) * 255);
        const y = Math.round((1 - py / rect.height) * 255);
        return { x: Math.max(0, Math.min(255, x)), y: Math.max(0, Math.min(255, y)) };
    }

    private isOutsideCanvas(event: MouseEvent): boolean {
        const canvas = this.canvasRef?.nativeElement;
        if (!canvas) return false;
        const rect = canvas.getBoundingClientRect();
        const margin = 20;
        return event.clientX < rect.left - margin || event.clientX > rect.right + margin ||
            event.clientY < rect.top - margin || event.clientY > rect.bottom + margin;
    }

    private findNearPoint(points: CurvePoint[], x: number, y: number): number {
        const threshold = 10; // in value space (0-255)
        for (let i = 0; i < points.length; i++) {
            const dx = Math.abs(points[i].x - x);
            const dy = Math.abs(points[i].y - y);
            if (dx < threshold && dy < threshold) return i;
        }
        return -1;
    }

    // ── LUT interpolation (tension-controlled Catmull-Rom for smoother curves) ─────

    private static readonly TENSION = 0.4; // 0=sharp Catmull-Rom, 1=linear

    buildLUT(points: CurvePoint[]): number[] {
        if (points.length < 2) {
            return Array.from({ length: 256 }, (_, i) => i);
        }

        const sorted = [...points].sort((a, b) => a.x - b.x);
        const xs = sorted.map(p => p.x);
        const ys = sorted.map(p => p.y);
        const lut: number[] = new Array(256);
        const tau = 1 - CurvesEditorComponent.TENSION;

        for (let i = 0; i < 256; i++) {
            if (i <= xs[0]) { lut[i] = ys[0]; continue; }
            if (i >= xs[xs.length - 1]) { lut[i] = ys[ys.length - 1]; continue; }

            let seg = 0;
            while (seg < xs.length - 2 && xs[seg + 1] < i) seg++;

            const x0 = xs[seg], x1 = xs[seg + 1];
            const y0 = ys[seg], y1 = ys[seg + 1];
            const t = (i - x0) / (x1 - x0);

            // Reflected phantom points for natural boundary tangents
            const ym1 = seg > 0 ? ys[seg - 1] : 2 * y0 - y1;
            const y2 = seg < xs.length - 2 ? ys[seg + 2] : 2 * y1 - y0;
            const t2 = t * t, t3 = t2 * t;

            const val = tau * 0.5 * (
                (2 * y0) +
                (-ym1 + y1) * t +
                (2 * ym1 - 5 * y0 + 4 * y1 - y2) * t2 +
                (-ym1 + 3 * y0 - 3 * y1 + y2) * t3
            ) + (1 - tau) * (y0 + (y1 - y0) * t);

            lut[i] = Math.max(0, Math.min(255, Math.round(val)));
        }

        return lut;
    }

    // ── Rendering ───────────────────────────────────────────────────────

    private render(): void {
        const canvas = this.canvasRef?.nativeElement;
        if (!canvas) return;

        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio;
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.scale(dpr, dpr);

        const w = rect.width;
        const h = rect.height;
        this.canvasSize = w;

        // Background
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, w, h);

        // Histogram ghost
        this.renderHistogram(ctx, w, h);

        // Grid
        this.renderGrid(ctx, w, h);

        // Identity diagonal
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, h);
        ctx.lineTo(w, 0);
        ctx.stroke();

        // Draw inactive channel curves faintly
        for (const chDef of this.channelDefs) {
            if (chDef.key === this.activeChannel()) continue;
            const pts = this.getPointsForChannel(chDef.key);
            if (pts.length >= 2) {
                this.renderCurve(ctx, w, h, pts, chDef.color, 0.2, 1);
            }
        }

        // Draw active channel curve
        const activeDef = this.channelDefs.find(c => c.key === this.activeChannel())!;
        const activePoints = this.getActivePoints();
        this.renderCurve(ctx, w, h, activePoints, activeDef.color, 1.0, 2);

        // Draw control points for active channel
        this.renderPoints(ctx, w, h, activePoints, activeDef.color);
    }

    private getPointsForChannel(key: ChannelKey): CurvePoint[] {
        switch (key) {
            case 'master': return this.masterCurve();
            case 'r': return this.rCurve();
            case 'g': return this.gCurve();
            case 'b': return this.bCurve();
        }
    }

    private renderGrid(ctx: CanvasRenderingContext2D, w: number, h: number): void {
        const divisions = this.showGrid() ? 10 : 4;
        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.lineWidth = 1;

        for (let i = 1; i < divisions; i++) {
            const pos = (i / divisions) * w;
            ctx.beginPath(); ctx.moveTo(pos, 0); ctx.lineTo(pos, h); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, pos); ctx.lineTo(w, pos); ctx.stroke();
        }
    }

    private renderHistogram(ctx: CanvasRenderingContext2D, w: number, h: number): void {
        const data = this.histogramData();
        if (!data) return;

        const ch = this.activeChannel();
        let arr: number[] | undefined;
        let color: string;

        if (ch === 'master') {
            arr = data.luminance;
            color = 'rgba(255,255,255,0.12)';
        } else {
            arr = data[ch];
            const colorMap = { r: 'rgba(255,80,80,0.15)', g: 'rgba(80,255,80,0.15)', b: 'rgba(80,120,255,0.15)' };
            color = colorMap[ch];
        }

        if (!arr) return;
        const max = Math.max(...arr, 1);

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(0, h);
        for (let i = 0; i < 256; i++) {
            const x = (i / 255) * w;
            const barH = (arr[i] / max) * h * 0.8;
            ctx.lineTo(x, h - barH);
        }
        ctx.lineTo(w, h);
        ctx.closePath();
        ctx.fill();
    }

    private renderCurve(ctx: CanvasRenderingContext2D, w: number, h: number, points: CurvePoint[], color: string, alpha: number, lineWidth: number): void {
        const lut = this.buildLUT(points);
        ctx.strokeStyle = color;
        ctx.globalAlpha = alpha;
        ctx.lineWidth = lineWidth;
        ctx.beginPath();

        for (let i = 0; i < 256; i++) {
            const x = (i / 255) * w;
            const y = (1 - lut[i] / 255) * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
    }

    private renderPoints(ctx: CanvasRenderingContext2D, w: number, h: number, points: CurvePoint[], color: string): void {
        const radius = 5;
        for (const p of points) {
            const px = (p.x / 255) * w;
            const py = (1 - p.y / 255) * h;

            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(px, py, radius, 0, Math.PI * 2);
            ctx.fill();

            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    }
}
