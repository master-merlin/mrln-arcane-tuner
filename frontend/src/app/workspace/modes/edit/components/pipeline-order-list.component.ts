import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { UpperCasePipe } from '@angular/common';
import { PipelineEditorState } from '../pipeline-editor.state';
import { OperationKind } from '../operation-defs';

interface Row {
    kind: OperationKind;
    label: string;
    enabled: boolean;
    summary: string;
}

@Component({
    selector: 'app-pipeline-order-list',
    standalone: true,
    imports: [UpperCasePipe],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="hint mono">execution sequence</div>
        <ul class="list" (dragover)="$event.preventDefault()">
            @for (row of rows(); track row.kind; let i = $index) {
                <li class="row"
                    draggable="true"
                    [class.dragging]="dragIndex() === i"
                    [class.drag-over]="dragOver() === i"
                    [class.enabled]="row.enabled"
                    (dragstart)="onDragStart(i, $event)"
                    (dragenter)="onDragEnter(i)"
                    (dragover)="$event.preventDefault()"
                    (drop)="onDrop(i)"
                    (dragend)="onDragEnd()">
                    <span class="idx mono">{{ (i + 1).toString().padStart(2, '0') }}</span>
                    <span class="grip" title="Drag to reorder">⋮⋮</span>
                    <label class="check" (click)="$event.stopPropagation()">
                        <input type="checkbox" [checked]="row.enabled"
                               (change)="toggle(row.kind, $any($event.target).checked)"/>
                    </label>
                    <span class="label">{{ row.label }}</span>
                    @if (row.summary) {
                        <span class="summary mono">{{ row.summary }}</span>
                    }
                </li>
            }
        </ul>

        @if (state.colorMatch().enabled && state.colorMatch().params.reference_path) {
            <div class="cm-chip" title="Always applied first (not reorderable)">
                <span class="cm-label">Color Match</span>
                <span class="cm-method mono">
                    {{ state.colorMatch().params.method | uppercase }}
                    · {{ (state.colorMatch().params.strength * 100).toFixed(0) }}%
                </span>
            </div>
        }
    `,
    styles: [`
        :host { display: block; }
        .hint {
            font-size: 9.5px; color: var(--color-text-subtle);
            text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;
        }
        .list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 2px; }
        .row {
            display: grid;
            grid-template-columns: 18px 12px 18px 1fr auto;
            align-items: center; gap: 8px;
            padding: 4px 8px;
            border-radius: var(--radius-theme-sm);
            border: 1px solid transparent;
            font-size: 11.5px;
            color: var(--color-text-muted);
            cursor: grab;
        }
        .row.enabled {
            background: var(--color-surface-mid);
            border-color: var(--color-border-subtle);
            color: var(--color-text-primary);
            font-weight: 500;
        }
        .row.dragging { opacity: 0.45; }
        .row.drag-over { border-color: var(--color-brand); background: color-mix(in oklab, var(--color-brand) 8%, transparent); }
        .idx { font-size: 10px; color: var(--color-text-subtle); width: 18px; text-align: center; user-select: none; }
        .grip { color: var(--color-text-subtle); font-size: 11px; user-select: none; }
        .check input { accent-color: var(--color-brand); }
        .summary { font-size: 9.5px; color: var(--color-text-subtle); }
        .row.enabled .summary { color: var(--color-brand); }
        .cm-chip {
            margin-top: 10px; padding: 6px 10px;
            border: 1px dashed color-mix(in oklab, var(--color-brand) 40%, transparent);
            border-radius: var(--radius-theme-sm);
            background: color-mix(in oklab, var(--color-brand) 6%, transparent);
            font-size: 11px;
            display: flex; align-items: center; gap: 8px;
        }
        .cm-label { font-weight: 600; color: var(--color-brand); }
        .cm-method { color: var(--color-text-muted); font-size: 10.5px; margin-left: auto; }
    `],
})
export class PipelineOrderListComponent {
    protected state = inject(PipelineEditorState);

    protected dragIndex = signal<number | null>(null);
    protected dragOver = signal<number | null>(null);

    protected rows = computed<Row[]>(() => {
        return this.state.operationOrder().map(kind => {
            const op = this.opFor(kind);
            return {
                kind,
                label: LABELS[kind],
                enabled: op.enabled,
                summary: this.summarize(kind, op),
            };
        });
    });

    private opFor(kind: OperationKind) {
        switch (kind) {
            case 'white_balance':  return this.state.whiteBalance();
            case 'curves':         return this.state.curves();
            case 'lut':            return this.state.lut();
            case 'hsl_selective':  return this.state.hslSelective();
            case 'color_tone':     return this.state.colorTone();
            case 'vignette':       return this.state.vignette();
            case 'lens':           return this.state.lens();
            case 'sharpen':        return this.state.sharpen();
            case 'denoise':        return this.state.denoise();
            case 'face_restore':   return this.state.faceRestore();
            case 'upscale':        return this.state.upscale();
            case 'color_match':    return this.state.colorMatch();
        }
    }

    private summarize(kind: OperationKind, op: { enabled: boolean; params: any }): string {
        if (!op.enabled) return '';
        const p = op.params;
        switch (kind) {
            case 'white_balance':  return `${p.temperature}K`;
            case 'curves':         return 'active';
            case 'lut':            return p.luts?.length ? `${p.luts.length} LUT(s)` : 'empty';
            case 'hsl_selective':  return Object.keys(p).length ? `${Object.keys(p).length} band(s)` : '';
            case 'color_tone': {
                const parts: string[] = [];
                if (p.hue_shift !== 0) parts.push(`H${p.hue_shift > 0 ? '+' : ''}${p.hue_shift}`);
                if (p.saturation !== 1) parts.push(`S${p.saturation.toFixed(2)}`);
                if (p.contrast !== 1) parts.push(`C${p.contrast.toFixed(2)}`);
                return parts.join(' ') || 'active';
            }
            case 'vignette':       return p.amount?.toFixed?.(2) ?? '';
            case 'lens':           return (p.barrel || p.v_keystone || p.h_keystone) ? 'active' : '';
            case 'sharpen':        return p.method ?? '';
            case 'denoise':        return p.model ? `${(p.strength * 100).toFixed(0)}%` : 'no model';
            case 'face_restore':   return p.model ? `${(p.strength * 100).toFixed(0)}%` : 'no model';
            case 'upscale':        return p.model ? `${p.target_scale}×` : 'no model';
            default:               return '';
        }
    }

    toggle(kind: OperationKind, enabled: boolean): void {
        this.state.setEnabled(kind, enabled);
    }

    onDragStart(i: number, e: DragEvent): void {
        this.dragIndex.set(i);
        if (e.dataTransfer) e.dataTransfer.effectAllowed = 'move';
    }
    onDragEnter(i: number): void { this.dragOver.set(i); }
    onDrop(i: number): void {
        const from = this.dragIndex();
        if (from === null || from === i) return;
        this.state.moveOperation(from, i);
        this.dragIndex.set(null);
        this.dragOver.set(null);
    }
    onDragEnd(): void { this.dragIndex.set(null); this.dragOver.set(null); }
}

const LABELS: Record<OperationKind, string> = {
    denoise: 'Denoise',
    face_restore: 'Face Restore',
    white_balance: 'White Balance',
    curves: 'Curves',
    lut: 'CUBE LUT',
    color_match: 'Color Match',
    hsl_selective: 'HSL Selective',
    color_tone: 'Color & Tone',
    vignette: 'Vignette',
    lens: 'Lens Correction',
    sharpen: 'Sharpening',
    upscale: 'Upscale',
};
