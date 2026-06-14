import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';
import { IcoComponent } from '../../../icons/ico.component';
import type { VideoSegment } from '../../../services/dataset';
import { FRAME_FAMILIES, estimateFrames, passesFamily } from './frame-rules';

/** One rendered row — the segment plus its derived duration / est-frame view. */
interface SegmentRow {
    seg: VideoSegment;
    index: number;
    duration: number;
    frames: number;
    /** Per-family pass flags, in {@link FRAME_FAMILIES} order. */
    family: boolean[];
}

/**
 * Shared, pure preview table for a list of cut {@link VideoSegment}s.
 *
 * Renders start / end / duration / (est-frames + family chips) / label rows.
 * When `editable` is on it supports deleting a row and merging a row into its
 * predecessor (the scene-detect curation path); every edit re-emits the whole
 * list through {@link segmentsChange} so the parent stays the source of truth.
 *
 * Stateless: the component never mutates the input array — it builds a new
 * list and emits it, so the parent's signal drives re-render.
 */
@Component({
    selector: 'app-segment-preview-table',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @if (rows().length === 0) {
            <div class="spt-empty" data-testid="spt-empty">No segments.</div>
        } @else {
            <table class="spt" data-testid="segment-preview-table">
                <thead>
                    <tr>
                        <th class="num">#</th>
                        <th>Start</th>
                        <th>End</th>
                        <th>Duration</th>
                        @if (showFrames()) { <th>Frames</th> }
                        <th>Label</th>
                        @if (editable()) { <th class="act"></th> }
                    </tr>
                </thead>
                <tbody>
                    @for (r of rows(); track r.index) {
                        <tr data-testid="spt-row">
                            <td class="num mono">{{ r.index + 1 }}</td>
                            <td class="mono">{{ fmt(r.seg.start_s) }}</td>
                            <td class="mono">{{ fmt(r.seg.end_s) }}</td>
                            <td class="mono">{{ fmt(r.duration) }}</td>
                            @if (showFrames()) {
                                <td class="frames">
                                    <span class="mono" data-testid="spt-frames">{{ r.frames || '—' }}</span>
                                    @if (r.frames > 0) {
                                        <span class="chips">
                                            @for (f of families; track f.label; let i = $index) {
                                                <span class="chip"
                                                      [class.pass]="r.family[i]"
                                                      [class.fail]="!r.family[i]"
                                                      [title]="f.label + (r.family[i] ? ' OK' : ' fails')">
                                                    {{ f.label }}
                                                </span>
                                            }
                                        </span>
                                    }
                                </td>
                            }
                            <td class="label" [title]="r.seg.label ?? ''">{{ r.seg.label ?? '—' }}</td>
                            @if (editable()) {
                                <td class="act">
                                    @if (r.index > 0) {
                                        <button type="button" class="ico-btn"
                                                data-testid="spt-merge"
                                                title="Merge into previous segment"
                                                (click)="merge(r.index)">
                                            <app-ico name="ArrowUpToLine" [size]="13"/>
                                        </button>
                                    }
                                    <button type="button" class="ico-btn danger"
                                            data-testid="spt-delete"
                                            title="Delete segment"
                                            (click)="remove(r.index)">
                                        <app-ico name="Trash2" [size]="13"/>
                                    </button>
                                </td>
                            }
                        </tr>
                    }
                </tbody>
            </table>
        }
    `,
    styles: [`
        :host { display: block; }
        .spt-empty {
            padding: 16px; text-align: center;
            color: var(--color-text-muted); font-size: 12px;
        }
        .spt { width: 100%; border-collapse: collapse; font-size: 12px; }
        .spt th {
            text-align: left; padding: 6px 8px;
            font-size: 9.5px; font-weight: 700; letter-spacing: 0.10em;
            text-transform: uppercase; color: var(--color-text-subtle);
            border-bottom: 1px solid var(--color-border-default);
        }
        .spt td {
            padding: 6px 8px; color: var(--color-text-secondary);
            border-bottom: 1px solid var(--color-border-subtle);
            vertical-align: middle;
        }
        .spt .num { width: 28px; color: var(--color-text-subtle); }
        .spt .label { color: var(--color-text-primary); max-width: 160px;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .spt .act { width: 1%; white-space: nowrap; text-align: right; }
        .frames .chips { display: inline-flex; gap: 4px; margin-left: 8px; }
        .chip {
            font-size: 9px; font-weight: 700; letter-spacing: 0.04em;
            padding: 1px 5px; border-radius: 999px;
            border: 1px solid var(--color-border-subtle);
        }
        .chip.pass {
            color: var(--color-success);
            background: color-mix(in oklab, var(--color-success) 12%, transparent);
            border-color: color-mix(in oklab, var(--color-success) 35%, transparent);
        }
        .chip.fail {
            color: var(--color-text-muted);
            background: var(--color-surface-mid);
        }
        .ico-btn {
            display: inline-flex; align-items: center; justify-content: center;
            width: 24px; height: 24px; border-radius: var(--radius-theme-md);
            border: 1px solid var(--color-border-subtle);
            background: var(--color-surface-mid); color: var(--color-text-muted);
            cursor: pointer;
        }
        .ico-btn:hover { color: var(--color-text-primary); }
        .ico-btn.danger:hover { color: var(--color-danger);
            border-color: color-mix(in oklab, var(--color-danger) 40%, transparent); }
        .ico-btn + .ico-btn { margin-left: 4px; }
    `],
})
export class SegmentPreviewTableComponent {
    /** The segments to render. */
    segments = input.required<VideoSegment[]>();
    /** Source fps — drives the est-frame-count column. Omit to hide frames. */
    fps = input<number | undefined>(undefined);
    /** When true, renders per-row delete + merge controls. */
    editable = input<boolean>(false);

    /** Re-emitted full segment list after a delete / merge edit. */
    segmentsChange = output<VideoSegment[]>();

    protected readonly families = FRAME_FAMILIES;

    /** Frames column is shown only when a usable fps was provided. */
    protected showFrames = computed<boolean>(() => {
        const f = this.fps();
        return !!f && f > 0;
    });

    protected rows = computed<SegmentRow[]>(() => {
        const fps = this.fps();
        return this.segments().map((seg, index) => {
            const frames = estimateFrames(seg.start_s, seg.end_s, fps);
            return {
                seg,
                index,
                duration: Math.max(0, seg.end_s - seg.start_s),
                frames,
                family: this.families.map(f => passesFamily(frames, f)),
            };
        });
    });

    /** mm:ss.s formatter for the time columns. */
    protected fmt(s: number): string {
        if (!Number.isFinite(s) || s < 0) return '0:00.0';
        const m = Math.floor(s / 60);
        const sec = s - m * 60;
        return `${m}:${sec.toFixed(1).padStart(4, '0')}`;
    }

    protected remove(index: number): void {
        const next = this.segments().filter((_, i) => i !== index);
        this.segmentsChange.emit(next);
    }

    /** Merge `index` into its predecessor: predecessor's end extends to this
     *  row's end; this row is dropped. No-op for the first row. */
    protected merge(index: number): void {
        if (index <= 0) return;
        const list = this.segments();
        const prev = list[index - 1];
        const cur = list[index];
        const merged: VideoSegment = {
            start_s: prev.start_s,
            end_s: cur.end_s,
            label: prev.label ?? cur.label ?? null,
        };
        const next = list.slice();
        next.splice(index - 1, 2, merged);
        this.segmentsChange.emit(next);
    }
}
