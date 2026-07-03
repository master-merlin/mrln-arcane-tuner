import {
    ChangeDetectionStrategy, Component, computed, inject, signal,
} from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { IcoComponent } from '../../icons/ico.component';
import {
    DatasetService, type DatasetPair, type VideoSegment, type VideoSplitMode,
} from '../../services/dataset';
import { ToastService } from '../../services/toast';
import { OverlayStore } from '../../state/overlay.store';
import { SegmentPreviewTableComponent } from '../../components/dataset/video/segment-preview-table';

type Step = 'pick' | 'review';

/** Payload passed via `overlay.openModal('cutlist-import', …)`. */
export interface CutlistImportData {
    /** HTTP name of the dataset being split. */
    datasetName: string;
    /** All pairs in the workspace — filtered to videos for the source picker. */
    videoPairs?: DatasetPair[];
}

/**
 * Cut-list import modal — turns a Final-Cut / CSV / TSV cut list into a clip
 * split.
 *
 * Step 1 (pick): choose a source video + upload the cut-list file, then
 * `parseCutlist` (synchronous) yields segments + warnings.
 * Step 2 (review): preview the segments, pick a split `mode`, set an
 * `output_prefix` + `archive_source`, then `splitVideo` enqueues a
 * `video_split` background task and the modal closes — the Task Center +
 * the backend `dataset.invalidated` broadcast drive the grid refresh, so
 * this modal is fire-and-forget.
 *
 * Registered in modal-layer and opened via `OverlayStore.openModal('cutlist-import', …)`;
 * modal-layer owns the backdrop / `.modal` chrome, so this component renders only
 * the dialog body. Inputs arrive through the overlay payload ({@link CutlistImportData}).
 */
@Component({
    selector: 'app-cutlist-import-modal',
    standalone: true,
    imports: [IcoComponent, SegmentPreviewTableComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head" data-testid="cutlist-import-modal">
                    <div>
                        <div class="eyebrow">CUT LIST IMPORT</div>
                        <div class="modal-title">Split a video into clips from a cut list</div>
                    </div>
                    <button class="icon-btn" type="button" (click)="close()" aria-label="Close">×</button>
                </div>

                <div class="modal-body cl-body">
                    @if (videoPairs().length === 0) {
                        <div class="cl-empty">
                            <app-ico name="Info" [size]="16"/>
                            No videos in this dataset — add a source video first.
                        </div>
                    } @else if (step() === 'pick') {
                        <section class="cl-section">
                            <label class="field-label" for="cl-src">Source video</label>
                            <select id="cl-src" class="cl-select"
                                    data-testid="cutlist-source-select"
                                    [value]="sourceRel() ?? ''"
                                    (change)="sourceRel.set($any($event.target).value || null)">
                                <option value="" disabled>Choose a video…</option>
                                @for (v of videoPairs(); track v.media_file) {
                                    <option [value]="v.media_file">{{ v.media_file }}</option>
                                }
                            </select>
                        </section>

                        <section class="cl-section">
                            <label class="field-label" for="cl-file">Cut-list file (.llc / .csv / .tsv)</label>
                            <input id="cl-file" type="file"
                                   data-testid="cutlist-file-input"
                                   accept=".llc,.csv,.tsv"
                                   (change)="onFile($event)">
                            @if (fileName()) { <div class="cl-filename mono">{{ fileName() }}</div> }
                        </section>

                        @if (parseError()) {
                            <div class="cl-error" data-testid="cutlist-parse-error">
                                <app-ico name="TriangleAlert" [size]="14"/> {{ parseError() }}
                            </div>
                        }
                    } @else {
                        <section class="cl-section">
                            <div class="cl-review-head">
                                <span class="eyebrow">{{ format() || 'PARSED' }} · {{ segments().length }} segments</span>
                            </div>
                            @if (warnings().length) {
                                <div class="cl-warnings" data-testid="cutlist-warnings">
                                    @for (w of warnings(); track w) {
                                        <div class="cl-warn"><app-ico name="TriangleAlert" [size]="12"/> {{ w }}</div>
                                    }
                                </div>
                            }
                            <app-segment-preview-table
                                [segments]="segments()"
                                [fps]="sourceFps()"/>
                        </section>

                        <section class="cl-section">
                            <span class="eyebrow">SPLIT MODE</span>
                            <div class="cl-modes">
                                @for (m of modes; track m.id) {
                                    <button type="button" class="cl-mode"
                                            [attr.data-testid]="'cutlist-mode-' + m.id"
                                            [class.active]="mode() === m.id"
                                            (click)="mode.set(m.id)">
                                        <div class="cl-mode-title">{{ m.label }}</div>
                                        <div class="cl-mode-hint">{{ m.hint }}</div>
                                    </button>
                                }
                            </div>
                        </section>

                        <section class="cl-section cl-options">
                            <div class="cl-opt">
                                <label class="field-label" for="cl-prefix">Output prefix (optional)</label>
                                <input id="cl-prefix" type="text" class="cl-text"
                                       data-testid="cutlist-prefix"
                                       placeholder="clip"
                                       [value]="outputPrefix()"
                                       (input)="outputPrefix.set($any($event.target).value)">
                            </div>
                            <label class="cl-check">
                                <input type="checkbox"
                                       data-testid="cutlist-archive"
                                       [checked]="archiveSource()"
                                       (change)="archiveSource.set($any($event.target).checked)">
                                Archive the source video after splitting
                            </label>
                        </section>
                    }
                </div>

                <div class="modal-foot">
                    @if (step() === 'review') {
                        <button class="btn ghost" type="button"
                                data-testid="cutlist-back"
                                (click)="step.set('pick')">Back</button>
                    }
                    <button class="btn ghost" type="button" (click)="close()">Cancel</button>
                    @if (step() === 'pick') {
                        <button class="btn primary" type="button"
                                data-testid="cutlist-parse-btn"
                                [disabled]="!canParse() || parsing()"
                                (click)="parse()">
                            <app-ico name="ScanLine" [size]="13"/>
                            {{ parsing() ? 'Parsing…' : 'Parse' }}
                        </button>
                    } @else {
                        <button class="btn primary" type="button"
                                data-testid="cutlist-split-btn"
                                [disabled]="segments().length === 0 || splitting()"
                                (click)="split()">
                            <app-ico name="Scissors" [size]="13"/> Split into clips
                        </button>
                    }
                </div>
    `,
    styles: [`
        :host { display: contents; }
        .modal-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .cl-body { display: flex; flex-direction: column; gap: 18px; }
        .cl-section { display: flex; flex-direction: column; gap: 8px; }
        .field-label { font-size: 12px; font-weight: 600; color: var(--color-text-secondary); }
        .cl-select, .cl-text {
            width: 100%; padding: 8px 10px; font-size: 13px;
            background: var(--color-surface-mid); color: var(--color-text-primary);
            border: 1px solid var(--color-border-default); border-radius: var(--radius-theme-md);
        }
        .cl-filename { font-size: 11px; color: var(--color-text-muted); }
        .cl-empty, .cl-error {
            display: flex; align-items: center; gap: 8px;
            padding: 14px; font-size: 12.5px;
        }
        .cl-empty { color: var(--color-text-muted); justify-content: center; }
        .cl-error {
            color: var(--color-danger);
            background: color-mix(in oklab, var(--color-danger) 8%, transparent);
            border: 1px solid color-mix(in oklab, var(--color-danger) 25%, transparent);
            border-radius: var(--radius-theme-md);
        }
        .cl-warnings { display: flex; flex-direction: column; gap: 4px; margin-bottom: 8px; }
        .cl-warn {
            display: flex; align-items: center; gap: 6px;
            font-size: 11px; color: var(--color-warning);
        }
        .cl-modes { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
        .cl-mode {
            text-align: left; padding: 10px 12px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-default);
            border-radius: var(--radius-theme-lg); cursor: pointer;
        }
        .cl-mode.active {
            border-color: var(--color-brand);
            background: color-mix(in oklab, var(--color-brand) 8%, var(--color-surface-mid));
        }
        .cl-mode-title { font-size: 12.5px; font-weight: 700; color: var(--color-text-primary); }
        .cl-mode-hint { font-size: 10px; color: var(--color-text-subtle); margin-top: 2px; line-height: 1.4; }
        .cl-options { gap: 12px; }
        .cl-check {
            display: flex; align-items: center; gap: 8px;
            font-size: 12px; color: var(--color-text-secondary); cursor: pointer;
        }
    `],
})
export class CutlistImportModalComponent {
    private overlay = inject(OverlayStore);
    private api = inject(DatasetService);
    private toast = inject(ToastService);

    private data = (this.overlay.topModal()?.data ?? {}) as CutlistImportData;
    /** HTTP name of the dataset being split. */
    protected datasetName = signal<string>(this.data.datasetName ?? '');
    /** All pairs in the workspace — filtered to videos for the source picker. */
    protected videoPairs = signal<DatasetPair[]>(this.data.videoPairs ?? []);

    protected readonly modes: ReadonlyArray<{ id: VideoSplitMode; label: string; hint: string }> = [
        { id: 'auto',     label: 'Auto',     hint: 'Copy where keyframes align, re-encode the rest.' },
        { id: 'copy',     label: 'Copy',     hint: 'Fast, no re-encode. Cuts snap to keyframes.' },
        { id: 'reencode', label: 'Re-encode', hint: 'Frame-exact cuts. Slowest, highest quality.' },
    ];

    protected step = signal<Step>('pick');
    protected sourceRel = signal<string | null>(null);
    protected file = signal<File | null>(null);
    protected fileName = signal<string>('');
    protected parsing = signal<boolean>(false);
    protected parseError = signal<string>('');

    protected segments = signal<VideoSegment[]>([]);
    protected format = signal<string>('');
    protected warnings = signal<string[]>([]);

    protected mode = signal<VideoSplitMode>('auto');
    protected outputPrefix = signal<string>('');
    protected archiveSource = signal<boolean>(false);
    protected splitting = signal<boolean>(false);

    protected canParse = computed<boolean>(() => !!this.sourceRel() && !!this.file());

    /** fps of the chosen source — feeds the est-frame column. */
    protected sourceFps = computed<number | undefined>(() => {
        const rel = this.sourceRel();
        const v = this.videoPairs().find(p => p.media_file === rel);
        return v?.metadata?.fps;
    });

    protected onFile(event: Event): void {
        const f = (event.target as HTMLInputElement).files?.[0] ?? null;
        this.file.set(f);
        this.fileName.set(f?.name ?? '');
        this.parseError.set('');
    }

    protected async parse(): Promise<void> {
        const name = this.datasetName();
        const src = this.sourceRel();
        const f = this.file();
        if (!name || !src || !f) return;
        this.parsing.set(true);
        this.parseError.set('');
        try {
            const res = await firstValueFrom(this.api.parseCutlist(name, f, src));
            this.segments.set(res.segments ?? []);
            this.format.set(res.format ?? '');
            this.warnings.set(res.warnings ?? []);
            if ((res.segments ?? []).length === 0) {
                this.parseError.set('No segments found in the cut list.');
            } else {
                this.step.set('review');
            }
        } catch (err: unknown) {
            this.parseError.set(this.errMsg(err, 'Could not parse the cut list'));
        } finally {
            this.parsing.set(false);
        }
    }

    protected split(): void {
        const name = this.datasetName();
        const src = this.sourceRel();
        if (!name || !src || this.segments().length === 0) return;
        this.splitting.set(true);
        this.api.splitVideo(name, {
            source_rel_path: src,
            segments: this.segments(),
            mode: this.mode(),
            output_prefix: this.outputPrefix().trim() || null,
            archive_source: this.archiveSource(),
        }).subscribe({
            next: () => {
                this.toast.success('Splitting in background — see the Task Center.');
                this.close();
            },
            error: (err: unknown) => {
                this.splitting.set(false);
                this.toast.error(this.errMsg(err, 'Could not start the split'));
            },
        });
    }

    protected close(): void {
        this.overlay.closeModal();
    }

    private errMsg(err: unknown, fallback: string): string {
        const e = err as { error?: { detail?: string }; message?: string } | null;
        const detail = e?.error?.detail || e?.message;
        return detail ? `${fallback}: ${detail}` : fallback;
    }
}
