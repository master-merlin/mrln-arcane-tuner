import {
    ChangeDetectionStrategy, Component, OnDestroy, computed, inject, signal,
} from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { IcoComponent } from '../../icons/ico.component';
import {
    DatasetService, type DatasetPair, type VideoSegment,
} from '../../services/dataset';
import { ToastService } from '../../services/toast';
import { OverlayStore } from '../../state/overlay.store';
import { SegmentPreviewTableComponent } from '../../components/dataset/video/segment-preview-table';

type Step = 'config' | 'detecting' | 'review';

/** Payload passed via `overlay.openModal('scene-detect', …)`. */
export interface SceneDetectData {
    /** HTTP name of the dataset. */
    datasetName: string;
    /** Workspace pairs — filtered to videos for the source picker. */
    videoPairs?: DatasetPair[];
}

/**
 * Scene-detect modal — detects scene cuts in a source video and splits on them.
 *
 * Step 1 (config): pick a source video + `threshold` + `min_scene_len_s`,
 * then `sceneDetect` enqueues a `scene_detect` background task.
 * Step 2 (detecting): poll `getSceneProposals` until `ready` (auto every 2s,
 * plus a manual "Check results" button). Polling is cleaned up on destroy.
 * Step 3 (review): curate the proposed segments (delete / merge via the
 * editable preview table) then confirm → `splitVideo` (mode `auto`) enqueues
 * a `video_split` task and the modal closes — the Task Center takes over.
 *
 * Registered in modal-layer and opened via `OverlayStore.openModal('scene-detect', …)`;
 * modal-layer owns the backdrop / `.modal` chrome, so this component renders only
 * the dialog body. Inputs arrive through the overlay payload ({@link SceneDetectData}).
 */
@Component({
    selector: 'app-scene-detect-modal',
    standalone: true,
    imports: [IcoComponent, SegmentPreviewTableComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head" data-testid="scene-detect-modal">
                    <div>
                        <div class="eyebrow">SCENE DETECT</div>
                        <div class="modal-title">Auto-detect scene cuts and split into clips</div>
                    </div>
                    <button class="icon-btn" type="button" (click)="close()"
                            data-testid="scene-detect-close" aria-label="Close">×</button>
                </div>

                <div class="modal-body sd-body">
                    @if (videoPairs().length === 0) {
                        <div class="sd-empty">
                            <app-ico name="Info" [size]="16"/>
                            No videos in this dataset — add a source video first.
                        </div>
                    } @else if (step() === 'config') {
                        <section class="sd-section">
                            <label class="field-label" for="sd-src">Source video</label>
                            <select id="sd-src" class="sd-select"
                                    data-testid="scene-source-select"
                                    [value]="sourceRel() ?? ''"
                                    (change)="sourceRel.set($any($event.target).value || null)">
                                <option value="" disabled>Choose a video…</option>
                                @for (v of videoPairs(); track v.media_file) {
                                    <option [value]="v.media_file">{{ v.media_file }}</option>
                                }
                            </select>
                        </section>

                        <section class="sd-section">
                            <div class="sd-slider-head">
                                <label class="field-label" for="sd-thr">Threshold</label>
                                <span class="mono">{{ threshold().toFixed(1) }}</span>
                            </div>
                            <input id="sd-thr" type="range" min="5" max="60" step="0.5"
                                   data-testid="scene-threshold"
                                   [value]="threshold()"
                                   (input)="threshold.set(+$any($event.target).value)">
                            <p class="sd-hint">Lower = more cuts (sensitive); higher = fewer cuts.</p>
                        </section>

                        <section class="sd-section">
                            <div class="sd-slider-head">
                                <label class="field-label" for="sd-min">Min scene length</label>
                                <span class="mono">{{ minSceneLen().toFixed(1) }}s</span>
                            </div>
                            <input id="sd-min" type="range" min="0" max="10" step="0.25"
                                   data-testid="scene-min-len"
                                   [value]="minSceneLen()"
                                   (input)="minSceneLen.set(+$any($event.target).value)">
                        </section>

                        @if (errorMsg()) {
                            <div class="sd-error" data-testid="scene-error">
                                <app-ico name="TriangleAlert" [size]="14"/> {{ errorMsg() }}
                            </div>
                        }
                    } @else if (step() === 'detecting') {
                        <div class="sd-detecting" data-testid="scene-detecting">
                            <app-ico name="Loader2" [size]="20"/>
                            <div class="sd-detecting-txt">Detecting scenes…</div>
                            <button class="btn sm" type="button"
                                    data-testid="scene-check-btn"
                                    (click)="checkResults()">Check results</button>
                        </div>
                    } @else {
                        <section class="sd-section">
                            <span class="eyebrow">{{ segments().length }} scenes proposed</span>
                            <app-segment-preview-table
                                [segments]="segments()"
                                [fps]="sourceFps()"
                                [editable]="true"
                                (segmentsChange)="segments.set($event)"/>
                        </section>
                    }
                </div>

                <div class="modal-foot">
                    <button class="btn ghost" type="button" (click)="close()"
                            data-testid="scene-detect-cancel">Cancel</button>
                    @if (step() === 'config') {
                        <button class="btn primary" type="button"
                                data-testid="scene-detect-btn"
                                [disabled]="!sourceRel() || detecting()"
                                (click)="detect()">
                            <app-ico name="ScanSearch" [size]="13"/>
                            {{ detecting() ? 'Starting…' : 'Detect scenes' }}
                        </button>
                    } @else if (step() === 'review') {
                        <button class="btn ghost" type="button"
                                data-testid="scene-redetect"
                                (click)="step.set('config')">Re-detect</button>
                        <button class="btn primary" type="button"
                                data-testid="scene-split-btn"
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
        .sd-body { display: flex; flex-direction: column; gap: 18px; }
        .sd-section { display: flex; flex-direction: column; gap: 8px; }
        .field-label { font-size: 12px; font-weight: 600; color: var(--color-text-secondary); }
        .sd-select {
            width: 100%; padding: 8px 10px; font-size: 13px;
            background: var(--color-surface-mid); color: var(--color-text-primary);
            border: 1px solid var(--color-border-default); border-radius: var(--radius-theme-md);
        }
        .sd-slider-head { display: flex; align-items: center; justify-content: space-between; }
        .sd-hint { font-size: 10px; color: var(--color-text-subtle); }
        .sd-empty, .sd-error {
            display: flex; align-items: center; gap: 8px; padding: 14px; font-size: 12.5px;
        }
        .sd-empty { color: var(--color-text-muted); justify-content: center; }
        .sd-error {
            color: var(--color-danger);
            background: color-mix(in oklab, var(--color-danger) 8%, transparent);
            border: 1px solid color-mix(in oklab, var(--color-danger) 25%, transparent);
            border-radius: var(--radius-theme-md);
        }
        .sd-detecting {
            display: flex; flex-direction: column; align-items: center; gap: 12px;
            padding: 28px; color: var(--color-text-muted);
        }
        .sd-detecting app-ico { color: var(--color-brand); }
        .sd-detecting-txt { font-size: 13px; }
    `],
})
export class SceneDetectModalComponent implements OnDestroy {
    private overlay = inject(OverlayStore);
    private api = inject(DatasetService);
    private toast = inject(ToastService);

    private data = (this.overlay.topModal()?.data ?? {}) as SceneDetectData;
    /** HTTP name of the dataset. */
    protected datasetName = signal<string>(this.data.datasetName ?? '');
    /** Workspace pairs — filtered to videos for the source picker. */
    protected videoPairs = signal<DatasetPair[]>(this.data.videoPairs ?? []);

    protected step = signal<Step>('config');
    protected sourceRel = signal<string | null>(null);
    protected threshold = signal<number>(27);
    protected minSceneLen = signal<number>(1);
    protected detecting = signal<boolean>(false);
    protected errorMsg = signal<string>('');

    protected segments = signal<VideoSegment[]>([]);
    protected splitting = signal<boolean>(false);

    /** Auto-poll handle; cleared on destroy or once proposals are ready. */
    private pollHandle: ReturnType<typeof setInterval> | null = null;

    protected sourceFps = computed<number | undefined>(() => {
        const rel = this.sourceRel();
        const v = this.videoPairs().find(p => p.media_file === rel);
        return v?.metadata?.fps;
    });

    ngOnDestroy(): void {
        this.stopPolling();
    }

    protected detect(): void {
        const name = this.datasetName();
        const src = this.sourceRel();
        if (!name || !src) return;
        this.detecting.set(true);
        this.errorMsg.set('');
        this.api.sceneDetect(name, {
            source_rel_path: src,
            threshold: this.threshold(),
            min_scene_len_s: this.minSceneLen(),
        }).subscribe({
            next: () => {
                this.detecting.set(false);
                this.step.set('detecting');
                this.startPolling();
            },
            error: (err: unknown) => {
                this.detecting.set(false);
                this.errorMsg.set(this.errMsg(err, 'Could not start scene detection'));
            },
        });
    }

    private startPolling(): void {
        this.stopPolling();
        // Poll every 2s; `setInterval` is fine under zoneless — we drive the
        // signal directly and OnPush re-checks on the signal write.
        this.pollHandle = setInterval(() => void this.checkResults(), 2000);
    }

    private stopPolling(): void {
        if (this.pollHandle != null) {
            clearInterval(this.pollHandle);
            this.pollHandle = null;
        }
    }

    /** Fetch the latest proposals; advance to review once `ready`. */
    protected async checkResults(): Promise<void> {
        const name = this.datasetName();
        const src = this.sourceRel();
        if (!name || !src) return;
        try {
            const res = await firstValueFrom(this.api.getSceneProposals(name, src));
            if (res.ready) {
                this.stopPolling();
                this.segments.set(res.segments ?? []);
                this.step.set('review');
            }
        } catch {
            // Transient — the next poll (or manual check) retries.
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
            mode: 'auto',
            output_prefix: null,
            archive_source: false,
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
        this.stopPolling();
        this.overlay.closeModal();
    }

    private errMsg(err: unknown, fallback: string): string {
        const e = err as { error?: { detail?: string }; message?: string } | null;
        const detail = e?.error?.detail || e?.message;
        return detail ? `${fallback}: ${detail}` : fallback;
    }
}
