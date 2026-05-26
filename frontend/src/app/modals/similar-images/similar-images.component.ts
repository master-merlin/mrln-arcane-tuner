import {
    ChangeDetectionStrategy,
    Component,
    OnInit,
    computed,
    inject,
    signal,
} from '@angular/core';
import { IcoComponent } from '../../icons/ico.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';
import { RuntimeConfigService } from '../../services/runtime-config.service';

interface SimilarItem {
    path: string;
    score?: number;
    width?: number;
    height?: number;
    isOriginal?: boolean;
}

interface SimilarImagesData {
    datasetId?: string;
    datasetName?: string;
    items?: SimilarItem[];
}

/**
 * Inline SVG placeholder for the broken-thumbnail case. Mirrors the helper
 * used in the Analyze modal so failed loads look intentional across the app.
 */
const THUMB_FALLBACK_DATA_URI =
    'data:image/svg+xml;utf8,' +
    encodeURIComponent(
        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 90" preserveAspectRatio="xMidYMid slice">
            <rect width="160" height="90" fill="oklch(0.14 0.01 265)"/>
            <g stroke="oklch(0.40 0.01 265)" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round" transform="translate(56 30)">
                <rect x="0" y="0" width="48" height="32" rx="3"/>
                <circle cx="14" cy="12" r="3"/>
                <path d="M4 28 L18 17 L30 22 L44 12"/>
            </g>
        </svg>`,
    );

/**
 * Similar Images modal — visualizes a cluster of near-duplicates.
 *
 * Ports the workflow from the orphan
 * [viewer-similar-images-modal](../../components/dataset/dataset-viewer/components/viewer-similar-images-modal.ts).
 * The caller passes the cluster in `modal.data.items`. Each non-original
 * item gets a delete button; deletes flow through `DatasetService.deletePair`
 * and close the modal so the analysis caller can re-run.
 *
 * Design source: `modals-more.jsx → SimilarImagesModal`.
 */
@Component({
    selector: 'app-modal-similar-images',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">SIMILARITY CLUSTERS</div>
                <div class="modal-title">Found {{ duplicateCount() }} potential duplicate{{ duplicateCount() === 1 ? '' : 's' }}</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body si-body">
            @if (items().length === 0) {
                <div class="si-empty">
                    <app-ico name="Search" [size]="18"/>
                    No cluster passed to the modal.
                </div>
            } @else {
                <div class="si-grid">
                    @for (it of items(); track it.path) {
                        <div class="si-card">
                            <div class="si-thumb-wrap">
                                <img class="si-thumb"
                                     [src]="thumbUrl(it.path)"
                                     alt=""
                                     (error)="onThumbError($event)">

                                <div class="si-badges">
                                    @if (it.isOriginal) {
                                        <span class="si-tag brand">ORIGINAL</span>
                                    } @else {
                                        <span class="si-tag warning">SIM {{ ((it.score ?? 0) * 100).toFixed(1) }}%</span>
                                    }
                                    @if (it.width && it.height) {
                                        <span class="si-tag dark mono">{{ it.width }}×{{ it.height }}</span>
                                    }
                                    @if (!it.isOriginal && resCompare(it) === 'higher') {
                                        <span class="si-tag success"><app-ico name="ChevronUp" [size]="9"/> higher res</span>
                                    } @else if (!it.isOriginal && resCompare(it) === 'lower') {
                                        <span class="si-tag danger"><app-ico name="ChevronDown" [size]="9"/> lower res</span>
                                    } @else if (!it.isOriginal && resCompare(it) === 'same') {
                                        <span class="si-tag dark">same res</span>
                                    }
                                </div>

                                <!-- Filename anchored bottom-left in the shared filename-label
                                     style; delete button is a separate bottom-right overlay so
                                     the filename's vertical position is identical on every card,
                                     regardless of whether the row also carries an action button. -->
                                <span class="filename-label si-filename" [title]="it.path">{{ it.path }}</span>
                                @if (!it.isOriginal) {
                                    <button class="icon-btn si-delete"
                                            type="button"
                                            [disabled]="deleting() === it.path"
                                            (click)="deleteOne(it)"
                                            title="Delete similar image">
                                        <app-ico name="Trash2" [size]="12"/>
                                    </button>
                                }
                            </div>
                        </div>
                    }
                </div>
            }
        </div>

        <div class="modal-foot si-foot">
            <span class="muted si-hint">Deleted files move to dataset trash for 30 days.</span>
            <button class="btn ghost" type="button" (click)="overlay.closeModal()">Back to analysis</button>
        </div>
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .si-empty {
            display: flex; align-items: center; gap: 10px;
            padding: 24px; justify-content: center;
            color: var(--color-text-muted); font-size: 13px;
        }
        .si-body { padding: 14px 18px; }

        .si-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
        .si-card {
            border-radius: var(--radius-theme-lg);
            overflow: hidden;
            background: var(--color-base);
            border: 1px solid var(--color-border-subtle);
            box-shadow: var(--shadow-md);
        }
        /* Square thumb area — uses the full 256×256 thumbnail space.
           object-fit:contain so portraits and landscapes both render whole
           without cropping (letterboxed on the short axis against the dark
           surface). */
        .si-thumb-wrap {
            position: relative;
            width: 100%;
            aspect-ratio: 1 / 1;
            background:
                linear-gradient(135deg,
                    oklch(0.12 0.01 265) 0%,
                    oklch(0.16 0.01 265) 100%);
        }
        .si-thumb {
            width: 100%; height: 100%;
            object-fit: contain;
            display: block;
        }

        .si-badges {
            position: absolute; top: 10px; left: 10px;
            display: flex; gap: 6px; align-items: center; flex-wrap: wrap;
            max-width: calc(100% - 20px);
        }
        .si-tag {
            padding: 2px 8px; border-radius: 3px;
            font-size: 10px; font-weight: 800;
            letter-spacing: 0.10em; text-transform: uppercase;
        }
        .si-tag.brand   { background: var(--color-brand);   color: white; }
        .si-tag.warning { background: var(--color-warning); color: oklch(0.18 0.05 75); }
        .si-tag.success { background: color-mix(in oklab, var(--color-success) 80%, transparent); color: white; display: inline-flex; align-items: center; gap: 4px; }
        .si-tag.danger  { background: color-mix(in oklab, var(--color-danger) 80%, transparent); color: white; display: inline-flex; align-items: center; gap: 4px; }
        .si-tag.dark    { background: oklch(0.10 0.01 265 / 0.7); color: var(--color-text-secondary); font-weight: 600; }
        .si-tag.mono { font-family: var(--font-mono); }

        /* Filename label — anchored bottom-left so it's always at the same
           y-coordinate, regardless of whether the card carries a delete button. */
        .si-filename {
            position: absolute;
            left: 10px;
            bottom: 10px;
            max-width: calc(100% - 56px);
        }
        .si-delete {
            position: absolute;
            right: 10px;
            bottom: 10px;
            background: oklch(0.08 0.01 265 / 0.78);
            backdrop-filter: blur(6px);
            border: 1px solid oklch(0.70 0.17 25 / 0.4);
            box-shadow: 0 2px 6px oklch(0 0 0 / 0.55);
            color: var(--color-danger);
            width: 28px; height: 28px;
            border-radius: var(--radius-theme-md);
        }
        .si-delete:hover {
            background: oklch(0.70 0.17 25 / 0.25);
            border-color: oklch(0.70 0.17 25 / 0.7);
        }
        .si-delete:disabled { opacity: 0.5; cursor: not-allowed; }

        .si-foot { display: flex; align-items: center; gap: 10px; }
        .modal-foot .si-hint { margin-right: auto; font-size: 11.5px; color: var(--color-text-muted); }
    `],
})
export class SimilarImagesModalComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    private datasetsApi = inject(DatasetService);
    private toast = inject(ToastService);
    private rtc = inject(RuntimeConfigService);

    protected data: SimilarImagesData = (this.overlay.topModal()?.data as SimilarImagesData) ?? {};

    protected items = signal<SimilarItem[]>([]);
    protected deleting = signal<string | null>(null);

    protected duplicateCount = computed(() =>
        Math.max(0, this.items().filter(i => !i.isOriginal).length),
    );

    private originalMP = computed(() => {
        const orig = this.items().find(i => i.isOriginal);
        if (!orig?.width || !orig?.height) return 0;
        return orig.width * orig.height;
    });

    ngOnInit(): void {
        this.items.set(this.data.items ?? []);
    }

    protected thumbUrl(path: string): string {
        // Use the API thumbnail endpoint instead of the static /media mount —
        // /media only resolves for datasets stored under default_root, and
        // encodeURIComponent on the path turns "/" into "%2F" which the static
        // mount won't decode. The /thumbnail endpoint resolves dataset paths
        // through dataset_manager.get_dataset so it works for every dataset.
        const name = this.data.datasetName ?? '';
        return `${this.rtc.apiUrl}/datasets/${encodeURIComponent(name)}/thumbnail?image_rel_path=${encodeURIComponent(path)}`;
    }

    /** Drop a placeholder in when /thumbnail returns 404 or the decode fails. */
    protected onThumbError(event: Event): void {
        const img = event.target as HTMLImageElement;
        if (img.dataset['fallback'] === '1') return;
        img.dataset['fallback'] = '1';
        img.src = THUMB_FALLBACK_DATA_URI;
    }

    protected resCompare(it: SimilarItem): 'higher' | 'lower' | 'same' {
        const orig = this.originalMP();
        if (orig === 0 || !it.width || !it.height) return 'same';
        const itMP = it.width * it.height;
        const ratio = itMP / orig;
        if (ratio > 1.01) return 'higher';
        if (ratio < 0.99) return 'lower';
        return 'same';
    }

    protected deleteOne(it: SimilarItem): void {
        if (!this.data.datasetName) return;
        if (!confirm(`Delete ${it.path}? This permanently removes the image, caption, and any masks.`)) return;
        this.deleting.set(it.path);
        this.datasetsApi.deletePair(this.data.datasetName, it.path).subscribe({
            next: () => {
                this.toast.success(`Deleted ${it.path}`);
                this.deleting.set(null);
                this.overlay.closeModal();
            },
            error: (err: any) => {
                this.toast.error(`Delete failed: ${err.error?.detail || err.message}`);
                this.deleting.set(null);
            },
        });
    }
}
