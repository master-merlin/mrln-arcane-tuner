import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';
import { IcoComponent } from '../../icons/ico.component';

export type DatasetMode = 'embed' | 'reference' | 'exclude';

export interface ExportCheckItem {
    id: string;
    label: string;
    /** Optional meta subline (e.g. a training template's definition_id or a
     *  caption/mask template's model_id) so same-named rows are distinguishable. */
    sub?: string;
    checked?: boolean;
}

export interface ExportGroup {
    key: string;
    label: string;
    items: ExportCheckItem[];
}

export interface ExportDatasetChoice {
    name: string;
    sizeLabel?: string;
    /** Optional preview thumbnail URL. Falls back to a placeholder icon. */
    thumbUrl?: string;
    mode?: DatasetMode;
}

export interface ExportSelection {
    groups: Record<string, string[]>;
    datasets: { name: string; mode: DatasetMode }[];
}

export interface ExportOptionsData {
    title: string;
    groups?: ExportGroup[];
    datasets?: ExportDatasetChoice[];
    confirmLabel?: string;
    onExport: (selection: ExportSelection) => void;
}

const MODES: DatasetMode[] = ['embed', 'reference', 'exclude'];

@Component({
    selector: 'app-modal-export-options',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">EXPORT</div>
                <div class="modal-title">{{ data().title }}</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body eo-body">
            @for (g of data().groups ?? []; track g.key) {
                @if (g.items.length) {
                    <div class="eo-section">
                        <div class="eo-section-head">
                            <span class="eo-section-title">
                                {{ g.label }}
                                <span class="eo-count mono">{{ checkedCount(g.key) }}/{{ g.items.length }}</span>
                            </span>
                            <span class="eo-bulk">
                                <button type="button" class="eo-link" (click)="setAll(g, true)">All</button>
                                <span class="eo-bulk-sep">·</span>
                                <button type="button" class="eo-link" (click)="setAll(g, false)">None</button>
                            </span>
                        </div>
                        @for (it of g.items; track it.id) {
                            <label class="eo-row">
                                <input type="checkbox" [checked]="isChecked(g.key, it.id)"
                                       (change)="toggle(g.key, it.id)">
                                <span class="eo-row-main">
                                    <span class="eo-row-name">{{ it.label }}</span>
                                    @if (it.sub) { <span class="eo-row-sub mono">{{ it.sub }}</span> }
                                </span>
                            </label>
                        }
                    </div>
                }
            }

            @if ((data().datasets ?? []).length) {
                <div class="eo-section">
                    <div class="eo-section-head">
                        <span class="eo-section-title">
                            Datasets
                            <span class="eo-count mono">{{ (data().datasets ?? []).length }}</span>
                        </span>
                    </div>
                    @for (d of data().datasets ?? []; track d.name) {
                        <div class="eo-ds">
                            <div class="eo-thumb">
                                @if (thumbOk(d.name, d.thumbUrl)) {
                                    <img [src]="d.thumbUrl" alt="" (error)="onThumbError(d.name)">
                                } @else {
                                    <app-ico name="Image" [size]="16"/>
                                }
                            </div>
                            <div class="eo-ds-info">
                                <div class="eo-ds-name">{{ d.name }}</div>
                                @if (d.sizeLabel) { <div class="eo-ds-size mono">{{ d.sizeLabel }}</div> }
                            </div>
                            <div class="seg eo-modes">
                                @for (m of modes; track m) {
                                    <button type="button" [class.active]="modeOf(d.name) === m"
                                            (click)="setMode(d.name, m)">{{ m }}</button>
                                }
                            </div>
                        </div>
                    }
                </div>
            }
        </div>

        <div class="modal-foot eo-foot">
            <span class="eo-summary">{{ summary() }}</span>
            <span class="eo-foot-actions">
                <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
                <button class="btn primary" type="button" (click)="confirm()">
                    <app-ico name="Download" [size]="14"/> {{ data().confirmLabel ?? 'Export' }}
                </button>
            </span>
        </div>
    `,
    styles: [`
        .eo-body { display: flex; flex-direction: column; gap: 4px; }
        .eo-section { margin-bottom: 10px; }
        .eo-section-head {
            position: sticky; top: -18px; z-index: 1;
            display: flex; align-items: center; justify-content: space-between;
            padding: 8px 0 6px;
            margin: 0 -22px; padding-left: 22px; padding-right: 22px;
            background: var(--color-surface-low);
            border-bottom: 1px solid var(--color-border-subtle);
        }
        .eo-section-title {
            font-size: 11px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
            color: var(--color-text-muted); display: inline-flex; align-items: baseline; gap: 8px;
        }
        .eo-count { font-size: 11px; letter-spacing: normal; color: var(--color-text-subtle); }
        .eo-bulk { display: inline-flex; align-items: center; gap: 6px; }
        .eo-bulk-sep { color: var(--color-text-subtle); font-size: 11px; }
        .eo-link {
            background: none; border: none; cursor: pointer; padding: 0;
            font-size: 11px; font-weight: 600; color: var(--color-brand);
        }
        .eo-link:hover { text-decoration: underline; }
        .eo-row { display: flex; align-items: center; gap: 10px; padding: 6px 2px; cursor: pointer; }
        .eo-row:hover { background: var(--color-surface-mid); border-radius: 6px; }
        .eo-row-main { display: flex; align-items: baseline; gap: 10px; min-width: 0; }
        .eo-row-name { font-size: 13px; font-weight: 500; white-space: nowrap; }
        .eo-row-sub { font-size: 11px; color: var(--color-text-muted); white-space: nowrap;
                      overflow: hidden; text-overflow: ellipsis; }
        .eo-ds { display: flex; align-items: center; gap: 12px; padding: 7px 2px; }
        .eo-thumb {
            width: 40px; height: 40px; border-radius: 8px; flex-shrink: 0;
            background: var(--color-surface-mid); border: 1px solid var(--color-border-subtle);
            display: flex; align-items: center; justify-content: center;
            color: var(--color-text-subtle); overflow: hidden;
        }
        .eo-thumb img { width: 100%; height: 100%; object-fit: cover; }
        .eo-ds-info { flex: 1; min-width: 0; }
        .eo-ds-name { font-size: 13px; font-weight: 500; white-space: nowrap;
                      overflow: hidden; text-overflow: ellipsis; }
        .eo-ds-size { font-size: 11px; color: var(--color-text-muted); margin-top: 1px; }
        .eo-modes { flex-shrink: 0; }
        .eo-modes button { text-transform: capitalize; }
        .eo-foot { justify-content: space-between; align-items: center; }
        .eo-summary { font-size: 12px; color: var(--color-text-muted); }
        .eo-foot-actions { display: inline-flex; gap: 8px; }
    `],
})
export class ExportOptionsModalComponent {
    protected overlay = inject(OverlayStore);
    protected modes = MODES;

    protected data = computed<ExportOptionsData>(
        () => (this.overlay.topModal()?.data ?? { title: '', onExport: () => undefined }) as ExportOptionsData,
    );

    private checked = signal<Record<string, Set<string>>>({});
    private modeMap = signal<Record<string, DatasetMode>>({});
    private failedThumbs = signal<Set<string>>(new Set());

    constructor() {
        const d = this.data();
        const chk: Record<string, Set<string>> = {};
        for (const g of d.groups ?? []) {
            chk[g.key] = new Set(g.items.filter((i) => i.checked).map((i) => i.id));
        }
        this.checked.set(chk);
        const modes: Record<string, DatasetMode> = {};
        for (const ds of d.datasets ?? []) modes[ds.name] = ds.mode ?? 'reference';
        this.modeMap.set(modes);
    }

    /** Footer summary — selected template count + included (non-excluded) datasets. */
    protected summary = computed<string>(() => {
        const d = this.data();
        let tpl = 0;
        for (const g of d.groups ?? []) tpl += this.checked()[g.key]?.size ?? 0;
        const parts = [`${tpl} template${tpl === 1 ? '' : 's'}`];
        const datasets = d.datasets ?? [];
        if (datasets.length) {
            const included = datasets.filter((x) => this.modeOf(x.name) !== 'exclude').length;
            parts.push(`${included} dataset${included === 1 ? '' : 's'}`);
        }
        return parts.join(' · ');
    });

    isChecked(groupKey: string, id: string): boolean {
        return this.checked()[groupKey]?.has(id) ?? false;
    }

    checkedCount(groupKey: string): number {
        return this.checked()[groupKey]?.size ?? 0;
    }

    toggle(groupKey: string, id: string): void {
        this.checked.update((state) => {
            const next = { ...state };
            const set = new Set(next[groupKey] ?? []);
            if (set.has(id)) set.delete(id);
            else set.add(id);
            next[groupKey] = set;
            return next;
        });
    }

    /** Check or clear every item in a group (Select all / none). */
    setAll(group: ExportGroup, value: boolean): void {
        this.checked.update((state) => ({
            ...state,
            [group.key]: value ? new Set(group.items.map((i) => i.id)) : new Set<string>(),
        }));
    }

    modeOf(name: string): DatasetMode {
        return this.modeMap()[name] ?? 'reference';
    }

    setMode(name: string, mode: DatasetMode): void {
        this.modeMap.update((state) => ({ ...state, [name]: mode }));
    }

    thumbOk(name: string, url?: string): boolean {
        return !!url && !this.failedThumbs().has(name);
    }

    onThumbError(name: string): void {
        this.failedThumbs.update((s) => new Set(s).add(name));
    }

    confirm(): void {
        const d = this.data();
        const groups: Record<string, string[]> = {};
        for (const g of d.groups ?? []) {
            groups[g.key] = [...(this.checked()[g.key] ?? [])];
        }
        const datasets = (d.datasets ?? []).map((ds) => ({ name: ds.name, mode: this.modeOf(ds.name) }));
        d.onExport({ groups, datasets });
        this.overlay.closeModal();
    }
}
