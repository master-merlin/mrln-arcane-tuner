import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';
import { IcoComponent } from '../../icons/ico.component';

export type DatasetMode = 'embed' | 'reference' | 'exclude';

export interface ExportCheckItem {
    id: string;
    label: string;
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

        <div class="modal-body">
            @for (g of data().groups ?? []; track g.key) {
                <div class="eo-group">
                    <div class="eo-group-label">{{ g.label }}</div>
                    @for (it of g.items; track it.id) {
                        <label class="eo-check">
                            <input type="checkbox" [checked]="isChecked(g.key, it.id)"
                                   (change)="toggle(g.key, it.id)">
                            <span>{{ it.label }}</span>
                        </label>
                    }
                </div>
            }

            @if ((data().datasets ?? []).length) {
                <div class="eo-group">
                    <div class="eo-group-label">Datasets</div>
                    @for (d of data().datasets ?? []; track d.name) {
                        <div class="eo-ds">
                            <div class="eo-ds-name">
                                {{ d.name }}
                                @if (d.sizeLabel) { <span class="eo-ds-size">{{ d.sizeLabel }}</span> }
                            </div>
                            <div class="seg eo-modes">
                                @for (m of modes; track m) {
                                    <button type="button" [class.on]="modeOf(d.name) === m"
                                            (click)="setMode(d.name, m)">{{ m }}</button>
                                }
                            </div>
                        </div>
                    }
                </div>
            }
        </div>

        <div class="modal-foot">
            <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
            <button class="btn primary" type="button" (click)="confirm()">
                <app-ico name="Download" [size]="14"/> {{ data().confirmLabel ?? 'Export' }}
            </button>
        </div>
    `,
    styles: [`
        .eo-group { margin-bottom: 14px; }
        .eo-group-label { font-size: 12px; opacity: .7; text-transform: uppercase; margin-bottom: 6px; }
        .eo-check { display: flex; align-items: center; gap: 8px; padding: 4px 0; cursor: pointer; }
        .eo-ds { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 6px 0; }
        .eo-ds-size { opacity: .6; font-size: 12px; margin-left: 6px; }
        .eo-modes button { text-transform: capitalize; }
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

    isChecked(groupKey: string, id: string): boolean {
        return this.checked()[groupKey]?.has(id) ?? false;
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

    modeOf(name: string): DatasetMode {
        return this.modeMap()[name] ?? 'reference';
    }

    setMode(name: string, mode: DatasetMode): void {
        this.modeMap.update((state) => ({ ...state, [name]: mode }));
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
