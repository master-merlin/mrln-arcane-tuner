import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { firstValueFrom } from 'rxjs';

import { IcoComponent } from '../../icons/ico.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetStore } from '../../state/dataset.store';
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';

type Transport = 'file' | 'path';
type ConflictChoice = 'rename' | 'overwrite';

/**
 * Import Dataset modal.
 *
 * Two-mode dialog. The default state offers a transport toggle — upload a
 * `.zip` from the browser, or point at a `.zip` already on the server
 * filesystem. On a name collision the backend returns HTTP 409 with
 * `err.error.detail = { conflict, name, message }`; the modal then flips
 * into a collision-prompt offering Rename / Overwrite / Cancel.
 */
@Component({
    selector: 'app-modal-import-dataset',
    standalone: true,
    imports: [FormsModule, IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">IMPORT</div>
                <div class="modal-title">Import Dataset</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body">
            @if (conflictName(); as cname) {
                <div class="conflict">
                    <p>A dataset named <strong>{{ cname }}</strong> already exists. What would you like to do?</p>
                    <label class="field">
                        <span>New name (for Rename)</span>
                        <input type="text" [(ngModel)]="renameValue" [placeholder]="cname + ' (imported)'">
                    </label>
                </div>
            } @else {
                <div class="seg">
                    <button type="button" [class.on]="transport() === 'file'" (click)="transport.set('file')">
                        <app-ico name="Upload" [size]="14"/> Upload ZIP
                    </button>
                    <button type="button" [class.on]="transport() === 'path'" (click)="transport.set('path')">
                        <app-ico name="HardDrive" [size]="14"/> Server path
                    </button>
                </div>

                @if (transport() === 'file') {
                    <label class="dropzone" [class.dz-drag]="dragOver()" [class.dz-has]="!!file()"
                           (dragover)="onDragOver($event)" (dragleave)="onDragLeave()" (drop)="onDrop($event)">
                        <app-ico [name]="file() ? 'FileCheck' : 'FileUp'" [size]="26"/>
                        <div class="dz-title">
                            @if (file(); as f) { {{ f.name }} }
                            @else { Drop a <strong>.zip</strong> here, or click to browse }
                        </div>
                        <div class="dz-sub">{{ file() ? 'Click or drop to replace' : 'dataset archive (.zip)' }}</div>
                        <input type="file" accept=".zip" hidden (change)="onFile($event)">
                    </label>
                } @else {
                    <label class="field">
                        <span>Server path to .zip</span>
                        <input type="text" [(ngModel)]="serverPath" placeholder="D:/exports/Portraits_1.2.0.zip">
                    </label>
                }
            }
        </div>

        <div class="modal-foot">
            <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
            @if (conflictName()) {
                <button class="btn ghost" type="button" [disabled]="submitting()" (click)="resolve('rename')">Rename</button>
                <button class="btn primary" type="button" [disabled]="submitting()" (click)="resolve('overwrite')">Overwrite</button>
            } @else {
                <button class="btn primary" type="button" [disabled]="submitting() || !canSubmit()" (click)="submit()">
                    {{ submitting() ? 'Importing…' : 'Import' }}
                </button>
            }
        </div>
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .seg { display: flex; gap: 8px; margin-bottom: 14px; }
        .seg button {
            flex: 1;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            padding: 8px;
            border-radius: 8px;
            border: 1px solid var(--color-border, var(--color-border-subtle));
            background: var(--color-surface-input, var(--color-surface-low));
            color: var(--color-text-muted);
            cursor: pointer;
        }
        .seg button.on {
            color: var(--color-text);
            outline: 2px solid var(--color-brand, #6aa0ff);
        }
        .field { display: flex; flex-direction: column; gap: 6px; margin-bottom: 12px; }
        .field span { font-size: 11px; color: var(--color-text-muted); }
        .field input[type="text"] {
            padding: 8px;
            border: 1px solid var(--color-border, var(--color-border-subtle));
            background: var(--color-surface-input, var(--color-surface-low));
            color: inherit;
            border-radius: 6px;
        }
        .conflict p { margin: 0 0 12px; }
    `],
})
export class ImportDatasetModalComponent {
    protected overlay = inject(OverlayStore);
    private datasets = inject(DatasetStore);
    private api = inject(DatasetService);
    private toast = inject(ToastService);

    protected transport = signal<Transport>('file');
    protected serverPath = '';
    protected renameValue = '';
    protected submitting = signal(false);

    protected file = signal<File | null>(null);
    protected conflictName = signal<string | null>(null);
    protected dragOver = signal(false);

    protected onFile(event: Event): void {
        const files = (event.target as HTMLInputElement).files;
        this.setFile(files && files.length ? files[0] : null);
    }

    protected onDragOver(event: DragEvent): void {
        event.preventDefault();
        this.dragOver.set(true);
    }

    protected onDragLeave(): void {
        this.dragOver.set(false);
    }

    protected onDrop(event: DragEvent): void {
        event.preventDefault();
        this.dragOver.set(false);
        const f = event.dataTransfer?.files?.[0] ?? null;
        this.setFile(f);
    }

    /** Accept a dropped/picked file, rejecting anything that isn't a `.zip`. */
    private setFile(f: File | null): void {
        if (f && !f.name.toLowerCase().endsWith('.zip')) {
            this.toast.warning('Please choose a .zip archive.');
            return;
        }
        this.file.set(f);
    }

    protected canSubmit(): boolean {
        return this.transport() === 'file' ? !!this.file() : this.serverPath.trim().length > 0;
    }

    protected async submit(): Promise<void> {
        await this.run(undefined, undefined);
    }

    protected async resolve(choice: ConflictChoice): Promise<void> {
        await this.run(choice, choice === 'rename' ? this.renameValue.trim() || undefined : undefined);
    }

    private async run(onConflict?: ConflictChoice, newName?: string): Promise<void> {
        this.submitting.set(true);
        try {
            const result$ =
                this.transport() === 'file'
                    ? this.api.importDatasetFile(this.file()!, onConflict, newName)
                    : this.api.importDatasetPath(this.serverPath.trim(), onConflict, newName);
            const ds = await firstValueFrom(result$);
            this.toast.success(`Imported "${ds.name}".`);
            await this.datasets.loadAll().catch(() => undefined);
            this.overlay.closeModal();
        } catch (err: unknown) {
            const e = err as { status?: number; message?: string; error?: { detail?: unknown } };
            const detail = e?.error?.detail;
            const detailObj = detail && typeof detail === 'object'
                ? detail as { conflict?: boolean; name?: string; message?: string }
                : null;
            if (e?.status === 409 && detailObj?.conflict) {
                // Switch the modal into collision-prompt mode and let the user choose.
                this.conflictName.set(detailObj.name ?? '');
            } else {
                const msg = detailObj?.message || (typeof detail === 'string' ? detail : null) || e?.message || 'unknown error';
                this.toast.error('Import failed: ' + msg);
            }
        } finally {
            this.submitting.set(false);
        }
    }
}
