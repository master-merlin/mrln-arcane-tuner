import {
    ChangeDetectionStrategy,
    Component,
    computed,
    inject,
    signal,
} from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';

/**
 * Open with:
 *   overlay.openModal('version-edit', {
 *     datasetName: ds.name,
 *     currentVersion: ds.version,
 *     onSaved: (v) => …, // optional — workspace uses it for upsertLocal
 *   });
 *
 * Fixup R parity addition — the legacy frontend carried a TODO to let
 * users hand-edit the version when an accidental bump pushed it off.
 * The click-on-tag MAJOR bump stays as the primary affordance; this
 * modal sits behind the small pencil button next to the version tag
 * and serves the rare "I just promoted my dataset to 5.0.0 by accident"
 * recovery path.
 *
 * Modal self-contains the HTTP call + toast so future callers (e.g. a
 * future mass-version-correction tool) can open it without re-wiring
 * the error path. The optional ``onSaved`` callback lets the caller
 * keep its own local cache in sync without an extra HTTP fetch.
 */
export interface VersionEditModalData {
    /** HTTP name of the dataset whose version is being edited. */
    datasetName: string;
    /** Current version string — pre-fills the input. */
    currentVersion: string;
    /** Optional success callback (workspace uses it to upsertLocal). */
    onSaved?: (newVersion: string) => void;
}

const SEMVER_RE = /^\d+\.\d+\.\d+$/;

@Component({
    selector: 'app-modal-version-edit',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div class="modal-title">Edit version</div>
            <button class="icon-btn" type="button" (click)="cancel()" aria-label="Close">×</button>
        </div>
        <div class="modal-body">
            <p class="ve-hint-block">
                Manually overwrite the dataset version. Use this to recover
                from an accidental bump — for example, if "Click to bump to
                next MAJOR" was triggered by mistake.
            </p>
            <label class="ve-label" for="ve-version-input">Version</label>
            <input id="ve-version-input"
                   class="ve-input"
                   type="text"
                   inputmode="decimal"
                   autocomplete="off"
                   [disabled]="inFlight()"
                   [value]="versionInput()"
                   (input)="onInput($event)"
                   (keydown.enter)="save()"
                   placeholder="1.2.3"/>
            @if (errorMessage(); as e) {
                <div class="ve-error">{{ e }}</div>
            } @else {
                <div class="ve-format-hint" [class.ve-format-hint-ok]="isValidFormat()">
                    @if (isValidFormat()) {
                        Format OK
                    } @else {
                        Format: X.Y.Z (e.g. 1.2.3)
                    }
                </div>
            }
        </div>
        <div class="modal-foot">
            <button class="btn ghost" type="button"
                    [disabled]="inFlight()"
                    (click)="cancel()">
                Cancel
            </button>
            <button class="btn primary"
                    type="button"
                    [disabled]="!isValid() || inFlight()"
                    (click)="save()">
                {{ inFlight() ? 'Saving…' : 'Save' }}
            </button>
        </div>
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; }
        .ve-hint-block {
            color: var(--color-text-secondary);
            font-size: 12.5px;
            line-height: 1.55;
            margin: 0 0 14px 0;
        }
        .ve-label {
            display: block;
            font-size: 11px;
            font-weight: 600;
            color: var(--color-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 6px;
        }
        .ve-input {
            width: 100%;
            box-sizing: border-box;
            padding: 8px 10px;
            font-family: var(--font-mono, monospace);
            font-size: 14px;
            border: 1px solid var(--color-border);
            border-radius: var(--radius-theme-sm);
            background: var(--color-bg-input, var(--color-bg-elevated));
            color: var(--color-text-primary);
        }
        .ve-input:focus {
            outline: none;
            border-color: var(--color-brand);
        }
        .ve-format-hint {
            margin-top: 6px;
            font-size: 11.5px;
            color: var(--color-text-muted);
        }
        .ve-format-hint-ok { color: var(--color-success); }
        .ve-error {
            margin-top: 6px;
            font-size: 12px;
            color: var(--color-danger);
        }
    `],
})
export class VersionEditModalComponent {
    private overlay = inject(OverlayStore);
    private datasetsApi = inject(DatasetService);
    private toast = inject(ToastService);

    protected data = computed<VersionEditModalData>(
        () => (this.overlay.topModal()?.data ?? {
            datasetName: '',
            currentVersion: '',
        }) as VersionEditModalData,
    );

    protected versionInput = signal<string>(this.data().currentVersion ?? '');
    protected inFlight = signal<boolean>(false);
    protected errorMessage = signal<string | null>(null);

    protected isValidFormat = computed<boolean>(
        () => SEMVER_RE.test(this.versionInput()),
    );

    /**
     * Save is enabled only when input is well-formed semver AND
     * different from the current version — no point hitting the API
     * to set the version to what it already is.
     */
    protected isValid = computed<boolean>(
        () => this.isValidFormat() && this.versionInput() !== this.data().currentVersion,
    );

    protected onInput(event: Event): void {
        const v = (event.target as HTMLInputElement).value ?? '';
        this.versionInput.set(v);
        // Clear stale backend error as the user keeps typing.
        if (this.errorMessage()) this.errorMessage.set(null);
    }

    protected async save(): Promise<void> {
        if (!this.isValid() || this.inFlight()) return;
        const d = this.data();
        const target = this.versionInput();
        this.inFlight.set(true);
        this.errorMessage.set(null);
        try {
            const res = await firstValueFrom(
                this.datasetsApi.setVersion(d.datasetName, target),
            );
            d.onSaved?.(res.version);
            this.toast.success(`Version set to ${res.version}`);
            this.overlay.closeModal();
        } catch (err: unknown) {
            const e = err as { error?: { detail?: string }; message?: string };
            const msg = e?.error?.detail ?? e?.message ?? 'Failed to set version';
            this.errorMessage.set(msg);
            this.toast.error(`Failed to set version: ${msg}`);
            this.inFlight.set(false);
        }
    }

    protected cancel(): void {
        if (this.inFlight()) return;
        this.overlay.closeModal();
    }
}
