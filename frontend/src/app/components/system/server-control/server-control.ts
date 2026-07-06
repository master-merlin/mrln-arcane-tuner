import { Component, inject, signal, OnInit, effect, ChangeDetectionStrategy } from '@angular/core';
import { ToastService } from '../../../services/toast';
import { FormsModule } from '@angular/forms';
import { ModelService, type ModelGlobalSettings, type ModelGlobalSettingsPatch } from '../../../services/model.service';
import { FilesystemService } from '../../../services/filesystem.service';
import { SettingsStore } from '../../../state/settings.store';
import { IcoComponent } from '../../../icons/ico.component';

interface ApplicationSettings {
    backend_port: number;
    frontend_port: number;
    log_level: string;
    start_frontend: boolean;
}

/**
 * Server settings form, redesigned to the Hi-Fi `ScreenServer` layout
 * (audit 09 Theme-F): two DS `.card`s — Connection (ports / log level /
 * auto-start) and Models (default path / offline mode) — using the global
 * design-system classes instead of the legacy bespoke Tailwind.
 *
 * The internal page header + restart button were removed: the server-screen
 * page-head owns the title and the Restart action now (it drives the global
 * SystemControlService), which also resolves the old duplicate-restart-button
 * note. Clear-logs moved to the live-log-viewer header per the Hi-Fi. All
 * settings logic (store mirroring, immediate vs staged saves, folder picker)
 * is unchanged.
 */
@Component({
    selector: 'app-server-control',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [FormsModule, IcoComponent],
    template: `
    <div class="server-settings-grid">
        <!-- Connection -->
        <div class="card">
            <div class="card-head"><div class="card-title"><app-ico name="Server" [size]="11" /> Connection</div></div>
            <div class="card-body sc-body">
                <div class="sc-grid-2">
                    <div class="sc-field">
                        <label class="field-label">Backend Port</label>
                        <input type="number" class="input mono"
                               [ngModel]="settings()?.backend_port"
                               (ngModelChange)="onSettingChange('backend_port', $event)"
                               data-testid="setting-backend-port" />
                        <p class="sc-hint">requires restart</p>
                    </div>
                    <div class="sc-field">
                        <label class="field-label">Frontend Port</label>
                        <input type="number" class="input mono"
                               [ngModel]="settings()?.frontend_port"
                               (ngModelChange)="onSettingChange('frontend_port', $event)"
                               data-testid="setting-frontend-port" />
                        <p class="sc-hint">requires restart</p>
                    </div>
                </div>
                <div class="sc-grid-2">
                    <div class="sc-field">
                        <label class="field-label">Log Level</label>
                        <select class="select mono"
                                [ngModel]="settings()?.log_level"
                                (ngModelChange)="onSettingChange('log_level', $event)"
                                data-testid="setting-log-level">
                            @for (level of logLevels; track level) {
                                <option [value]="level">{{ level }}</option>
                            }
                        </select>
                        <p class="sc-hint">applied immediately</p>
                    </div>
                    <div class="sc-field">
                        <label class="field-label">Auto-Start Frontend</label>
                        <label class="sc-toggle-row" data-testid="setting-start-frontend">
                            <button type="button" class="toggle" [class.on]="settings()?.start_frontend"
                                    (click)="onToggleChange('start_frontend', !settings()?.start_frontend)"
                                    role="switch" [attr.aria-checked]="settings()?.start_frontend"></button>
                            <span class="sc-toggle-label">{{ settings()?.start_frontend ? 'Enabled' : 'Disabled' }}</span>
                        </label>
                        <p class="sc-hint">launch dev server on cold start</p>
                    </div>
                </div>
                @if (settingsDirty()) {
                    <div class="sc-save-row">
                        <span class="chip warning"><span class="dot"></span> Unsaved changes</span>
                        <button type="button" class="btn primary sm" (click)="saveSettings()" [disabled]="isSavingSettings()">
                            {{ isSavingSettings() ? 'Saving…' : 'Save Settings' }}
                        </button>
                    </div>
                }
            </div>
        </div>

        <!-- Models -->
        <div class="card">
            <div class="card-head"><div class="card-title"><app-ico name="Box" [size]="11" /> Models</div></div>
            <div class="card-body sc-body">
                <div class="sc-field">
                    <label class="field-label">Default Model Path</label>
                    <div class="sc-path-row">
                        <input type="text" class="input mono"
                               [ngModel]="modelSettings()?.default_model_path"
                               (ngModelChange)="pendingModelPath.set($event); modelPathDirty.set(true)"
                               placeholder="D:\\Models" data-testid="setting-default-model-path" />
                        <button type="button" class="btn sm" (click)="browseModelPath()"
                                [disabled]="browsingModelPath()" title="Browse for folder">
                            <app-ico name="Folder" [size]="13" />
                        </button>
                    </div>
                    @if (modelPathDirty()) {
                        <div class="sc-save-row">
                            <button type="button" class="btn primary sm" (click)="saveModelPath()">Save Path</button>
                        </div>
                    }
                    <p class="sc-hint">base directory for model storage — used as the initial browse location</p>
                </div>
                <div class="sc-field">
                    <label class="field-label">Global Offline Mode</label>
                    <label class="sc-toggle-row" data-testid="setting-global-offline">
                        <button type="button" class="toggle" [class.on]="modelSettings()?.global_offline_mode"
                                (click)="onModelSettingToggle('global_offline_mode', !modelSettings()?.global_offline_mode)"
                                role="switch" [attr.aria-checked]="modelSettings()?.global_offline_mode"></button>
                        <span class="sc-toggle-label">{{ modelSettings()?.global_offline_mode ? 'Offline' : 'Online' }}</span>
                    </label>
                    <p class="sc-hint">block ALL HuggingFace network requests — use only locally cached models</p>
                </div>
                <div class="sc-field">
                    <label class="field-label">Hugging Face Token</label>
                    <div class="sc-path-row">
                        <input type="password" class="input mono"
                               [ngModel]="pendingHfToken()"
                               (ngModelChange)="pendingHfToken.set($event)"
                               [placeholder]="modelSettings()?.hf_token_set ? '•••••••••••• (token saved)' : 'hf_…'"
                               autocomplete="off" data-testid="setting-hf-token" />
                        <button type="button" class="btn primary sm" (click)="saveHfToken()"
                                [disabled]="!pendingHfToken().trim()">Save</button>
                        @if (modelSettings()?.hf_token_set) {
                            <button type="button" class="btn sm" (click)="clearHfToken()"
                                    data-testid="clear-hf-token">Clear</button>
                        }
                    </div>
                    <p class="sc-hint">authenticates downloads of gated models (e.g. some FLUX weights). Stored server-side; an <code>HF_TOKEN</code> environment variable, if set, takes precedence.</p>
                </div>
            </div>
        </div>
    </div>
  `,
    styles: [`
        .server-settings-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
        @media (max-width: 1100px) { .server-settings-grid { grid-template-columns: 1fr; } }
        .sc-body { display: flex; flex-direction: column; gap: 14px; }
        .sc-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .sc-field { display: flex; flex-direction: column; min-width: 0; }
        .sc-hint { font-size: 10.5px; color: var(--color-text-disabled); margin: 5px 0 0; }
        .sc-toggle-row { display: flex; align-items: center; gap: 8px; padding: 6px 0; cursor: pointer; }
        .sc-toggle-label { font-size: 12px; color: var(--color-text-secondary); }
        .sc-path-row { display: flex; gap: 8px; }
        .sc-path-row .input { flex: 1; font-size: 11.5px; }
        .sc-save-row { display: flex; align-items: center; justify-content: flex-end; gap: 10px; }
    `],
})
export class ServerControlComponent implements OnInit {
    private toast = inject(ToastService);
    private modelService = inject(ModelService);
    private filesystem = inject(FilesystemService);
    private settingsStore = inject(SettingsStore);

    constructor() {
        // Mirror SettingsStore's `application` module into the local
        // `settings` signal. The store is canonical (cross-tab updates
        // arrive via entity.changed:settings); local `settings` carries
        // pending edit state via `pendingChanges` that the store doesn't
        // model, so we only overwrite the signal when the store row
        // changes — never clobbering the user's in-flight edits beyond
        // what the store has already persisted.
        effect(() => {
            const row = this.settingsStore.byId('application')();
            if (!row) return;
            this.settings.set(row.settings as unknown as ApplicationSettings);
        });
    }

    settings = signal<ApplicationSettings | null>(null);
    settingsDirty = signal(false);
    isSavingSettings = signal(false);

    logLevels = ['DEBUG', 'INFO', 'WARNING', 'ERROR'];

    // Pending (unsaved) values
    private pendingChanges: Record<string, unknown> = {};

    // Model settings
    modelSettings = signal<ModelGlobalSettings | null>(null);
    pendingModelPath = signal('');
    modelPathDirty = signal(false);
    browsingModelPath = signal(false);
    pendingHfToken = signal('');

    ngOnInit() {
        this.loadSettings();
        this.loadModelSettings();
    }

    loadSettings() {
        // Seed the store; the effect above mirrors the resulting row into
        // the local `settings` signal.
        void this.settingsStore.loadModule('application')
            .then(() => {
                this.pendingChanges = {};
                this.settingsDirty.set(false);
            })
            .catch((err) => console.error('Failed to fetch settings', err));
    }

    onSettingChange(key: keyof ApplicationSettings, value: string | number) {
        // Log level is applied immediately (hot-swap) through the store.
        // The store toasts on failure and rolls the merge back; success
        // toast remains a custom message so we fire it eagerly.
        if (key === 'log_level') {
            this.toast.success(`Log level set to ${value}`);
            void this.settingsStore.updateModule('application', { [key]: value });
            return;
        }

        // For port settings, stage the change locally — saveSettings will
        // flush them through the store.
        this.pendingChanges[key] = value;
        this.settingsDirty.set(true);
        // Update local display
        const cur = this.settings();
        if (cur) this.settings.set({ ...cur, [key]: value } as ApplicationSettings);
    }

    onToggleChange(key: keyof ApplicationSettings, value: boolean) {
        // Apply toggle settings immediately through the store.
        this.toast.success(`${key.replace(/_/g, ' ')} ${value ? 'enabled' : 'disabled'}`);
        void this.settingsStore.updateModule('application', { [key]: value });
    }

    saveSettings() {
        if (!this.settings()) return;
        if (Object.keys(this.pendingChanges).length === 0) {
            this.settingsDirty.set(false);
            return;
        }
        this.isSavingSettings.set(true);

        // Push only the diff — the backend merges into the existing
        // module dict, and SettingsStore mirrors that merge locally.
        const diff = { ...this.pendingChanges };
        this.pendingChanges = {};
        this.settingsDirty.set(false);
        this.toast.success('Settings saved. Port changes require a restart to take effect.');

        void this.settingsStore.updateModule('application', diff)
            .finally(() => this.isSavingSettings.set(false));
    }

    // ── Model Settings ──────────────────────────────────────────────────

    loadModelSettings() {
        this.modelService.getModelSettings().subscribe({
            next: (s) => {
                this.modelSettings.set(s);
                this.pendingModelPath.set(s.default_model_path);
                this.modelPathDirty.set(false);
            },
            error: () => console.error('Failed to load model settings'),
        });
    }

    onModelSettingToggle(key: string, value: boolean) {
        this.modelService.updateModelSettings({ [key]: value } as ModelGlobalSettingsPatch).subscribe({
            next: (s) => {
                this.modelSettings.set(s);
                this.toast.success(`${key === 'global_offline_mode' ? 'Offline mode' : key} ${value ? 'enabled' : 'disabled'}`);
            },
            error: (err) => this.toast.error(err.error?.detail || 'Failed to update model settings'),
        });
    }

    browseModelPath() {
        this.browsingModelPath.set(true);
        this.filesystem.pickFolder(this.pendingModelPath() || '', 'Select Default Model Directory').subscribe({
            next: (res) => {
                this.browsingModelPath.set(false);
                if (res.path) {
                    this.pendingModelPath.set(res.path);
                    this.modelPathDirty.set(true);
                }
            },
            error: () => {
                this.browsingModelPath.set(false);
                this.toast.error('Folder picker failed');
            },
        });
    }

    saveModelPath() {
        this.modelService.updateModelSettings({
            default_model_path: this.pendingModelPath(),
        }).subscribe({
            next: (s) => {
                this.modelSettings.set(s);
                this.modelPathDirty.set(false);
                this.toast.success('Default model path saved');
            },
            error: (err) => this.toast.error(err.error?.detail || 'Failed to save model path'),
        });
    }

    // ── Hugging Face token (write-only; the raw value is never fetched back) ──

    saveHfToken() {
        const token = this.pendingHfToken().trim();
        if (!token) return;
        this.modelService.updateModelSettings({
            hf_token: token,
        }).subscribe({
            next: (s) => {
                this.modelSettings.set(s);
                this.pendingHfToken.set('');
                this.toast.success('Hugging Face token saved');
            },
            error: (err) => this.toast.error(err.error?.detail || 'Failed to save token'),
        });
    }

    clearHfToken() {
        this.modelService.updateModelSettings({
            hf_token: '',
        }).subscribe({
            next: (s) => {
                this.modelSettings.set(s);
                this.pendingHfToken.set('');
                this.toast.success('Hugging Face token cleared');
            },
            error: (err) => this.toast.error(err.error?.detail || 'Failed to clear token'),
        });
    }
}
