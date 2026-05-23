import { Component, inject, signal, output, OnInit, effect } from '@angular/core';
import { ToastService } from '../../../services/toast';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { LiveLogViewerComponent } from '../live-log-viewer/live-log-viewer';
import { SettingsStore } from '../../../state/settings.store';

interface ApplicationSettings {
    backend_port: number;
    frontend_port: number;
    log_level: string;
    start_frontend: boolean;
}

interface ModelGlobalSettings {
    global_offline_mode: boolean;
    default_model_path: string;
}

@Component({
    selector: 'app-server-control',
    standalone: true,
    imports: [LiveLogViewerComponent, FormsModule],
    template: `
    <div class="space-y-8 animate-in fade-in duration-500">
      
        <!-- Header + Power Controls -->
        <div class="bg-surface-low/50 border border-border-default rounded-theme-xl p-8 shadow-xl">
            <div class="flex items-start justify-between">
                <div>
                    <h2 class="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-brand"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect><rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect><line x1="6" y1="6" x2="6.01" y2="6"></line><line x1="6" y1="18" x2="6.01" y2="18"></line></svg>
                        Server Control
                    </h2>
                    <p class="text-text-muted">Manage the backend server instance, settings, and logs.</p>
                </div>
                <button (click)="onRestart()" 
                    [disabled]="isActionPending()"
                    class="group flex items-center gap-2 px-4 py-2.5 rounded-theme-xl bg-surface-mid border border-border-default hover:border-red-500/50 hover:bg-surface-mid/80 transition-all text-sm font-medium text-text-secondary hover:text-red-400 shrink-0">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-text-subtle group-hover:text-red-400 transition-colors"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path><path d="M3 3v5h5"></path><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"></path><path d="M16 16h5v5"></path></svg>
                    Restart Server
                </button>
            </div>
        </div>

        <!-- Content Card -->
        <div class="bg-surface-low border border-surface-mid rounded-theme-xl shadow-2xl p-6 space-y-8">

            <!-- System Settings -->
            <div class="space-y-4">
                <h3 class="text-lg font-medium text-white flex items-center gap-2">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-brand">
                        <circle cx="12" cy="12" r="3"></circle>
                        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                    </svg>
                    System Settings
                </h3>

                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <!-- Backend Port -->
                    <div class="p-4 rounded-theme-lg bg-base/50 border border-surface-high space-y-2">
                        <label class="text-[10px] uppercase text-text-subtle font-bold tracking-wider block">Backend Port</label>
                        <input type="number" [ngModel]="settings()?.backend_port" (ngModelChange)="onSettingChange('backend_port', $event)"
                            class="w-full bg-surface-mid text-sm text-white border border-surface-high rounded-theme-lg px-3 py-2 focus:outline-none focus:border-brand font-mono"
                            data-testid="setting-backend-port">
                        <p class="text-[10px] text-text-disabled">Requires restart to take effect</p>
                    </div>

                    <!-- Frontend Port -->
                    <div class="p-4 rounded-theme-lg bg-base/50 border border-surface-high space-y-2">
                        <label class="text-[10px] uppercase text-text-subtle font-bold tracking-wider block">Frontend Port</label>
                        <input type="number" [ngModel]="settings()?.frontend_port" (ngModelChange)="onSettingChange('frontend_port', $event)"
                            class="w-full bg-surface-mid text-sm text-white border border-surface-high rounded-theme-lg px-3 py-2 focus:outline-none focus:border-brand font-mono"
                            data-testid="setting-frontend-port">
                        <p class="text-[10px] text-text-disabled">Requires restart to take effect</p>
                    </div>

                    <!-- Log Level -->
                    <div class="p-4 rounded-theme-lg bg-base/50 border border-surface-high space-y-2">
                        <label class="text-[10px] uppercase text-text-subtle font-bold tracking-wider block">Log Level</label>
                        <select [ngModel]="settings()?.log_level" (ngModelChange)="onSettingChange('log_level', $event)"
                            class="w-full bg-surface-high text-sm text-white border border-surface-high/50 rounded-theme-lg px-3 py-2 focus:outline-none focus:border-brand font-mono"
                            data-testid="setting-log-level">
                            @for (level of logLevels; track level) {
                                <option [value]="level">{{ level }}</option>
                            }
                        </select>
                        <p class="text-[10px] text-text-disabled">Applied immediately</p>
                    </div>

                    <!-- Auto-Start Frontend -->
                    <div class="p-4 rounded-theme-lg bg-base/50 border border-surface-high space-y-2">
                        <label class="text-[10px] uppercase text-text-subtle font-bold tracking-wider block">Auto-Start Frontend</label>
                        <label class="relative inline-flex items-center cursor-pointer mt-1"
                               data-testid="setting-start-frontend">
                            <input type="checkbox" [ngModel]="settings()?.start_frontend" (ngModelChange)="onToggleChange('start_frontend', $event)"
                                class="sr-only peer">
                            <div class="w-11 h-6 bg-surface-high rounded-full peer peer-checked:bg-brand/60 transition-colors
                                        after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all
                                        peer-checked:after:translate-x-full"></div>
                            <span class="ml-3 text-sm text-text-secondary">{{ settings()?.start_frontend ? 'Enabled' : 'Disabled' }}</span>
                        </label>
                        <p class="text-[10px] text-text-disabled">Launch Angular dev server and open browser on cold start</p>
                    </div>
                </div>

                @if (settingsDirty()) {
                    <div class="flex items-center justify-end gap-3 pt-2 animate-in fade-in duration-200">
                        <span class="text-xs text-amber-400 flex items-center gap-1.5">
                            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                            Unsaved changes
                        </span>
                        <button (click)="saveSettings()" [disabled]="isSavingSettings()"
                            class="px-4 py-2 bg-brand/20 text-brand hover:bg-brand/30 rounded-theme-lg transition-colors text-xs font-bold border border-brand/30 flex items-center gap-1.5 disabled:opacity-40">
                            @if (isSavingSettings()) {
                                <svg class="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>
                                Saving...
                            } @else {
                                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
                                Save Settings
                            }
                        </button>
                    </div>
                }
            </div>

            <!-- Divider -->
            <div class="border-t border-surface-mid/50"></div>

            <!-- Model Settings -->
            <div class="space-y-4">
                <h3 class="text-lg font-medium text-white flex items-center gap-2">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-emerald-400">
                        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                        <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
                        <line x1="12" y1="22.08" x2="12" y2="12"/>
                    </svg>
                    Model Settings
                </h3>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <!-- Global Offline Mode -->
                    <div class="p-4 rounded-theme-lg bg-base/50 border border-surface-high space-y-2">
                        <label class="text-[10px] uppercase text-text-subtle font-bold tracking-wider block">Global Offline Mode</label>
                        <label class="relative inline-flex items-center cursor-pointer mt-1"
                               data-testid="setting-global-offline">
                            <input type="checkbox" [ngModel]="modelSettings()?.global_offline_mode" (ngModelChange)="onModelSettingToggle('global_offline_mode', $event)"
                                class="sr-only peer">
                            <div class="w-11 h-6 bg-surface-high rounded-full peer peer-checked:bg-amber-500/60 transition-colors
                                        after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all
                                        peer-checked:after:translate-x-full"></div>
                            <span class="ml-3 text-sm text-text-secondary">{{ modelSettings()?.global_offline_mode ? 'Enabled' : 'Disabled' }}</span>
                        </label>
                        <p class="text-[10px] text-text-disabled">Block ALL HuggingFace network requests — use only locally cached models</p>
                    </div>

                    <!-- Default Model Path -->
                    <div class="p-4 rounded-theme-lg bg-base/50 border border-surface-high space-y-2">
                        <label class="text-[10px] uppercase text-text-subtle font-bold tracking-wider block">Default Model Path</label>
                        <div class="flex gap-2">
                            <input type="text" [ngModel]="modelSettings()?.default_model_path" (ngModelChange)="pendingModelPath.set($event); modelPathDirty.set(true)"
                                placeholder="D:\\Models"
                                class="flex-1 bg-surface-mid text-sm text-white border border-surface-high rounded-theme-lg px-3 py-2 focus:outline-none focus:border-brand font-mono"
                                data-testid="setting-default-model-path">
                            <button type="button" (click)="browseModelPath()"
                                [disabled]="browsingModelPath()"
                                title="Browse for folder"
                                class="px-2.5 py-2 bg-surface-high hover:bg-surface-mid text-text-secondary hover:text-white text-sm rounded-theme-lg border border-surface-high transition-colors disabled:opacity-40 font-bold">
                                @if (browsingModelPath()) {
                                    <svg class="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
                                } @else {
                                    ···
                                }
                            </button>
                        </div>
                        @if (modelPathDirty()) {
                            <div class="flex items-center justify-end gap-2 pt-1 animate-in fade-in duration-200">
                                <button (click)="saveModelPath()"
                                    class="px-3 py-1.5 bg-brand/20 text-brand hover:bg-brand/30 rounded-theme-lg transition-colors text-[10px] font-bold border border-brand/30 flex items-center gap-1 uppercase tracking-wider">
                                    <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
                                    Save Path
                                </button>
                            </div>
                        }
                        <p class="text-[10px] text-text-disabled">Base directory for model storage — used as initial browse location</p>
                    </div>
                </div>
            </div>

            <!-- Divider -->
            <div class="border-t border-surface-mid/50"></div>

            <!-- Log Management -->
            <div class="space-y-4">
                <h3 class="text-lg font-medium text-white flex items-center gap-2">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-yellow-400"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                    Log Management
                </h3>

                <!-- Controls Row -->
                <div class="flex items-center justify-between p-3 rounded-lg bg-base/50 border border-border-default">
                    <span class="font-mono text-xs text-text-muted">backend/server.log</span>
                    <div class="flex items-center gap-3">
                        <button (click)="clearLogs()" 
                            [disabled]="isActionPending()"
                            class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface-mid hover:bg-nav-active text-xs text-text-secondary font-medium transition-colors border border-border-default hover:border-border-default">
                            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                            Clear
                        </button>
                    </div>
                </div>

                <!-- Live Log Viewer -->
                <app-live-log-viewer></app-live-log-viewer>
            </div>
        </div>
    </div>
  `
})
export class ServerControlComponent implements OnInit {
    private http = inject(HttpClient);
    private toast = inject(ToastService);
    private rtc = inject(RuntimeConfigService);
    private settingsStore = inject(SettingsStore);

    readonly restartRequest = output<void>();

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

    isActionPending = signal<boolean>(false);
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
        this.http.get<ModelGlobalSettings>(`${this.rtc.apiUrl}/models/settings`).subscribe({
            next: (s) => {
                this.modelSettings.set(s);
                this.pendingModelPath.set(s.default_model_path);
                this.modelPathDirty.set(false);
            },
            error: () => console.error('Failed to load model settings'),
        });
    }

    onModelSettingToggle(key: string, value: boolean) {
        this.http.put<ModelGlobalSettings>(`${this.rtc.apiUrl}/models/settings`, { [key]: value }).subscribe({
            next: (s) => {
                this.modelSettings.set(s);
                this.toast.success(`${key === 'global_offline_mode' ? 'Offline mode' : key} ${value ? 'enabled' : 'disabled'}`);
            },
            error: (err) => this.toast.error(err.error?.detail || 'Failed to update model settings'),
        });
    }

    browseModelPath() {
        this.browsingModelPath.set(true);
        this.http.post<{ path: string }>(`${this.rtc.apiUrl}/filesystem/pick-folder`, {
            initial_dir: this.pendingModelPath() || '',
            title: 'Select Default Model Directory',
        }).subscribe({
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
        this.http.put<ModelGlobalSettings>(`${this.rtc.apiUrl}/models/settings`, {
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

    onRestart() {
        this.restartRequest.emit();
    }

    clearLogs() {
        if (!confirm('Are you sure you want to clear the server logs?')) return;

        this.isActionPending.set(true);

        this.http.post<{ message?: string, error?: string }>(`${this.rtc.apiUrl}/system/logs/clear`, {}).subscribe({
            next: (res) => {
                this.isActionPending.set(false);
                if (res.error) {
                    this.toast.error(`Error: ${res.error}`);
                } else {
                    this.toast.success(res.message || 'Logs cleared successfully');
                }
            },
            error: (err) => {
                this.isActionPending.set(false);
                this.toast.error('Failed to clear logs');
                console.error(err);
            }
        });
    }
}
