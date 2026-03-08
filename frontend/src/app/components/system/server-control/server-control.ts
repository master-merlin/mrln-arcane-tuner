import { Component, inject, signal, output, OnInit } from '@angular/core';
import { ToastService } from '../../../services/toast';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { LiveLogViewerComponent } from '../live-log-viewer/live-log-viewer';

interface ApplicationSettings {
    backend_port: number;
    frontend_port: number;
    log_level: string;
    start_frontend: boolean;
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

    readonly restartRequest = output<void>();

    isActionPending = signal<boolean>(false);
    settings = signal<ApplicationSettings | null>(null);
    settingsDirty = signal(false);
    isSavingSettings = signal(false);

    logLevels = ['DEBUG', 'INFO', 'WARNING', 'ERROR'];

    // Pending (unsaved) values
    private pendingChanges: Record<string, unknown> = {};

    ngOnInit() {
        this.loadSettings();
    }

    loadSettings() {
        this.http.get<ApplicationSettings>(`${this.rtc.apiUrl}/settings/application`).subscribe({
            next: (s) => {
                this.settings.set(s);
                this.pendingChanges = {};
                this.settingsDirty.set(false);
            },
            error: (err) => console.error('Failed to fetch settings', err)
        });
    }

    onSettingChange(key: keyof ApplicationSettings, value: string | number) {
        // Log level is applied immediately (hot-swap)
        if (key === 'log_level') {
            this.http.put(`${this.rtc.apiUrl}/settings/application`, { [key]: value }).subscribe({
                next: () => {
                    this.toast.success(`Log level set to ${value}`);
                    const cur = this.settings();
                    if (cur) this.settings.set({ ...cur, log_level: value as string });
                },
                error: (err) => {
                    this.toast.error(`Failed to set log level: ${err.error?.detail || err.message}`);
                }
            });
            return;
        }

        // For port settings, stage the change
        this.pendingChanges[key] = value;
        this.settingsDirty.set(true);
        // Update local display
        const cur = this.settings();
        if (cur) this.settings.set({ ...cur, [key]: value } as ApplicationSettings);
    }

    onToggleChange(key: keyof ApplicationSettings, value: boolean) {
        // Apply toggle settings immediately
        this.http.put(`${this.rtc.apiUrl}/settings/application`, { [key]: value }).subscribe({
            next: () => {
                this.toast.success(`${key.replace(/_/g, ' ')} ${value ? 'enabled' : 'disabled'}`);
                const cur = this.settings();
                if (cur) this.settings.set({ ...cur, [key]: value } as ApplicationSettings);
            },
            error: (err) => {
                this.toast.error(`Failed to update: ${err.error?.detail || err.message}`);
            }
        });
    }

    saveSettings() {
        if (!this.settings()) return;
        this.isSavingSettings.set(true);

        const payload = { ...this.settings()!, ...this.pendingChanges };

        this.http.put(`${this.rtc.apiUrl}/settings/application`, payload).subscribe({
            next: () => {
                this.isSavingSettings.set(false);
                this.settingsDirty.set(false);
                this.pendingChanges = {};
                this.toast.success('Settings saved. Port changes require a restart to take effect.');
            },
            error: (err) => {
                this.isSavingSettings.set(false);
                this.toast.error('Failed to save settings: ' + (err.error?.detail || err.message));
            }
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
