import { Component, inject, signal, viewChild, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { TrainingDynamicConfigComponent } from './components/training/training-dynamic-config/training-dynamic-config';
import { TrainingJobQueueComponent } from './components/training/training-job-queue/training-job-queue';
import { DatasetManagerComponent } from './components/dataset/dataset-manager/dataset-manager';
import { LoraToolsComponent } from './components/tools/lora-tools/lora-tools';
import { JobService } from './services/job';
import { ToastService } from './services/toast';
import { ToastContainerComponent } from './components/shared/toast-container/toast-container';
import { RuntimeConfigService } from './services/runtime-config.service';
import { ServerControlComponent } from './components/system/server-control/server-control';
import { SystemMonitorComponent } from './components/system/system-monitor/system-monitor';

type ViewMode = 'datasets' | 'train' | 'jobs' | 'tools' | 'server';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [TrainingDynamicConfigComponent, TrainingJobQueueComponent, DatasetManagerComponent, ServerControlComponent, LoraToolsComponent, ToastContainerComponent, SystemMonitorComponent],
  template: `
    <div class="min-h-screen bg-base text-text-primary font-sans selection:bg-brand/30 selection:text-white">
      
      <!-- Header -->
      <header class="border-b border-border-default bg-surface-low/50 backdrop-blur-md sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div class="flex items-center gap-8">
             <div class="flex items-center gap-2">
                 <div class="w-8 h-8 rounded-theme-lg bg-gradient-to-br from-brand to-brand-gradient-end flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-brand/20">M</div>
                 <h1 class="text-xl font-bold tracking-tight text-white">MRLN Arcane Tuner</h1>
             </div>

             <!-- Navigation -->
            <nav class="flex gap-1 bg-surface-mid/50 p-1 rounded-theme-lg">
                <button (click)="setView('datasets')" 
                    data-testid="nav-datasets"
                    [class.bg-nav-active]="currentView() === 'datasets'" 
                    [class.text-white]="currentView() === 'datasets'" 
                    class="px-4 py-1.5 rounded-theme-md text-sm font-medium text-text-muted hover:text-white transition-all">
                    Datasets
                </button>
                <button (click)="setView('train')" 
                    data-testid="nav-train"
                    [class.bg-nav-active]="currentView() === 'train'" 
                    [class.text-white]="currentView() === 'train'" 
                    class="px-4 py-1.5 rounded-theme-md text-sm font-medium text-text-muted hover:text-white transition-all">
                    Training
                </button>
                <button (click)="setView('jobs')" 
                    data-testid="nav-jobs"
                    [class.bg-nav-active]="currentView() === 'jobs'" 
                    [class.text-white]="currentView() === 'jobs'" 
                    class="px-4 py-1.5 rounded-theme-md text-sm font-medium text-text-muted hover:text-white transition-all">
                    Jobs
                </button>
                <button (click)="setView('tools')" 
                    data-testid="nav-tools"
                    [class.bg-nav-active]="currentView() === 'tools'" 
                    [class.text-white]="currentView() === 'tools'" 
                    class="px-4 py-1.5 rounded-theme-md text-sm font-medium text-text-muted hover:text-white transition-all">
                    Tools
                </button>
                <button (click)="setView('server')" 
                    data-testid="nav-server"
                    [class.bg-nav-active]="currentView() === 'server'" 
                    [class.text-white]="currentView() === 'server'" 
                    class="px-4 py-1.5 rounded-theme-md text-sm font-medium text-text-muted hover:text-white transition-all">
                    Server
                </button>
             </nav>
          </div>
          <div class="flex items-center gap-4">
            <div class="text-sm text-text-muted">v{{ appVersion() }}</div>
          </div>
        </div>
      </header>

      <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 backface-hidden">
        
        @if (error()) {
            <div class="bg-danger/10 border border-danger/40 text-danger px-4 py-3 rounded-theme-xl mb-8 flex items-center gap-3">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                <span>{{ error() }}</span>
                <button (click)="fetchModels()" class="ml-auto bg-danger/60 hover:bg-danger/80 text-white text-sm font-medium py-1 px-3 rounded-theme-md transition-colors">Retry</button>
            </div>
        }

        @if (currentView() === 'datasets') {
            <div class="space-y-8 animate-in fade-in duration-300">
                <app-dataset-manager></app-dataset-manager>
            </div>
        } @else if (currentView() === 'train') {
            <div class="space-y-8 animate-in fade-in duration-300">
                
                <!-- Header -->
                <div class="bg-surface-low/50 border border-border-default rounded-theme-xl p-8 shadow-xl">
                    <h2 class="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-brand"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
                        Training
                    </h2>
                    <p class="text-text-muted">Configure and launch model fine-tuning jobs.</p>
                </div>

                <!-- Configuration Panel -->
                @if (currentSchema()) {
                    <section class="animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <app-training-dynamic-config 
                          [schema]="currentSchema()" 
                          [availableModels]="availableModels()"
                          (configSubmitted)="queueJob($event)">
                        </app-training-dynamic-config>
                    </section>
                } @else {
                    <div class="p-12 text-center text-text-subtle bg-surface-low/30 rounded-theme-xl border border-border-default border-dashed">
                        Select a model to configure training parameters.
                    </div>
                }
            </div>
        } @else if (currentView() === 'jobs') {
            <div class="space-y-8 animate-in fade-in duration-300">

                <!-- Header -->
                <div class="bg-surface-low/50 border border-border-default rounded-theme-xl p-8 shadow-xl">
                    <h2 class="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-brand"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path></svg>
                        Jobs
                    </h2>
                    <p class="text-text-muted">Monitor system resources and manage training job queue.</p>
                </div>

                <!-- Content Card -->
                <div class="bg-surface-low border border-surface-mid rounded-theme-xl shadow-2xl p-6 space-y-8">
                    <!-- System Monitor -->
                    <app-system-monitor></app-system-monitor>

                    <!-- Job Queue -->
                    <app-training-job-queue
                      (saveAsTemplate)="handleSaveAsTemplate($event)"
                      (reloadConfig)="handleReloadConfig($event)">
                    </app-training-job-queue>
                </div>
            </div>
        } @else if (currentView() === 'tools') {
            <div class="space-y-8 animate-in fade-in duration-300">
                <app-lora-tools></app-lora-tools>
            </div>
        } @else {
            <div class="space-y-8 animate-in fade-in duration-300">
                <app-server-control (restartRequest)="restartServer()"></app-server-control>
            </div>
        }

      </main>

      <!-- Restart Overlay -->
      @if (isRestarting()) {
        <div class="fixed inset-0 bg-overlay backdrop-blur-xl z-[9999] flex flex-col items-center justify-center p-8 text-center animate-in fade-in duration-500">
            <div class="w-16 h-16 border-4 border-brand/20 border-t-brand rounded-full animate-spin mb-6"></div>
            <h2 class="text-2xl font-bold text-white mb-2">Server Restarting</h2>
            <p class="text-text-muted max-w-md">The backend is performing a graceful restart. The connection will be restored automatically in a few seconds.</p>
            <div class="mt-8 flex gap-2">
                <div class="w-1.5 h-1.5 rounded-full bg-brand animate-bounce [animation-delay:-0.3s]"></div>
                <div class="w-1.5 h-1.5 rounded-full bg-brand animate-bounce [animation-delay:-0.15s]"></div>
                <div class="w-1.5 h-1.5 rounded-full bg-brand animate-bounce"></div>
            </div>
        </div>
      }

      <app-toast-container></app-toast-container>
    </div>
  `,
  styles: []
})
export class AppComponent implements OnInit {
  private http = inject(HttpClient);
  private jobService = inject(JobService);
  private toast = inject(ToastService);
  private rtc = inject(RuntimeConfigService);

  availableModels = signal<any[]>([]);
  private readonly pluginId = 'standard';
  selectedDefinitionId = signal<string | null>(null);
  currentSchema = signal<any>(null);
  error = signal<string | null>(null);
  currentView = signal<ViewMode>('datasets');
  isRestarting = signal<boolean>(false);
  appVersion = signal<string>('…');

  jobQueue = viewChild(TrainingJobQueueComponent);
  configEditor = viewChild(TrainingDynamicConfigComponent);

  ngOnInit() {
    this.fetchModels();
    this.fetchVersion();
  }

  fetchVersion() {
    this.http.get<{ version: string }>(`${this.rtc.apiUrl.replace('/api', '/')}`).subscribe({
      next: (res) => this.appVersion.set(res.version),
      error: () => this.appVersion.set('?.?.?')
    });
  }

  setView(view: ViewMode) {
    const prevView = this.currentView();
    this.currentView.set(view);
    if (view === 'train' && prevView !== 'train' && this.selectedDefinitionId()) {
      this.selectDefinition(this.selectedDefinitionId()!);
    }
  }

  fetchModels() {
    this.http.get<any[]>(`${this.rtc.apiUrl}/models/definitions`).subscribe({
      next: (defs) => {
        this.availableModels.set(defs);
        this.error.set(null);
        if (defs.length > 0) {
          this.selectDefinition(defs[0].id);
        }
      },
      error: (err) => {
        console.error('Failed to fetch definitions', err);
        this.error.set(`Failed to connect to backend at ${this.rtc.apiUrl}. Is it running?`);
      }
    });
  }

  selectDefinition(id: string) {
    this.selectedDefinitionId.set(id);

    this.http.get(`${this.rtc.apiUrl}/plugins/${this.pluginId}/schema?t=${Date.now()}`).subscribe({
      next: (schema: any) => {
        this.currentSchema.set(schema);
      },
      error: (err) => console.error('Failed to fetch schema', err)
    });
  }

  queueJob(config: any) {
    this.jobService.createJob(this.pluginId, config).subscribe({
      next: () => {
        if (this.currentView() === 'train' || this.currentView() === 'jobs') {
          this.jobQueue()?.loadJobs();
        }
      },
      error: (err: any) => this.toast.error('Failed to create job: ' + (err.error?.detail || err.message))
    });
  }

  restartServer() {
    if (!confirm('Are you sure you want to restart the backend server? Active training jobs will continue running, but communication will be lost for a few seconds.')) {
      return;
    }

    this.isRestarting.set(true);
    this.http.post(`${this.rtc.apiUrl}/system/restart`, {}).subscribe({
      next: () => {
        this.pollForServer();
      },
      error: () => {
        this.pollForServer();
      }
    });
  }

  pollForServer() {
    // Wait bit then start polling
    setTimeout(() => {
      const interval = setInterval(() => {
        this.http.get(`${this.rtc.apiUrl}/models/definitions`).subscribe({
          next: () => {
            clearInterval(interval);
            this.isRestarting.set(false);
            this.fetchModels(); // Refresh data
          },
          error: () => { /* Server still restarting, retry silently */ }
        });
      }, 2000);
    }, 2000);
  }

  handleSaveAsTemplate(event: { name: string, config: any, definition_id: string }) {
    this.configEditor()?.importTemplate(event.name, event.config, event.definition_id);
  }

  handleReloadConfig(config: any) {
    this.configEditor()?.loadExternalConfig(config);
  }
}
