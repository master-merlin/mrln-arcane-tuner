import { Component, OnInit, DestroyRef, inject, signal, computed, effect, output, HostListener } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { DatePipe, JsonPipe, DecimalPipe, UpperCasePipe } from '@angular/common';
import { JobService, Job, JobStatus } from '../../../services/job';
import { JobStore } from '../../../state/job.store';
import { WebSocketService } from '../../../services/websocket.service';
import { interval } from 'rxjs';
import { FormsModule } from '@angular/forms';
import { TrainingChartComponent, ChartDataPoint, SmoothingMode } from '../training-chart/training-chart';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { ProjectService } from '../../../services/project.service';
import { ModelService, ModelSourceOverride } from '../../../services/model.service';

@Component({
  selector: 'app-training-job-queue',
  standalone: true,
  imports: [DatePipe, JsonPipe, DecimalPipe, UpperCasePipe, FormsModule, TrainingChartComponent],
  template: `
    <div class="bg-surface-low/50 rounded-theme-xl border border-surface-mid overflow-hidden shadow-2xl">
      <div class="px-6 py-4 border-b border-surface-mid flex justify-between items-center bg-surface-low/80">
        <div class="flex items-center gap-2">
          <div class="w-2 h-2 bg-brand rounded-full animate-pulse"></div>
          <h3 class="text-sm font-semibold text-text-secondary uppercase tracking-wider">Active Workspace Queue</h3>
        </div>
        <div class="flex items-center gap-2">
          <button (click)="refreshAll()" 
            data-testid="refresh-jobs-btn"
            class="text-xs font-medium text-brand hover:text-brand/80 transition-colors flex items-center gap-1 group">
            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="group-hover:rotate-180 transition-transform duration-500"><path d="M21 2v6h-6"></path><path d="M3 12a9 9 0 0 1 15-6.7L21 8"></path><path d="M3 22v-6h6"></path><path d="M21 12a9 9 0 0 1-15 6.7L3 16"></path></svg>
            Refresh
          </button>
          <div class="h-4 w-px bg-surface-mid mx-1"></div>
          <label class="relative inline-flex items-center cursor-pointer group" title="When enabled, pending jobs will start automatically when no other job is running">
              <input type="checkbox" [checked]="autoQueue()" (change)="toggleAutoQueue()" class="sr-only peer">
              <div class="w-9 h-5 bg-surface-high border border-surface-mid/50 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-brand/50 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-text-muted after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-brand peer-checked:after:bg-white transition-colors"></div>
              <span class="ml-2 text-xs font-medium text-text-muted group-hover:text-text-secondary transition-colors">Auto Queue</span>
          </label>
        </div>
      </div>

      <div class="divide-y divide-surface-mid/50">
        @for (job of activeJobs(); track job.id) {
          <div [attr.data-testid]="'job-item-' + job.id"
               class="px-6 py-5 hover:bg-surface-mid/30 transition-all group border-l-2 border-transparent" [class.border-l-brand]="job.status === JobStatus.RUNNING">
            <div class="flex flex-col gap-3">
              <!-- Top Row: ID and Status -->
              <div class="flex items-center justify-between">
                <div class="flex items-center gap-3">
                  <div class="flex flex-col">
                    <span class="text-white font-bold text-sm tracking-tight flex items-center gap-2">
                       <span class="text-brand-light">{{ job.config['lora_name'] || 'UNNAMED' }}</span>
                       <span class="text-text-subtle font-normal">on</span>
                       {{ job.config['definition_id'] || job.plugin_id | uppercase }}
                    </span>
                    <div class="text-[10px] text-text-muted flex items-center gap-2 mt-0.5 font-mono flex-wrap">
                      <span class="text-text-disabled">#{{ job.id.slice(0, 8) }}</span>
                      <span>&bull;</span>
                      <span>PID: {{ job.pid || 'N/A' }}</span>
                      <span>&bull;</span>
                      <span>{{ job.created_at * 1000 | date:'MMM d, HH:mm' }}</span>
                      @if (job.config['project_id']) {
                        <span>&bull;</span>
                        <span class="font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full bg-brand/15 text-brand-light border border-brand/30 cursor-default">{{ getProjectName(job.config['project_id']) }}</span>
                      }
                      @if (getModelSource(job); as src) {
                        @if (src.source_type !== 'hf_hub') {
                          <span>&bull;</span>
                          <span class="font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full cursor-default"
                                [title]="src.local_path || ''"
                                [class]="src.source_type === 'local_diffusers'
                                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                                  : 'bg-amber-500/10 text-amber-400 border border-amber-500/30'">
                            {{ src.source_type === 'local_diffusers' ? 'LOCAL' : 'SAFETENSORS' }}
                          </span>
                        } @else if (src.skip_update) {
                          <span>&bull;</span>
                          <span class="font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/30 cursor-default">
                            OFFLINE
                          </span>
                        }
                      }
                    </div>
                  </div>
                </div>
                
                <div class="flex items-center gap-3">
                  <span [class]="getStatusClass(job.status)" 
                    [attr.data-testid]="'job-status-' + job.id"
                    class="text-[10px] px-2.5 py-1 rounded-full uppercase font-black tracking-widest shadow-sm">
                    {{ job.status }}
                  </span>
                  @if (job.status_label && (job.status === JobStatus.RUNNING || job.status === JobStatus.PAUSED)) {
                    <span class="text-[10px] text-text-muted font-medium tracking-wide">· {{ job.status_label }}</span>
                  }
                  
                  <div class="flex items-center gap-1 opacity-100 transition-opacity">
                    <!-- Samples Toggle -->
                    @if (job.config['sample_every_n_steps'] > 0) {
                      <button (click)="toggleSamples(job.id)" 
                        [attr.data-testid]="'toggle-job-samples-' + job.id"
                        class="p-1.5 rounded-theme-lg transition-colors" 
                        [class]="!jobsWithSamples().has(job.id) ? 'text-text-disabled cursor-not-allowed opacity-30' : samplesExpandedJobs().has(job.id) ? 'text-brand hover:text-brand/80' : 'text-text-subtle hover:text-brand'"
                        [disabled]="!jobsWithSamples().has(job.id)"
                        [title]="!jobsWithSamples().has(job.id) ? 'No samples yet' : samplesExpandedJobs().has(job.id) ? 'Hide Samples' : 'Show Samples'">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
                      </button>
                    }
                    <!-- Chart Toggle -->
                    <button (click)="toggleChart(job.id)" 
                      [attr.data-testid]="'toggle-job-chart-' + job.id"
                      [disabled]="!getChartData(job)"
                      class="p-1.5 text-text-subtle hover:text-brand rounded-theme-lg transition-colors disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:text-text-subtle" [title]="chartExpandedJobs().has(job.id) ? 'Hide Chart' : 'Show Chart'">
                      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
                    </button>

                    <button (click)="toggleConfig(job.id)" 
                      [attr.data-testid]="'toggle-job-config-' + job.id"
                      class="p-1.5 text-text-subtle hover:text-white rounded-theme-lg transition-colors" [title]="configExpandedJobs().has(job.id) ? 'Collapse Config' : 'View Config'">
                      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" [class.rotate-180]="configExpandedJobs().has(job.id)" class="transition-transform"><path d="m6 9 6 6 6-6"/></svg>
                    </button>
                    @if (job.status === JobStatus.PENDING) {
                       <button (click)="startJob(job.id)" 
                        [attr.data-testid]="'start-job-' + job.id"
                        class="p-1.5 text-success hover:bg-success/10 rounded-theme-lg transition-colors" title="Start Job">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                      </button>
                    }
                    @if (job.status === JobStatus.RUNNING) {
                      <button (click)="pauseJob(job.id)" 
                        class="p-1.5 text-amber-400 hover:bg-amber-400/10 rounded-theme-lg transition-colors" title="Pause Training">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
                      </button>
                    }
                    @if (job.status === JobStatus.PAUSED) {
                      <button (click)="resumeJob(job.id)" 
                        class="p-1.5 text-success hover:bg-success/10 rounded-theme-lg transition-colors" title="Resume Training">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                      </button>
                    }
                    @if (job.status === JobStatus.RUNNING || job.status === JobStatus.PAUSED) {
                      <button (click)="openStopModal(job.id)" 
                        [attr.data-testid]="'stop-job-' + job.id"
                        class="p-1.5 text-warning hover:bg-warning/10 rounded-theme-lg transition-colors" title="Stop Job">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg>
                      </button>
                    }
                    @if (job.status === JobStatus.FAILED || job.status === JobStatus.STOPPED) {
                      <button (click)="restartJob(job.id)" 
                        [attr.data-testid]="'restart-job-' + job.id"
                        class="p-1.5 text-blue-400 hover:bg-blue-400/10 rounded-theme-lg transition-colors" title="Restart Job">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path><path d="M3 3v5h5"></path></svg>
                      </button>
                    }
                    <button (click)="deleteJob(job.id)" 
                      [attr.data-testid]="'delete-job-' + job.id"
                      class="p-1.5 text-text-subtle hover:text-danger hover:bg-danger/10 rounded-theme-lg transition-colors" title="Delete Job">
                      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"></path><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path></svg>
                    </button>
                  </div>
                </div>
              </div>

              <!-- Stop Modal -->
              @if (stopModalJobId() === job.id) {
                <div class="bg-surface-mid/95 backdrop-blur-sm p-4 rounded-theme-xl border border-surface-high animate-in fade-in slide-in-from-top-2 duration-200">
                  <p class="text-sm text-text-secondary mb-3 font-medium">How would you like to stop this job?</p>
                  <div class="flex items-center gap-2">
                    <button (click)="softStopJob(job.id)" 
                      class="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-brand/20 hover:bg-brand/30 text-brand border border-brand/30 rounded-theme-lg text-sm font-semibold transition-colors">
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>
                      Soft Stop
                    </button>
                    <button (click)="hardStopJob(job.id)" 
                      class="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-danger/20 hover:bg-danger/30 text-danger border border-danger/30 rounded-theme-lg text-sm font-semibold transition-colors">
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/></svg>
                      Hard Stop
                    </button>
                    <button (click)="closeStopModal()" 
                      class="px-3 py-2 text-text-muted hover:text-white hover:bg-surface-high rounded-theme-lg text-sm transition-colors">
                      Cancel
                    </button>
                  </div>
                  <p class="text-[10px] text-text-subtle mt-2"><strong>Soft Stop</strong> saves a checkpoint before stopping. <strong>Hard Stop</strong> kills immediately.</p>
                </div>
              }

              <!-- Samples Grid Expander -->
              @if (samplesExpandedJobs().has(job.id)) {
                <div class="bg-base/25 border border-surface-mid/20 rounded-theme-xl p-4 mt-2 mb-2 shadow-inner animate-in slide-in-from-top-2">
                  <div class="flex justify-between items-center mb-3">
                    <h4 class="text-xs font-bold text-text-muted uppercase tracking-wider">Sample Previews</h4>
                    <div class="flex items-center gap-2">
                      <!-- Cadence Selector -->
                      <div class="flex items-center gap-1">
                        <span class="text-[10px] text-text-subtle font-medium">Every</span>
                        @if (!customCadenceMode().has(job.id)) {
                          <select [value]="samplingCadence().get(job.id) ?? job.config['sample_every_n_steps'] ?? 0"
                            (change)="onCadenceChange(job.id, $event)"
                            data-testid="sampling-cadence-select"
                            class="text-[10px] font-mono font-bold bg-surface-high border border-surface-mid/50 text-text-secondary rounded px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-brand/50 appearance-none cursor-pointer">
                            @if (job.config['sample_every_n_steps'] && ![50,100,150,200,250].includes(+job.config['sample_every_n_steps'])) {
                              <option [value]="job.config['sample_every_n_steps']">{{ job.config['sample_every_n_steps'] }}</option>
                            }
                            <option [value]="50">50</option>
                            <option [value]="100">100</option>
                            <option [value]="150">150</option>
                            <option [value]="200">200</option>
                            <option [value]="250">250</option>
                            <option value="custom">Custom…</option>
                          </select>
                        } @else {
                          <input type="number" min="1" step="1"
                            [value]="samplingCadence().get(job.id) ?? job.config['sample_every_n_steps'] ?? 50"
                            (keydown.enter)="applyCustomCadence(job.id, $event)"
                            (blur)="applyCustomCadence(job.id, $event)"
                            data-testid="sampling-cadence-input"
                            class="w-14 text-[10px] font-mono font-bold bg-surface-high border border-brand/50 text-text-secondary rounded px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-brand/50"
                            placeholder="#">
                        }
                        <span class="text-[10px] text-text-subtle font-medium">steps</span>
                      </div>
                      <div class="w-px h-3 bg-surface-mid"></div>
                      <label class="relative inline-flex items-center cursor-pointer group" title="Pause sampling during training">
                        <input type="checkbox" [checked]="samplingPausedJobs().has(job.id)" (change)="toggleSamplingPause(job.id)" class="sr-only peer">
                        <div class="w-7 h-4 bg-surface-high border border-surface-mid/50 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-text-muted after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-amber-500 peer-checked:after:bg-white transition-colors"></div>
                        <span class="ml-1 text-[10px] font-medium transition-colors" [class]="samplingPausedJobs().has(job.id) ? 'text-amber-400' : 'text-text-subtle'">Pause</span>
                      </label>
                      <button (click)="loadSamples(job.id)" 
                        class="p-1 text-brand hover:text-brand/80 transition-colors group" title="Refresh samples">
                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="group-hover:rotate-180 transition-transform duration-500"><path d="M21 2v6h-6"></path><path d="M3 12a9 9 0 0 1 15-6.7L21 8"></path><path d="M3 22v-6h6"></path><path d="M21 12a9 9 0 0 1-15 6.7L3 16"></path></svg>
                      </button>
                    </div>
                  </div>
                  @if (jobSamples().get(job.id); as samples) {
                    @if (samples.length > 0) {
                      <div class="grid grid-cols-6 gap-2 max-h-[33rem] overflow-y-auto scrollbar-thin scrollbar-thumb-surface-high pr-1">
                        @for (sample of samples; track sample.filename) {
                          <button (click)="openSampleModal(job.id, sample)" class="relative group rounded-theme-lg overflow-hidden border border-surface-mid hover:border-brand/50 transition-all hover:shadow-lg hover:shadow-brand/10 aspect-square bg-surface-high">
                            <img [src]="getSampleImageUrl(job.id, sample.filename)" 
                                 [alt]="'Step ' + sample.step"
                                 class="w-full h-full object-cover transition-transform group-hover:scale-105" loading="lazy">
                            <div class="absolute bottom-0 left-0 right-0 bg-base/80 px-2 py-1">
                              <span class="text-[10px] text-white font-mono font-bold">{{ sample.step === 999999 ? 'Final' : 'Step ' + sample.step }}</span>
                            </div>
                          </button>
                        }
                      </div>
                    } @else {
                      <div class="text-center text-xs text-text-subtle py-8 italic">No samples yet — waiting for first sampling step...</div>
                    }
                  } @else {
                    <div class="text-center text-xs text-text-subtle py-8 italic">Loading samples...</div>
                  }
                </div>
              }

              <!-- Chart Expander (above metrics) -->
              @if (chartExpandedJobs().has(job.id)) {
                <div class="bg-base/25 border border-surface-mid/20 rounded-theme-xl p-4 mt-2 mb-2 shadow-inner animate-in slide-in-from-top-2">
                    <div class="flex justify-between items-center mb-4">
                        <h4 class="text-xs font-bold text-text-muted uppercase tracking-wider">Training Curves</h4>
                        <div class="flex items-center gap-3">
                             <button (click)="smoothingMode.set(smoothingMode() === 'ema' ? 'sma' : 'ema')"
                                     data-testid="smoothing-mode-toggle"
                                     class="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full border transition-colors"
                                     [class]="smoothingMode() === 'ema'
                                       ? 'border-brand/40 bg-brand/10 text-brand-light'
                                       : 'border-sky-500/40 bg-sky-500/10 text-sky-400'">
                               {{ smoothingMode() === 'ema' ? 'EMA' : 'SMA' }}
                             </button>
                             <span class="text-[10px] text-text-subtle uppercase font-bold">Smoothing: {{ smoothingFactor() }}</span>
                            <input type="range" min="0" max="0.99" step="0.01" 
                                   [ngModel]="smoothingFactor()" (ngModelChange)="smoothingFactor.set($event)"
                                   class="w-24 h-1 bg-surface-high rounded-lg appearance-none cursor-pointer accent-brand">
                        </div>
                    </div>
                    
                    @if(getChartData(job); as chart) {
                        @if (chart.length > 1) {
                            <app-training-chart [data]="chart" [smoothing]="smoothingFactor()" [smoothingMode]="smoothingMode()" [height]="180"
                                [totalSteps]="getLatestMetrics(job)?.total_steps || 0"
                                (plateauDetected)="onPlateauDetected(job, $event)"></app-training-chart>
                        } @else {
                             <div class="text-center text-xs text-text-subtle py-8 italic">Collecting data (wait for step 5)...</div>
                        }
                    }
                </div>
              }

              <!-- Diagnostics Warnings -->
              @if (job.warnings?.length) {
                <div class="bg-amber-950/40 border border-amber-500/40 rounded-theme-xl p-3 mt-2 animate-in slide-in-from-top-1">
                  <div class="flex items-center gap-2 mb-1">
                    <svg class="w-4 h-4 text-amber-400 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fill-rule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 6a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 6zm0 9a1 1 0 100-2 1 1 0 000 2z" clip-rule="evenodd"/>
                    </svg>
                    <span class="text-amber-400 text-[10px] font-bold uppercase tracking-wider">Diagnostics</span>
                  </div>
                  @for (warning of job.warnings; track warning) {
                    <p class="text-amber-200/80 text-xs leading-relaxed pl-6">{{ warning }}</p>
                  }
                </div>
              }

              <!-- Metrics Row (if running or has logs) -->
              @if (getLatestMetrics(job); as metrics) {
                <div [attr.data-testid]="'job-metrics-' + job.id"
                     class="space-y-2">
                  <!-- Progress Bar Card -->
                  <div class="bg-base/25 p-3 rounded-theme-xl border border-surface-mid/20 w-full">
                    <div class="flex items-center gap-3">
                      <div class="flex-1 h-1.5 bg-surface-high rounded-full overflow-hidden">
                        <div class="h-full bg-brand transition-all duration-1000 shadow-[0_0_10px_rgba(var(--color-brand-rgb),0.5)]"
                             [style.width.%]="metrics.progress">
                        </div>
                      </div>
                      <span class="text-xl font-black text-white italic w-[3.5rem] text-right">{{ metrics.progress }}%</span>
                    </div>
                  </div>

                  <!-- Unified Metrics Grid Card -->
                  <div class="flex justify-between w-full bg-base/25 p-3 rounded-theme-xl border border-surface-mid/20">
                    <!-- Col 1: Progress (Step / Epoch) -->
                    <div class="flex flex-col space-y-1 w-24">
                      <div>
                        <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">Step</span>
                        <span class="text-xs text-brand font-mono font-bold">{{ metrics.step }}/{{ metrics.total_steps || '?' }}</span>
                      </div>
                      <div>
                        <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">Epoch</span>
                        <span class="text-xs text-purple-400 font-mono font-bold">{{ metrics.epoch || '—' }}</span>
                      </div>
                    </div>
                    
                    <!-- Col 2: Optimization (Loss / Status) -->
                    <div class="flex flex-col space-y-1 w-28">
                      <div>
                        <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">Loss</span>
                        <span class="text-xs text-brand-light font-mono font-bold">{{ metrics.loss | number:'1.4-6' }}</span>
                      </div>
                      <div>
                        <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">Status</span>
                        @if (getLossStatus(job); as status) {
                          <span class="text-xs font-bold uppercase tracking-wider tooltip-trigger"
                                [class.text-emerald-400]="status.icon === '🟢'"
                                [class.text-amber-400]="status.icon === '🟡'"
                                [class.text-danger]="status.icon === '🔴'"
                                [title]="status.tooltip">
                            {{ status.text }}
                          </span>
                        } @else {
                          <span class="text-xs text-text-muted italic">—</span>
                        }
                      </div>
                    </div>

                    <!-- Col 3: Throughput (Step Time / Samples/s) -->
                    <div class="flex flex-col space-y-1 w-28">
                      <div>
                        <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">Step Time</span>
                        <span class="text-xs text-brand font-mono font-bold">{{ metrics.step_time }}s</span>
                      </div>
                      <div>
                        <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">Samples/s</span>
                        <span class="text-xs text-sky-400 font-mono font-bold">{{ metrics.samples_per_sec || '—' }}</span>
                      </div>
                    </div>

                    <!-- Col 4: Hardware & Optimization (VRAM+Res / Grad Norm) -->
                    <div class="flex flex-col space-y-1 w-32">
                      <div>
                        <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">VRAM <span class="lowercase text-[8px] opacity-70">/ res</span></span>
                        <span class="text-xs text-emerald-400 font-mono font-bold">
                          @if (metrics.vram_reserved_mb || metrics.vram_allocated_mb) {
                            {{ ((metrics.vram_reserved_mb || metrics.vram_allocated_mb) / 1024) | number:'1.1-1' }} GB
                          } @else {
                            —
                          }
                        </span>
                        @if (metrics.resolution) {
                          <span class="text-[9px] text-teal-400 font-mono ml-1">[{{ metrics.resolution }}]</span>
                        }
                      </div>
                      <div>
                        <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">Grad Norm</span>
                        <span class="text-xs text-text-secondary font-mono font-bold">{{ metrics.grad_norm != null ? formatGradNorm(metrics.grad_norm) : '—' }}</span>
                      </div>
                    </div>

                    <!-- Col 5: Time (Elapsed / ETC) -->
                    <div class="flex flex-col space-y-1 w-24 text-right">
                      <div>
                        <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">Elapsed</span>
                        <span class="text-xs text-text-secondary font-mono font-bold">{{ getDuration(job) }}</span>
                      </div>
                      <div>
                        <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">ETC</span>
                        <span class="text-xs text-blue-300 font-mono font-bold">{{ formatEta(metrics.eta) }}</span>
                      </div>
                    </div>
                  </div>

                  <!-- Anomaly / Secondary Warnings Card -->
                  @if (metrics.nan_count) {
                    <div class="bg-red-500/5 p-2 rounded-theme-xl border border-red-500/30 w-full animate-pulse">
                      <div class="flex items-center gap-3 px-1">
                          <span class="text-[10px] font-black uppercase tracking-wider text-red-400"
                                title="NaN losses detected — consider reducing learning rate">
                            ⚠ {{ metrics.nan_count }} NaN
                          </span>
                      </div>
                    </div>
                  }
                </div>

              }

              <!-- Config Area (Collapsible) -->
              @if (configExpandedJobs().has(job.id)) {
                <div class="bg-base/70 p-4 rounded-theme-xl border border-surface-mid animate-in slide-in-from-top-2 duration-300">
                  <div class="flex items-center justify-between mb-2">
                    <span class="text-[10px] text-text-subtle uppercase font-black tracking-widest">Job Configuration</span>
                    <div class="flex items-center gap-1">
                      <button (click)="onSaveAsTemplate(job)" type="button"
                        class="p-1.5 text-text-subtle hover:text-brand hover:bg-brand/10 rounded-theme-lg transition-colors" title="Save as Template">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z"/></svg>
                      </button>
                      <button (click)="onReloadConfig(job)" type="button"
                        class="p-1.5 text-text-subtle hover:text-green-400 hover:bg-green-400/10 rounded-theme-lg transition-colors" title="Reload into Settings">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>
                      </button>
                    </div>
                  </div>
                  <pre class="text-[10px] text-brand/80 font-mono overflow-auto max-h-40 scrollbar-thin scrollbar-thumb-surface-high"
                       [attr.data-testid]="'job-config-json-' + job.id">{{ job.config | json }}</pre>
                </div>
              }

              @if (job.error) {
                <div [attr.data-testid]="'job-error-' + job.id"
                     class="text-xs text-danger mt-1 flex items-start gap-2 bg-danger/10 p-2 rounded-theme-lg border border-danger/30">
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="mt-0.5 shrink-0"><circle cx="12" cy="12" r="10"></circle><line x1="12" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                  <span class="break-words">{{ job.error }}</span>
                </div>
              }
            </div>
          </div>
        } @empty {
          <div class="px-6 py-12 text-center animate-pulse">
            <div class="text-text-disabled text-sm italic">Waiting for new training directives...</div>
          </div>
        }
      </div>

      <!-- Archive Section (always visible) -->
        <div class="mt-4 bg-surface-low/50 rounded-theme-xl border border-surface-mid overflow-hidden shadow-lg">
          <div class="px-6 py-3 flex justify-between items-center bg-surface-low/80 hover:bg-surface-mid/30 transition-colors border-b border-surface-mid/50">
            <button (click)="toggleArchive()" data-testid="toggle-archive-btn" class="flex-1 flex items-center gap-2 cursor-pointer text-left">
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-text-muted">
                <rect width="20" height="5" x="2" y="3" rx="1"/><path d="M4 8v11a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8"/><path d="M10 12h4"/>
              </svg>
              <span class="text-xs font-semibold text-text-muted uppercase tracking-wider">Archive</span>
              <span class="text-[10px] font-bold text-text-disabled bg-surface-high px-2 py-0.5 rounded-full">{{ archivedJobs().length }}</span>
            </button>
            <div class="flex items-center gap-4 pl-4 border-l border-surface-mid/50">
               <select [ngModel]="archiveProjectFilter()" (ngModelChange)="onArchiveScopeChange($event)"
                   data-testid="archive-project-selector"
                   class="bg-surface-high border border-surface-mid text-white text-[10px] rounded-theme-md px-2 py-1 outline-none focus:border-brand uppercase tracking-wider font-semibold">
                   <option [value]="'all'">All Projects</option>
                   @for (p of projectService.allProjects(); track p.id) {
                       <option [value]="p.id">{{ p.name }}</option>
                   }
               </select>
               <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                class="text-text-muted transition-transform cursor-pointer" [class.rotate-180]="archiveExpanded()" (click)="toggleArchive()">
                <path d="m6 9 6 6 6-6"/>
               </svg>
            </div>
          </div>
          @if (archiveExpanded()) {
            <div class="divide-y divide-surface-mid/50">
              @for (job of archivedJobs(); track job.id) {
                <div [attr.data-testid]="'job-item-' + job.id"
                     class="px-6 py-5 hover:bg-surface-mid/30 transition-all group border-l-2 border-transparent opacity-80">
                  <div class="flex flex-col gap-3">
                    <div class="flex items-center justify-between">
                      <div class="flex items-center gap-3">
                        <div class="flex flex-col">
                          <span class="text-white font-bold text-sm tracking-tight flex items-center gap-2">
                             <span class="text-brand-light">{{ job.config['lora_name'] || 'UNNAMED' }}</span>
                             <span class="text-text-subtle font-normal">on</span>
                             {{ job.config['definition_id'] || job.plugin_id | uppercase }}
                          </span>
                          <div class="text-[10px] text-text-muted flex items-center gap-2 mt-0.5 font-mono flex-wrap">
                            <span class="text-text-disabled">#{{ job.id.slice(0, 8) }}</span>
                            <span>&bull;</span>
                            <span>{{ job.created_at * 1000 | date:'MMM d, HH:mm' }}</span>
                            @if (job.finished_at) {
                              <span>&bull;</span>
                              <span>{{ getDuration(job) }}</span>
                            }
                            @if (job.config['project_id']) {
                              <span>&bull;</span>
                              <span class="font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full bg-brand/15 text-brand-light border border-brand/30 cursor-default">{{ getProjectName(job.config['project_id']) }}</span>
                            }
                            @if (getModelSource(job); as src) {
                              @if (src.source_type !== 'hf_hub') {
                                <span>&bull;</span>
                                <span class="font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full cursor-default"
                                      [title]="src.local_path || ''"
                                      [class]="src.source_type === 'local_diffusers'
                                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                                        : 'bg-amber-500/10 text-amber-400 border border-amber-500/30'">
                                  {{ src.source_type === 'local_diffusers' ? 'LOCAL' : 'SAFETENSORS' }}
                                </span>
                              } @else if (src.skip_update) {
                                <span>&bull;</span>
                                <span class="font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/30 cursor-default">
                                  OFFLINE
                                </span>
                              }
                            }
                          </div>
                        </div>
                      </div>
                      <div class="flex items-center gap-3">
                        <span [class]="getStatusClass(job.status)"
                          class="text-[10px] px-2.5 py-1 rounded-full uppercase font-black tracking-widest shadow-sm">
                          {{ job.status }}
                        </span>
                        <div class="flex items-center gap-1">
                          <!-- Samples Toggle -->
                          <button (click)="toggleSamples(job.id)"
                            class="p-1.5 rounded-theme-lg transition-colors"
                            [class]="!jobsWithSamples().has(job.id) ? 'text-text-disabled cursor-not-allowed opacity-30' : samplesExpandedJobs().has(job.id) ? 'text-brand hover:text-brand/80' : 'text-text-subtle hover:text-brand'"
                            [disabled]="!jobsWithSamples().has(job.id)"
                            [title]="!jobsWithSamples().has(job.id) ? 'No samples available' : samplesExpandedJobs().has(job.id) ? 'Hide Samples' : 'Show Samples'">
                            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
                          </button>
                          <button (click)="toggleConfig(job.id)"
                            class="p-1.5 text-text-subtle hover:text-white rounded-theme-lg transition-colors" [title]="configExpandedJobs().has(job.id) ? 'Collapse Config' : 'View Config'">
                            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" [class.rotate-180]="configExpandedJobs().has(job.id)" class="transition-transform"><path d="m6 9 6 6 6-6"/></svg>
                          </button>
                          @if (job.status === JobStatus.FAILED || job.status === JobStatus.STOPPED) {
                            <button (click)="restartJob(job.id)"
                              class="p-1.5 text-blue-400 hover:bg-blue-400/10 rounded-theme-lg transition-colors" title="Restart Job">
                              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path><path d="M3 3v5h5"></path></svg>
                            </button>
                          }
                          <button (click)="deleteJob(job.id)"
                            class="p-1.5 text-text-subtle hover:text-danger hover:bg-danger/10 rounded-theme-lg transition-colors" title="Delete Job">
                            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"></path><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path></svg>
                          </button>
                        </div>
                      </div>
                    </div>

                    <!-- Training Summary Card (completed/stopped jobs with metrics) -->
                    @if ((job.status === JobStatus.COMPLETED || job.status === JobStatus.STOPPED) && job['avg_loss']) {
                      <div data-testid="training-summary-card"
                           class="flex justify-between w-full bg-base/25 p-3 rounded-theme-xl border border-surface-mid/20 mt-2">
                        
                        <!-- Col 1: Progress (Steps / Epoch) -->
                        <div class="flex flex-col space-y-1 w-24">
                          <div>
                            <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">Steps</span>
                            <span class="text-xs text-white font-mono font-bold">{{ job['completed_steps'] || '—' }}</span>
                          </div>
                          <div>
                            <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">Epoch</span>
                            <span class="text-xs text-purple-400 font-mono font-bold">{{ getFinalEpoch(job) }}</span>
                          </div>
                        </div>

                        <!-- Col 2: Optimization (Final / Best Loss) -->
                        <div class="flex flex-col space-y-1 w-28">
                          <div>
                            <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">Final Loss</span>
                            <span class="text-xs text-white font-mono font-bold">{{ job['avg_loss'] | number:'1.4-6' }}</span>
                          </div>
                          <div>
                            <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">Best Loss</span>
                            <div class="flex items-baseline gap-1">
                              <span class="text-xs text-green-400 font-mono font-bold">{{ job['min_loss'] | number:'1.4-6' }}</span>
                              @if (job['min_loss_step']) {
                                <span class="text-[8px] text-text-muted font-mono">@{{ job['min_loss_step'] }}</span>
                              }
                            </div>
                          </div>
                        </div>

                        <!-- Col 3: Throughput & Performance (Improvement / Avg Step) -->
                        <div class="flex flex-col space-y-1 w-28">
                          <div>
                            <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">Improvement</span>
                            @if (job['avg_loss'] && job['min_loss'] && job['avg_loss'] > 0) {
                              <span class="text-xs font-mono font-bold"
                                    [class]="job['min_loss'] < job['avg_loss'] * 0.9 ? 'text-green-400' : 'text-amber-400'">
                                {{ ((1 - job['min_loss'] / job['avg_loss']) * 100) | number:'1.1-1' }}%
                              </span>
                            } @else {
                              <span class="text-xs text-text-muted italic">—</span>
                            }
                          </div>
                          <div>
                            <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">Avg Step</span>
                            <span class="text-xs text-white font-mono font-bold">{{ (job['avg_step_time'] || 0) | number:'1.2-2' }}s</span>
                          </div>
                        </div>

                        <!-- Col 4: Settings (Optimizer / LR & Batch) -->
                        <div class="flex flex-col space-y-1 w-32">
                          <div>
                            <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">Optimizer</span>
                            <span class="text-xs text-sky-400 font-mono font-bold tracking-tight">{{ job.config['optimizer_type'] || 'AdamW' }}</span>
                          </div>
                          <div>
                            <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">LR / Batch</span>
                            <div class="flex items-baseline gap-1">
                              <span class="text-xs text-white font-mono font-bold">{{ formatLR(job.config['learning_rate']) }}</span>
                              <span class="text-[8px] text-text-disabled font-mono">/ BS {{ job.config['train_batch_size'] || 1 }}</span>
                            </div>
                          </div>
                        </div>

                        <!-- Col 5: Time (Train Time) -->
                        <div class="flex flex-col space-y-1 w-24 text-right items-end">
                          <div>
                            <span class="text-[9px] text-text-subtle uppercase font-bold tracking-widest block mb-0.5">Train Time</span>
                            <span class="text-xs text-white font-mono font-bold">{{ formatTrainingTime(job['training_seconds']) }}</span>
                          </div>
                        </div>

                      </div>
                    }

                    <!-- Samples Grid (Archive) -->
                    @if (samplesExpandedJobs().has(job.id)) {
                      <div class="bg-base/25 border border-surface-mid/20 rounded-theme-xl p-4 mt-2 mb-2 shadow-inner animate-in slide-in-from-top-2">
                        <div class="flex justify-between items-center mb-3">
                          <h4 class="text-xs font-bold text-text-muted uppercase tracking-wider">Sample Previews</h4>
                          <button (click)="loadSamples(job.id)" class="p-1 text-brand hover:text-brand/80 transition-colors group" title="Refresh samples">
                            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="group-hover:rotate-180 transition-transform duration-500"><path d="M21 2v6h-6"></path><path d="M3 12a9 9 0 0 1 15-6.7L21 8"></path><path d="M3 22v-6h6"></path><path d="M21 12a9 9 0 0 1-15 6.7L3 16"></path></svg>
                          </button>
                        </div>
                        @if (jobSamples().get(job.id); as samples) {
                          @if (samples.length > 0) {
                            <div class="grid grid-cols-6 gap-2 max-h-[33rem] overflow-y-auto scrollbar-thin scrollbar-thumb-surface-high pr-1">
                              @for (sample of samples; track sample.filename) {
                                <button (click)="openSampleModal(job.id, sample)" class="relative group rounded-theme-lg overflow-hidden border border-surface-mid hover:border-brand/50 transition-all hover:shadow-lg hover:shadow-brand/10 aspect-square bg-surface-high">
                                  <img [src]="getSampleImageUrl(job.id, sample.filename)" [alt]="'Step ' + sample.step" class="w-full h-full object-cover transition-transform group-hover:scale-105" loading="lazy">
                                  <div class="absolute bottom-0 left-0 right-0 bg-base/80 px-2 py-1">
                                    <span class="text-[10px] text-white font-mono font-bold">{{ sample.step === 999999 ? 'Final' : 'Step ' + sample.step }}</span>
                                  </div>
                                </button>
                              }
                            </div>
                          } @else {
                            <div class="text-center text-xs text-text-subtle py-8 italic">No samples found on disk.</div>
                          }
                        } @else {
                          <div class="text-center text-xs text-text-subtle py-8 italic">Loading samples...</div>
                        }
                      </div>
                    }
                    @if (configExpandedJobs().has(job.id)) {
                      <div class="bg-base/70 p-4 rounded-theme-xl border border-surface-mid animate-in slide-in-from-top-2 duration-300">
                        <div class="flex items-center justify-between mb-2">
                          <span class="text-[10px] text-text-subtle uppercase font-black tracking-widest">Job Configuration</span>
                          <div class="flex items-center gap-1">
                            <button (click)="onSaveAsTemplate(job)" type="button"
                              class="p-1.5 text-text-subtle hover:text-brand hover:bg-brand/10 rounded-theme-lg transition-colors" title="Save as Template">
                              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z"/></svg>
                            </button>
                            <button (click)="onReloadConfig(job)" type="button"
                              class="p-1.5 text-text-subtle hover:text-green-400 hover:bg-green-400/10 rounded-theme-lg transition-colors" title="Reload into Settings">
                              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>
                            </button>
                          </div>
                        </div>
                        <pre class="text-[10px] text-brand/80 font-mono overflow-auto max-h-40 scrollbar-thin scrollbar-thumb-surface-high">{{ job.config | json }}</pre>
                      </div>
                    }
                    @if (job.error) {
                      <div class="text-xs text-danger mt-1 flex items-start gap-2 bg-danger/10 p-2 rounded-theme-lg border border-danger/30">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="mt-0.5 shrink-0"><circle cx="12" cy="12" r="10"></circle><line x1="12" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                        <span class="break-words">{{ job.error }}</span>
                      </div>
                    }
                  </div>
                </div>
              }
            </div>
          }
        </div>
    </div>

    <!-- Sample Preview Modal -->
    @if (sampleModalData(); as modal) {
      <div class="fixed inset-0 z-50 flex items-center justify-center bg-overlay backdrop-blur-sm animate-in fade-in duration-200"
           (click)="closeSampleModal()">

        <!-- Previous Arrow -->
        @if (hasPrevSample()) {
          <button (click)="navigateSample(-1); $event.stopPropagation()"
            data-testid="sample-modal-prev"
            class="absolute left-4 z-10 p-3 text-white/60 hover:text-white bg-base/50 hover:bg-base/80 rounded-full transition-all hover:scale-110 backdrop-blur-sm">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m15 18-6-6 6-6"/></svg>
          </button>
        }

        <!-- Next Arrow -->
        @if (hasNextSample()) {
          <button (click)="navigateSample(1); $event.stopPropagation()"
            data-testid="sample-modal-next"
            class="absolute right-4 z-10 p-3 text-white/60 hover:text-white bg-base/50 hover:bg-base/80 rounded-full transition-all hover:scale-110 backdrop-blur-sm">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg>
          </button>
        }

        <div class="relative max-w-[90vw] max-h-[90vh] flex flex-col items-center gap-3" (click)="$event.stopPropagation()">
          <div class="flex items-center gap-3">
            <span class="text-sm text-white font-mono font-bold bg-base/70 px-3 py-1.5 rounded-full">{{ modal.sample.step === 999999 ? 'Final' : 'Step ' + modal.sample.step }}</span>
            <button (click)="closeSampleModal()" 
              class="p-2 text-text-muted hover:text-white bg-base/70 hover:bg-overlay rounded-full transition-colors" title="Close">
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
            </button>
          </div>
          <img [src]="getSampleImageUrl(modal.jobId, modal.sample.filename)" 
               [alt]="'Step ' + modal.sample.step"
               class="max-w-full max-h-[80vh] rounded-theme-xl shadow-2xl border border-surface-mid/50 object-contain">
        </div>
      </div>
    }
  `,

  styles: []
})
export class TrainingJobQueueComponent implements OnInit {
  jobService = inject(JobService);
  projectService = inject(ProjectService);
  private modelService = inject(ModelService);
  private rtc = inject(RuntimeConfigService);
  jobs = signal<Job[]>([]);
  historicalJobs = signal<Job[]>([]);
  JobStatus = JobStatus;

  // Derived views: active queue vs archive
  private readonly ACTIVE_STATUSES = new Set([JobStatus.PENDING, JobStatus.RUNNING, JobStatus.PAUSED]);
  activeJobs = computed(() => this.jobs().filter(j => this.ACTIVE_STATUSES.has(j.status)));

  archivedJobs = computed(() => this.historicalJobs());
  archiveExpanded = signal<boolean>(false);
  archiveProjectScope = signal<boolean>(true);
  archiveProjectFilter = signal<string>('all');

  // Output events for config actions
  saveAsTemplate = output<any>();
  reloadConfig = output<any>();

  configExpandedJobs = signal<Set<string>>(new Set());
  chartExpandedJobs = signal<Set<string>>(new Set());
  samplesExpandedJobs = signal<Set<string>>(new Set());
  samplingPausedJobs = signal<Set<string>>(new Set());
  samplingCadence = signal<Map<string, number>>(new Map());
  customCadenceMode = signal<Set<string>>(new Set());
  jobsWithSamples = signal<Set<string>>(new Set());
  jobSamples = signal<Map<string, any[]>>(new Map());
  sampleModalData = signal<{ jobId: string; sample: any } | null>(null);
  sampleCacheBuster = signal<number>(Date.now());

  currentNow = signal<number>(Date.now());
  smoothingFactor = signal<number>(0.9);
  smoothingMode = signal<SmoothingMode>('ema');
  autoQueue = signal<boolean>(false);
  stopModalJobId = signal<string | null>(null);

  // Model source overrides cache (definition_id → source info)
  jobModelSources = signal<Map<string, ModelSourceOverride>>(new Map());

  // Track if we just triggered a start to prevent double-firing before refresh updates status
  private startingJobId: string | null = null;

  wsService = inject(WebSocketService);
  private destroyRef = inject(DestroyRef);
  private jobStore = inject(JobStore);

  // Tracks whether the JobStore has been seeded at least once. Until then,
  // the existing loadJobs/loadHistory subscribers are authoritative for first
  // render; after seeding, the effect below reconciles deletions from the
  // store into the local signals (so optimistic delete drops the row this
  // tick) without clobbering local state from WS job_update events.
  private storeSeeded = false;

  constructor() {
    // Reconcile JobStore deletions into the local jobs/historicalJobs signals.
    // The store is canonical for *which jobs exist*; the local signals carry
    // richer per-job state (logs, sample lists) that the store doesn't track,
    // so we only PRUNE rows missing from the store rather than full-overwrite.
    // This is the minimal hook needed to make optimistic delete visible while
    // leaving start/stop/pause refresh paths untouched.
    effect(() => {
      const all = this.jobStore.entities();
      if (all.length === 0 && !this.storeSeeded) return;
      this.storeSeeded = true;
      const knownIds = new Set(all.map(j => j.id));
      this.jobs.update(rows => rows.filter(j => knownIds.has(j.id)));
      this.historicalJobs.update(rows => rows.filter(j => knownIds.has(j.id)));
    });
  }

  ngOnInit() {
    // Restore preference
    const saved = localStorage.getItem('autoQueueEnabled');
    if (saved) this.autoQueue.set(saved === 'true');

    // Restore archive scope preference
    const savedScope = localStorage.getItem('archiveProjectScope');
    if (savedScope !== null) this.archiveProjectScope.set(savedScope === 'true');

    this.refreshAll();

    // Polling fallback (reduced frequency - 30s) just to sync deleted jobs or misses
    interval(30000).pipe(takeUntilDestroyed(this.destroyRef)).subscribe(() => this.refreshAll());
    interval(1000).pipe(takeUntilDestroyed(this.destroyRef)).subscribe(() => this.currentNow.set(Date.now()));

    // Subscribe to Real-time Events
    this.wsService.messages$.pipe(takeUntilDestroyed(this.destroyRef)).subscribe(event => {
      this.handleWsEvent(event);
    });

    // Refresh jobs immediately when server restarts (clears stale data)
    this.wsService.serverRestarted$.pipe(takeUntilDestroyed(this.destroyRef)).subscribe(() => {
      console.log('[JobQueue] Server restarted — refreshing jobs');
      this.loadJobs();
    });
  }

  handleWsEvent(event: any) {
    if (event.type === 'job_update') {
      const updatedJob = event.payload as Job;
      this.jobs.update(current => {
        const index = current.findIndex(j => j.id === updatedJob.id);
        if (index !== -1) {
          const newJobs = [...current];
          // Preserve logs if not present in payload to avoid clearing them
          const existingLogs = newJobs[index].logs || [];
          newJobs[index] = { ...updatedJob, logs: updatedJob.logs || existingLogs };
          return newJobs;
        } else {
          // New job
          return [updatedJob, ...current];
        }
      });
      // Check auto queue on status changes
      if (updatedJob.status === JobStatus.COMPLETED || updatedJob.status === JobStatus.FAILED) {
        this.processAutoQueue(this.jobs());
      }
    } else if (event.type === 'job_log') {
      const { job_id, message, timestamp } = event.payload;
      this.jobs.update(current => {
        const job = current.find(j => j.id === job_id);
        if (job) {
          if (!job.logs) job.logs = [];
          job.logs.push(message);
          const index = current.indexOf(job);
          const newJobs = [...current];
          newJobs[index] = { ...job, logs: [...job.logs] };
          return newJobs;
        }
        return current;
      });
      // Mark job as having samples + auto-refresh grid
      if (message && message.includes('sample_generated')) {
        this.jobsWithSamples.update(prev => { const n = new Set(prev); n.add(job_id); return n; });
        if (this.samplesExpandedJobs().has(job_id)) {
          this.loadSamples(job_id);
        }
      }
    } else if (event.type === 'job_warning') {
      const { job_id, message } = event.payload;
      this.jobs.update(current => {
        const job = current.find(j => j.id === job_id);
        if (job) {
          if (!job.warnings) job.warnings = [];
          // Deduplicate identical warnings
          if (!job.warnings.includes(message)) {
            job.warnings.push(message);
          }
          const index = current.indexOf(job);
          const newJobs = [...current];
          newJobs[index] = { ...job, warnings: [...job.warnings] };
          return newJobs;
        }
        return current;
      });
    }
  }

  toggleConfig(id: string) {
    this.configExpandedJobs.update(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  toggleChart(id: string) {
    this.chartExpandedJobs.update(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  toggleSamples(id: string) {
    this.samplesExpandedJobs.update(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
        this.loadSamples(id);
        // Load sampling pause status + cadence when expanding
        this.jobService.getSamplingStatus(id).subscribe({
          next: (res) => {
            if (res.sampling_paused) {
              this.samplingPausedJobs.update(prev => { const n = new Set(prev); n.add(id); return n; });
            }
          }
        });
        this.jobService.getSamplingCadence(id).subscribe({
          next: (res) => {
            this.samplingCadence.update(prev => { const n = new Map(prev); n.set(id, res.interval); return n; });
          }
        });
      }
      return next;
    });
  }

  toggleSamplingPause(jobId: string) {
    const isPaused = this.samplingPausedJobs().has(jobId);
    const action$ = isPaused
      ? this.jobService.resumeSampling(jobId)
      : this.jobService.pauseSampling(jobId);
    action$.subscribe({
      next: () => {
        this.samplingPausedJobs.update(prev => {
          const next = new Set(prev);
          if (isPaused) next.delete(jobId);
          else next.add(jobId);
          return next;
        });
      }
    });
  }

  onCadenceChange(jobId: string, event: Event) {
    const value = (event.target as HTMLSelectElement).value;
    if (value === 'custom') {
      this.customCadenceMode.update(prev => { const n = new Set(prev); n.add(jobId); return n; });
      return;
    }
    const interval = parseInt(value, 10);
    if (interval > 0) {
      this.jobService.setSamplingCadence(jobId, interval).subscribe({
        next: () => {
          this.samplingCadence.update(prev => { const n = new Map(prev); n.set(jobId, interval); return n; });
        }
      });
    }
  }

  applyCustomCadence(jobId: string, event: Event) {
    const input = event.target as HTMLInputElement;
    const interval = parseInt(input.value, 10);
    if (!interval || interval <= 0) return;
    this.jobService.setSamplingCadence(jobId, interval).subscribe({
      next: () => {
        this.samplingCadence.update(prev => { const n = new Map(prev); n.set(jobId, interval); return n; });
        this.customCadenceMode.update(prev => { const n = new Set(prev); n.delete(jobId); return n; });
      }
    });
  }

  loadSamples(jobId: string) {
    this.jobService.getJobSamples(jobId).subscribe({
      next: (samples) => {
        this.jobSamples.update(prev => {
          const next = new Map(prev);
          next.set(jobId, samples);
          return next;
        });
        if (samples && samples.length > 0) {
          this.jobsWithSamples.update(prev => { const n = new Set(prev); n.add(jobId); return n; });
        }
        // Bust browser cache so <img> tags re-fetch from server
        this.sampleCacheBuster.set(Date.now());
      },
      error: () => {
        this.jobSamples.update(prev => {
          const next = new Map(prev);
          next.set(jobId, []);
          return next;
        });
      }
    });
  }

  getSampleImageUrl(jobId: string, filename: string): string {
    return `${this.rtc.apiUrl}/jobs/${jobId}/samples/${filename}?t=${this.sampleCacheBuster()}`;
  }

  openSampleModal(jobId: string, sample: any) {
    this.sampleModalData.set({ jobId, sample });
  }

  closeSampleModal() {
    this.sampleModalData.set(null);
  }

  /** Index of current sample in the jobSamples list (most recent first). */
  private currentSampleIndex(): number {
    const modal = this.sampleModalData();
    if (!modal) return -1;
    const samples = this.jobSamples().get(modal.jobId);
    if (!samples) return -1;
    return samples.findIndex((s: any) => s.filename === modal.sample.filename);
  }

  /** Can navigate to a newer sample (toward index 0). */
  hasPrevSample = computed(() => {
    const idx = this.currentSampleIndex();
    return idx > 0;
  });

  /** Can navigate to an older sample (toward end of list). */
  hasNextSample = computed(() => {
    const modal = this.sampleModalData();
    if (!modal) return false;
    const samples = this.jobSamples().get(modal.jobId);
    if (!samples) return false;
    const idx = this.currentSampleIndex();
    return idx >= 0 && idx < samples.length - 1;
  });

  /** Navigate to prev (-1) or next (+1) sample in the list. */
  navigateSample(direction: -1 | 1) {
    const modal = this.sampleModalData();
    if (!modal) return;
    const samples = this.jobSamples().get(modal.jobId);
    if (!samples) return;
    const idx = this.currentSampleIndex();
    const newIdx = idx + direction;
    if (newIdx < 0 || newIdx >= samples.length) return;
    this.sampleModalData.set({ jobId: modal.jobId, sample: samples[newIdx] });
  }

  @HostListener('document:keydown.escape')
  onEscapeKey() {
    if (this.sampleModalData()) {
      this.closeSampleModal();
    }
  }

  @HostListener('document:keydown.arrowleft')
  onArrowLeft() {
    if (this.sampleModalData()) {
      this.navigateSample(-1);
    }
  }

  @HostListener('document:keydown.arrowright')
  onArrowRight() {
    if (this.sampleModalData()) {
      this.navigateSample(1);
    }
  }

  refreshAll() {
    this.loadJobs();
    this.loadHistory();
  }

  loadHistory() {
    const filter = this.archiveProjectFilter();
    const projectId = (filter && filter !== 'all') ? filter : null;
    this.jobService.listJobHistory(projectId).subscribe(jobs => {
      this.historicalJobs.set(jobs);
    });
    // Also seed the JobStore so optimistic deleteJob() can prune archived
    // rows. JobStore.loadHistory() currently ignores the project filter
    // (no-arg listJobHistory) — acceptable temporary duplication until the
    // store fully owns the archive view (Phase 5+).
    void this.jobStore.loadHistory();
  }

  loadJobs() {
    // Also seed the JobStore so optimistic deleteJob() works on the active
    // queue. One extra HTTP call until the store fully owns this list.
    void this.jobStore.loadAll();
    this.jobService.listJobs().subscribe({
      next: (jobs) => {
        this.jobs.set(jobs);
        this.processAutoQueue(jobs);
        this.loadModelSources(jobs);
        // Pre-check sample availability for jobs with sampling configured
        for (const job of jobs) {
          if (job.config?.['sample_every_n_steps'] > 0 && !this.jobsWithSamples().has(job.id)) {
            this.jobService.getJobSamples(job.id).subscribe({
              next: (samples) => {
                if (samples && samples.length > 0) {
                  this.jobsWithSamples.update(prev => { const n = new Set(prev); n.add(job.id); return n; });
                }
              }
            });
          }
        }
      },
      error: (err) => console.error('Failed to load jobs', err)
    });
  }

  toggleAutoQueue() {
    this.autoQueue.update(v => !v);
    localStorage.setItem('autoQueueEnabled', String(this.autoQueue()));
    // Trigger check immediately
    this.processAutoQueue(this.jobs());
  }

  private processAutoQueue(jobs: Job[]) {
    if (!this.autoQueue()) return;

    // 1. Check if anything is running
    const runningJob = jobs.find(j => j.status === JobStatus.RUNNING);

    if (runningJob) {
      // Reset our internal tracker if the job we started is now officially running
      if (this.startingJobId === runningJob.id) {
        this.startingJobId = null;
      }
      return;
    }

    // 2. If nothing running, find next pending
    // Sort by created_at (Oldest First - FIFO)
    const pendingJobs = jobs.filter(j => j.status === JobStatus.PENDING)
      .sort((a, b) => a.created_at - b.created_at);

    if (pendingJobs.length > 0) {
      const nextJob = pendingJobs[0];

      // Prevent spamming the start command for the same job
      if (this.startingJobId !== nextJob.id) {
        this.startingJobId = nextJob.id;
        this.startJob(nextJob.id);
      }
    }
  }

  private parseLogLine(line: string): any | null {
    const prefix = "STEP_LOG:";
    let jsonStr = line;
    if (line.includes(prefix)) {
      jsonStr = line.split(prefix)[1];
    } else if (!line.trim().startsWith('{')) {
      return null;
    }
    try {
      return JSON.parse(jsonStr);
    } catch {
      return null;
    }
  }

  /** Throttled fields emitted every N steps — carry forward last known value. */
  private static readonly CARRY_FORWARD_KEYS = [
    'vram_allocated_mb', 'vram_reserved_mb', 'amp_scale', 'resolution',
  ];

  getLatestMetrics(job: Job): any {
    if (!job.logs || job.logs.length === 0) return null;
    for (let i = job.logs.length - 1; i >= 0; i--) {
      const metrics = this.parseLogLine(job.logs[i]);
      if (metrics && metrics.status === 'training') {
        const totalSteps = job.config['max_train_steps'] || Math.round(metrics.step / (metrics.progress / 100)) || '?';

        // Back-fill throttled fields from recent log history
        const carryForward: Record<string, any> = {};
        const keysNeeded = new Set(
          TrainingJobQueueComponent.CARRY_FORWARD_KEYS.filter(k => metrics[k] == null)
        );
        if (keysNeeded.size > 0) {
          const lookback = Math.min(50, i); // Increased lookback to 50 for more tolerant caching of throttled values
          for (let j = i - 1; j >= i - lookback && keysNeeded.size > 0; j--) {
            const prev = this.parseLogLine(job.logs[j]);
            if (!prev) continue;
            for (const key of [...keysNeeded]) {
              if (prev[key] != null) {
                carryForward[key] = prev[key];
                keysNeeded.delete(key);
              }
            }
          }
        }

        return {
          ...metrics,
          ...carryForward,
          total_steps: totalSteps
        };
      }
    }
    return null;
  }

  // Cache chart data to avoid re-parsing on every change detection cycle
  private chartCache = new Map<string, { len: number; data: ChartDataPoint[] }>();

  getChartData(job: Job): ChartDataPoint[] | null {
    if (!job.logs) return null;
    const logsLen = job.logs.length;

    // Return cached data if logs haven't changed
    const cached = this.chartCache.get(job.id);
    if (cached && cached.len === logsLen) {
      return cached.data.length >= 2 ? cached.data : null;
    }

    const points: ChartDataPoint[] = [];
    const startStep = 5;
    for (const line of job.logs) {
      const m = this.parseLogLine(line);
      if (m && m.step >= startStep && typeof m.loss === 'number') {
        points.push({
          step: m.step,
          loss: m.loss,
          lr: m.learning_rate ?? 0,
          grad_norm: m.grad_norm ?? undefined,
          d_estimate: m.d_estimate ?? undefined,
        });
      }
    }

    this.chartCache.set(job.id, { len: logsLen, data: points });
    return points.length >= 2 ? points : null;
  }

  onPlateauDetected(job: Job, event: { step: number; loss: number }) {
    const warning = `⚠️ Loss appears to have plateaued at ~${event.loss} since step ${event.step}. This may indicate a model loading or configuration issue.`;
    this.jobs.update(current => {
      const target = current.find(j => j.id === job.id);
      if (target) {
        if (!target.warnings) target.warnings = [];
        if (!target.warnings.some(w => w.includes('plateaued'))) {
          target.warnings.push(warning);
          const index = current.indexOf(target);
          const newJobs = [...current];
          newJobs[index] = { ...target, warnings: [...target.warnings] };
          return newJobs;
        }
      }
      return current;
    });
  }

  getDuration(job: Job): string {
    if (!job.started_at) return '0:00';
    const end = job.finished_at ? job.finished_at * 1000
      : job.paused_at ? job.paused_at * 1000
        : this.currentNow();
    const seconds = Math.floor((end - (job.started_at * 1000)) / 1000);
    if (seconds < 0) return '0:00';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    return `${m}:${s.toString().padStart(2, '0')}`;
  }

  formatEta(seconds: number): string {
    if (!seconds || seconds < 0) return '--:--';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m ${s}s`;
  }

  /** Loss velocity status: compares recent vs earlier loss average. Uses a larger window (50 steps) for stability. */
  getLossStatus(job: Job): { icon: string, text: string, colorClass: string, tooltip: string } | null {
    if (!job.logs || job.logs.length < 50) return null;
    const window = 50;
    const losses: number[] = [];
    for (let i = job.logs.length - 1; i >= 0 && losses.length < window * 2; i--) {
      const m = this.parseLogLine(job.logs[i]);
      if (m?.loss != null) losses.unshift(m.loss);
    }
    if (losses.length < window) return null;
    const recent = losses.slice(-window).reduce((a, b) => a + b, 0) / window;
    const earlier = losses.slice(0, window).reduce((a, b) => a + b, 0) / window;
    const delta = (recent - earlier) / Math.max(earlier, 1e-8);
    
    let icon = '🟡';
    let text = 'Plateau';
    let colorClass = 'text-amber-400 bg-amber-500/10 border-amber-500/30';
    if (delta < -0.01) {
      icon = '🟢';
      text = 'Converging';
      colorClass = 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
    } else if (delta > 0.02) {
      icon = '🔴';
      text = 'Diverging';
      colorClass = 'text-danger bg-danger/10 border-danger/30';
    }

    return {
      icon,
      text,
      colorClass,
      tooltip: `Evaluating over last ${window} steps.\nRecent avg: ${recent.toFixed(5)}\nEarlier avg: ${earlier.toFixed(5)}`
    };
  }

  /** Format grad norm: use scientific notation for very large values. */
  formatGradNorm(gn: number): string {
    if (gn == null) return '';
    if (gn >= 1000) return gn.toExponential(1);
    if (gn >= 1) return gn.toFixed(2);
    return gn.toFixed(4);
  }

  /** Format training duration in seconds to human-readable. */
  formatTrainingTime(seconds: number | null | undefined): string {
    if (!seconds || seconds <= 0) return '—';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m`;
  }

  startJob(id: string) {
    this.jobService.startJob(id).subscribe(() => this.loadJobs());
  }

  stopJob(id: string) {
    this.jobService.stopJob(id).subscribe(() => this.loadJobs());
  }

  restartJob(id: string) {
    this.jobService.restartJob(id).subscribe({
      next: () => this.loadJobs(),
      error: (e) => console.error('Failed to restart job', e)
    });
  }

  deleteJob(id: string) {
    // Optimistic delete via JobStore: the store updates synchronously
    // (row disappears from store.entities() this tick), the effect above
    // prunes our local jobs/historicalJobs signals so the template
    // re-renders immediately. JobStore handles rollback + toast on failure.
    void this.jobStore.deleteJob(id);
  }

  getStatusClass(status: JobStatus): string {
    switch (status) {
      case JobStatus.PENDING: return 'bg-warning/20 text-warning border border-warning/30';
      case JobStatus.RUNNING: return 'bg-brand/20 text-brand border border-brand/30 animate-pulse';
      case JobStatus.PAUSED: return 'bg-amber-500/20 text-amber-400 border border-amber-500/30';
      case JobStatus.COMPLETED: return 'bg-success/20 text-success border border-success/30';
      case JobStatus.FAILED: return 'bg-danger/20 text-danger border border-danger/30';
      case JobStatus.STOPPED: return 'bg-warning/20 text-warning border border-warning/30';
      default: return 'bg-surface-high text-text-secondary';
    }
  }

  pauseJob(id: string) {
    this.jobService.pauseJob(id).subscribe(() => this.loadJobs());
  }

  resumeJob(id: string) {
    this.jobService.resumeJob(id).subscribe(() => this.loadJobs());
  }

  toggleArchiveScope() {
    const newScope = !this.archiveProjectScope();
    this.archiveProjectScope.set(newScope);
    localStorage.setItem('archiveProjectScope', String(newScope));
    this.loadHistory();
  }

  onArchiveScopeChange(value: string) {
    const pid = (value && value !== 'all') ? value : null;
    this.archiveProjectFilter.set(value);
    this.projectService.activeJobsProject.set(pid);
    this.archiveProjectScope.set(!!pid);
    localStorage.setItem('archiveProjectScope', String(!!pid));
    this.loadHistory();
  }

  getProjectName(projectId: string): string {
    const project = this.projectService.allProjects().find(p => p.id === projectId);
    return project?.name || projectId.slice(0, 8);
  }

  getModelSource(job: Job): ModelSourceOverride | null {
    const defId = job.config['definition_id'] || job.plugin_id;
    return this.jobModelSources().get(defId) || null;
  }

  /** Fetch model source overrides for all unique definition IDs in current jobs */
  private loadModelSources(jobs: Job[]) {
    const defIds = new Set(jobs.map(j => j.config['definition_id'] || j.plugin_id).filter(Boolean));
    const cached = this.jobModelSources();
    for (const defId of defIds) {
      if (!cached.has(defId)) {
        this.modelService.getModelSource(defId).subscribe({
          next: (src) => {
            this.jobModelSources.update(prev => {
              const next = new Map(prev);
              next.set(defId, src);
              return next;
            });
          },
          error: () => { /* No override = HF Hub default, skip */ }
        });
      }
    }
  }

  toggleArchive() {
    const willExpand = !this.archiveExpanded();
    this.archiveExpanded.set(willExpand);

    // Pre-check sample availability when expanding
    if (willExpand) {
      this.loadHistory(); // Load from API on expand

      // We need to wait for historicalJobs to populate, so subscribe or handle after
      setTimeout(() => {
        for (const job of this.archivedJobs()) {
          if (!this.jobsWithSamples().has(job.id)) {
            this.jobService.getJobSamples(job.id).subscribe({
              next: (samples) => {
                if (samples && samples.length > 0) {
                  this.jobsWithSamples.update(prev => { const n = new Set(prev); n.add(job.id); return n; });
                }
              }
            });
          }
        }
      }, 300);
    }
  }

  openStopModal(id: string) {
    this.stopModalJobId.set(id);
  }

  closeStopModal() {
    this.stopModalJobId.set(null);
  }

  softStopJob(id: string) {
    this.closeStopModal();
    this.jobService.softStopJob(id).subscribe(() => this.loadJobs());
  }

  hardStopJob(id: string) {
    this.closeStopModal();
    this.jobService.stopJob(id).subscribe(() => this.loadJobs());
  }

  onSaveAsTemplate(job: Job) {
    const name = prompt('Template name:');
    if (!name?.trim()) return;
    this.saveAsTemplate.emit({ name: name.trim(), config: job.config, definition_id: job.config['definition_id'] || job.plugin_id });
  }

  onReloadConfig(job: Job) {
    this.reloadConfig.emit(job.config);
  }

  getFinalEpoch(job: Job): string {
    const steps = job.completed_steps;
    const config = job.config;
    if (!steps || !config) return '—';

    // If active memory object still has the exact metric
    const metrics = this.getLatestMetrics(job);
    if (metrics && metrics.epoch !== undefined) {
      return metrics.epoch.toString();
    }

    // Use exact value if available via V6 schema logic
    if (job.completed_epochs !== undefined && job.completed_epochs !== null) {
      return job.completed_epochs.toFixed(2);
    }
    
    // Legacy fallback
    return '—';
  }

  formatLR(lr: any): string {
    if (lr == null || lr === 0) return '—';
    const n = Number(lr);
    if (n < 0.0001) return n.toExponential(1);
    return n.toString();
  }
}
