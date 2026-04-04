import { Component, OnInit, inject, input, output, computed, effect, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ProjectService, Project } from '../../../services/project.service';
import { DatasetService, Dataset } from '../../../services/dataset';
import { ToastService } from '../../../services/toast';
import { GeneralTemplatesComponent } from '../general-templates/general-templates';

@Component({
  selector: 'app-project-detail',
  standalone: true,
  imports: [FormsModule, GeneralTemplatesComponent],
  template: `
    <div class="space-y-8 animate-in fade-in duration-300">
      
      <!-- Header -->
      <div class="bg-surface-low/50 border border-border-default rounded-theme-xl p-8 shadow-xl">
          <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
              <div class="flex items-center gap-4">
                  <button (click)="back.emit()" class="p-2 -ml-2 text-text-muted hover:text-white hover:bg-surface-mid rounded-theme-md transition-colors">
                      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline></svg>
                  </button>
                  <div>
                      <div class="flex items-center gap-3 mb-2">
                        <div class="w-4 h-4 rounded-full" [style.backgroundColor]="project()?.color || '#3b82f6'"></div>
                        <h2 class="text-2xl font-bold text-white leading-none">{{ project()?.name || 'Project Details' }}</h2>
                      </div>
                      <p class="text-text-muted">{{ project()?.description || 'No description' }}</p>
                  </div>
              </div>
          </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          <!-- Left Column: Datasets & Templates -->
          <div class="lg:col-span-2 space-y-6">
              
              <!-- Stats Row -->
              <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-4">
                      <div class="text-sm text-text-muted mb-1">Datasets</div>
                      <div class="text-2xl font-bold text-white">{{ project()?.stats?.datasets || 0 }}</div>
                  </div>
                  <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-4">
                      <div class="text-sm text-text-muted mb-1">Training Jobs</div>
                      <div class="text-2xl font-bold text-white">{{ project()?.stats?.jobs || 0 }}</div>
                  </div>
                  <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-4">
                      <div class="text-sm text-text-muted mb-1">Training Templates</div>
                      <div class="text-2xl font-bold text-brand-light">{{ project()?.stats?.training_templates || 0 }}</div>
                  </div>
                  <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-4">
                      <div class="text-sm text-text-muted mb-1">Caption Templates</div>
                      <div class="text-2xl font-bold text-brand-light">{{ project()?.stats?.captioning_templates || 0 }}</div>
                  </div>
              </div>

              <!-- Datasets Association Area -->
              <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-6 shadow-lg">
                  <div class="flex items-center justify-between mb-4">
                    <div>
                      <h3 class="text-lg font-bold text-white">Datasets</h3>
                      <p class="text-text-muted text-sm mt-1">Associate datasets with this project for quick access.</p>
                    </div>
                    @if (!showDatasetPicker()) {
                      <button (click)="showDatasetPicker.set(true)"
                              class="flex items-center gap-1.5 bg-brand/20 hover:bg-brand/30 border border-brand/40 text-brand-light px-3 py-1.5 rounded-theme-md transition-all text-sm">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                        Add Dataset
                      </button>
                    }
                  </div>

                  <!-- Dataset picker dropdown -->
                  @if (showDatasetPicker()) {
                    <div class="mb-4 p-3 bg-surface-mid/50 border border-brand/30 rounded-theme-md">
                      <label class="text-xs text-text-subtle font-bold uppercase tracking-wider mb-2 block">Select Dataset</label>
                      <div class="flex gap-2">
                        <select [(ngModel)]="selectedDatasetToAdd" class="flex-1 bg-surface-low border border-surface-high text-white text-sm rounded-theme-md px-3 py-1.5 outline-none focus:border-brand">
                          <option value="">Choose a dataset...</option>
                          @for (ds of availableDatasets(); track ds.id) {
                            <option [value]="ds.id">{{ ds.name }}</option>
                          }
                        </select>
                        <button (click)="addDataset()" [disabled]="!selectedDatasetToAdd"
                                class="bg-brand/80 hover:bg-brand text-white px-3 py-1.5 rounded-theme-md text-sm transition-colors disabled:opacity-40">
                          Add
                        </button>
                        <button (click)="showDatasetPicker.set(false)"
                                class="text-text-muted hover:text-white px-2 py-1.5 rounded-theme-md text-sm transition-colors">
                          Cancel
                        </button>
                      </div>
                    </div>
                  }

                  <!-- Associated datasets list -->
                  <div class="space-y-2">
                    @for (ds of projectDatasets(); track ds.id) {
                      <div class="flex items-center justify-between p-3 bg-surface-mid border border-surface-high rounded-theme-md">
                        <div class="flex items-center gap-3">
                          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-brand-light">
                            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
                          </svg>
                          <span class="text-white text-sm font-medium">{{ ds.name }}</span>
                        </div>
                        <button (click)="removeDataset(ds.id)"
                                class="text-text-muted hover:text-danger transition-colors p-1" title="Remove from project">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                        </button>
                      </div>
                    } @empty {
                      <div class="text-center text-text-subtle p-4 bg-surface-high rounded-theme-md text-sm">
                        No datasets associated yet.
                      </div>
                    }
                  </div>
              </div>

              <!-- General Templates Branching Area -->
              <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-6 shadow-lg">
                  <h3 class="text-lg font-bold text-white mb-2">Branch Global Templates</h3>
                  <p class="text-text-muted mb-6 text-sm">
                      Copy "General" (global) templates into this project to customize them without affecting other projects.
                  </p>
                  
                  <app-general-templates [projectId]="projectId()"></app-general-templates>
              </div>

          </div>

          <!-- Right Column: Settings -->
          <div class="space-y-6">
              <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-6 shadow-lg">
                  <h3 class="text-lg font-bold text-white mb-4">Project Settings</h3>
                  
                  <div class="space-y-4">
                      <div>
                          <label class="block text-sm font-medium text-text-muted mb-1">Name</label>
                          <input type="text" [ngModel]="project()?.name" disabled
                                 class="w-full bg-surface-mid border border-border-default rounded-theme-md px-3 py-2 text-white focus:outline-none focus:border-brand/50 disabled:opacity-50">
                      </div>
                      <div>
                          <label class="block text-sm font-medium text-text-muted mb-1">Description</label>
                          <textarea [ngModel]="project()?.description" disabled rows="3"
                                    class="w-full bg-surface-mid border border-border-default rounded-theme-md px-3 py-2 text-white focus:outline-none focus:border-brand/50 disabled:opacity-50"></textarea>
                      </div>
                      <div>
                          <label class="block text-sm font-medium text-text-muted mb-1">Color</label>
                          <div class="flex items-center gap-3">
                              <input type="color" [ngModel]="project()?.color" disabled
                                     class="w-10 h-10 rounded border-0 bg-transparent disabled:opacity-50">
                          </div>
                      </div>
                      
                      <!-- Editable mode toggle button would go here -->
                  </div>
              </div>
          </div>
      </div>
    </div>
  `
})
export class ProjectDetailComponent implements OnInit {
  private projectService = inject(ProjectService);
  private datasetService = inject(DatasetService);
  private toast = inject(ToastService);
  
  projectId = input.required<string>();
  back = output<void>();

  // Dataset association state
  allDatasets = signal<Dataset[]>([]);
  projectDatasets = signal<any[]>([]);
  showDatasetPicker = signal(false);
  selectedDatasetToAdd = '';

  // A local signal to hold the fetched project details
  project = computed(() => {
    return this.projectService.allProjects().find(p => p.id === this.projectId()) || null;
  });

  // Compute available datasets (not yet associated with this project)
  availableDatasets = computed(() => {
    const associated = new Set(this.projectDatasets().map(d => d.id));
    return this.allDatasets().filter(d => !associated.has(d.id));
  });

  ngOnInit() {
    this.loadDatasets();
    this.loadProjectDatasets();
  }

  private loadDatasets() {
    this.datasetService.listDatasets().subscribe(ds => this.allDatasets.set(ds));
  }

  private loadProjectDatasets() {
    this.projectService.getProjectDatasets(this.projectId()).subscribe({
      next: (ds) => this.projectDatasets.set(ds),
      error: () => this.projectDatasets.set([])
    });
  }

  addDataset() {
    if (!this.selectedDatasetToAdd) return;
    this.projectService.addProjectDataset(this.projectId(), this.selectedDatasetToAdd).subscribe({
      next: () => {
        this.toast.success('Dataset added to project.');
        this.loadProjectDatasets();
        this.selectedDatasetToAdd = '';
        this.showDatasetPicker.set(false);
        // Refresh project stats
        this.projectService.loadProjects();
      },
      error: (err: any) => this.toast.error('Failed to add dataset: ' + (err.error?.detail || err.message))
    });
  }

  removeDataset(datasetId: string) {
    this.projectService.removeProjectDataset(this.projectId(), datasetId).subscribe({
      next: () => {
        this.toast.success('Dataset removed from project.');
        this.loadProjectDatasets();
        this.projectService.loadProjects();
      },
      error: (err: any) => this.toast.error('Failed to remove dataset: ' + (err.error?.detail || err.message))
    });
  }
}
