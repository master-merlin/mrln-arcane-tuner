import { Component, OnInit, OnDestroy, inject, signal, computed, effect } from '@angular/core';
import { DatasetService, Dataset } from '../../../services/dataset';
import { ToastService } from '../../../services/toast';
import { WebSocketService } from '../../../services/websocket.service';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { FormsModule } from '@angular/forms';
import { ProjectService } from '../../../services/project.service';
import { DatasetViewerComponent } from '../dataset-viewer/dataset-viewer';

// Sub-components
import { DatasetToolbarComponent } from './components/dataset-toolbar';
import { DatasetCardComponent } from './components/dataset-card';
import { DatasetEmptyStateComponent } from './components/dataset-empty-state';
import { DatasetFormModalComponent } from './components/dataset-form-modal';
import { ViewerRescanModalComponent } from '../dataset-viewer/components/viewer-rescan-modal';
import { ViewerCacheAdminModalComponent } from '../dataset-viewer/components/viewer-cache-admin-modal';
import { DatasetRescanOptionsModalComponent } from './components/dataset-rescan-options-modal';
import { DatasetSingleRescanModalComponent } from './components/dataset-single-rescan-modal';

@Component({
  selector: 'app-dataset-manager',
  standalone: true,
  imports: [
    DatasetViewerComponent,
    DatasetToolbarComponent,
    DatasetCardComponent,
    DatasetEmptyStateComponent,
    DatasetFormModalComponent,
    ViewerRescanModalComponent,
    ViewerCacheAdminModalComponent,
    DatasetRescanOptionsModalComponent,
    DatasetSingleRescanModalComponent,
    FormsModule
  ],
  template: `
    <div class="h-full flex flex-col">
      
      @if (showViewer()) {
          <app-dataset-viewer 
             [datasetName]="currentViewerDataset()" 
             (close)="closeViewer()">
          </app-dataset-viewer>
      } @else {
          <div class="space-y-8">
              <!-- Header -->
              <div class="bg-surface-low/50 border border-border-default rounded-theme-xl p-8 shadow-xl">
                  <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                      <div>
                          <h2 class="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-brand"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>
                              Dataset Library
                          </h2>
                          <p class="text-text-muted">Manage, inspect, and prepare your training datasets.</p>
                      </div>
                      <div class="flex items-center gap-3">
                          <span class="text-xs uppercase tracking-widest text-text-subtle font-bold">Project Context:</span>
                          <select [ngModel]="projectService.activeDatasetProject()" (ngModelChange)="onProjectChange($event)"
                              data-testid="dataset-project-selector"
                              class="bg-surface-mid border border-surface-high text-white text-sm rounded-theme-md px-3 py-1.5 outline-none focus:border-brand">
                              <option [value]="null">Global</option>
                              @for (p of projectService.allProjects(); track p.id) {
                                  <option [value]="p.id">{{ p.name }}</option>
                              }
                          </select>
                      </div>
                  </div>
              </div>

              <!-- Content Card -->
              <div class="bg-surface-low border border-surface-mid rounded-theme-xl shadow-2xl p-6 space-y-8">
                  
                  <!-- Actions Section -->
                  <div class="space-y-4">
                      <div class="flex items-center gap-4 border-b border-surface-mid/30 pb-2">
                          <div class="w-1 h-6 bg-brand rounded-full"></div>
                          <h3 class="text-sm font-black text-text-subtle uppercase tracking-[0.2em]">Actions</h3>
                      </div>
                      <app-dataset-toolbar
                        [initialSearch]="searchTerm()"
                        (searchChanged)="searchTerm.set($event)"
                        (createRequested)="openCreateModal()"
                        (rescanLibraryRequested)="scanAllDatasets($event)"
                      ></app-dataset-toolbar>
                  </div>

                  <!-- Datasets Section -->
                  <div class="space-y-4">
                      <div class="flex items-center gap-4 border-b border-surface-mid/30 pb-2">
                          <div class="w-1 h-6 bg-brand rounded-full"></div>
                          <h3 class="text-sm font-black text-text-subtle uppercase tracking-[0.2em]">Datasets</h3>
                      </div>
                      <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                    @for (ds of filteredDatasets(); track ds.name) {
                      <app-dataset-card
                        [dataset]="ds"
                        [activeProjectId]="projectService.activeDatasetProject()"
                        [isInProject]="isDatasetInProject(ds.id)"
                        (view)="openViewer($event)"
                        (edit)="openEditModal($event)"
                        (rescan)="rescanPromptTarget.set($event.name)"
                        (delete)="deleteDataset($event.name, true)"
                        (upload)="handleUpload($event)"
                        (cache)="cacheTargetDataset.set($event.name)"
                        (download)="downloadDataset($event)"
                        (addToProject)="addDatasetToProject($event)"
                        (removeFromProject)="removeDatasetFromProject($event)"
                      ></app-dataset-card>
                    } @empty {
                      <app-dataset-empty-state [searchTerm]="searchTerm()"></app-dataset-empty-state>
                    }
                  </div>
                  </div>
              </div>

              <!-- Create/Edit Modal -->
              @if (showCreateModal()) {
                <app-dataset-form-modal
                  [dataset]="editingDataset()"
                  [reusableClassifiers]="reusableClassifiers()"
                  (save)="saveDataset($event)"
                  (cancel)="showCreateModal.set(false)"
                ></app-dataset-form-modal>
              }
              <!-- Rescan Modal -->
              @if (showRescanModal()) {
                <app-viewer-rescan-modal
                   [forceFull]="rescanForceFull()"
                   (close)="showRescanModal.set(false)"
                   (completed)="onRescanComplete()"
                ></app-viewer-rescan-modal>
              }

              <!-- Rescan Prompt Modal (Individual) -->
              @if (rescanPromptTarget()) {
                  <app-dataset-rescan-options-modal
                     [datasetName]="rescanPromptTarget()!"
                     (close)="rescanPromptTarget.set(null)"
                     (confirm)="startSingleScan(rescanPromptTarget()!, $event)"
                  ></app-dataset-rescan-options-modal>
              }

              <!-- Single Rescan Progress Modal -->
              @if (activeSingleScan(); as scanTarget) {
                  <app-dataset-single-rescan-modal
                     [datasetName]="scanTarget.name"
                     [forceFull]="scanTarget.forceFull"
                     (completed)="onSingleRescanComplete()"
                     (close)="activeSingleScan.set(null)"
                  ></app-dataset-single-rescan-modal>
              }

              <!-- Cache Admin Modal (from Library card) -->
              @if (cacheTargetDataset()) {
                <app-viewer-cache-admin-modal
                   [datasetName]="cacheTargetDataset()!"
                   (close)="cacheTargetDataset.set(null)"
                ></app-viewer-cache-admin-modal>
              }
          </div>
      }

    </div>
  `,
  styles: []
})
export class DatasetManagerComponent implements OnInit, OnDestroy {
  private datasetService = inject(DatasetService);
  private toast = inject(ToastService);
  private ws = inject(WebSocketService);
  protected projectService = inject(ProjectService);
  private destroy$ = new Subject<void>();

  datasets = signal<Dataset[]>([]);
  searchTerm = signal('');

  // UI State
  showCreateModal = signal(false);
  showRescanModal = signal(false);
  rescanForceFull = signal(false);
  showViewer = signal(false);
  currentViewerDataset = signal('');
  editingDataset = signal<Dataset | null>(null);
  cacheTargetDataset = signal<string | null>(null);
  rescanPromptTarget = signal<string | null>(null);
  activeSingleScan = signal<{ name: string, forceFull: boolean } | null>(null);

  /** Dataset IDs currently associated with the active project */
  projectDatasetIds = signal<Set<string>>(new Set());

  filteredDatasets = computed(() => {
    const term = this.searchTerm().toLowerCase();
    return this.datasets().filter(ds =>
      ds.name.toLowerCase().includes(term) ||
      (ds.description && ds.description.toLowerCase().includes(term)) ||
      (ds.classifier && ds.classifier.toLowerCase().includes(term))
    );
  });

  reusableClassifiers = computed(() => {
    const standard = ['vehicle', 'person', 'style', 'object', 'landscape'];
    const used = this.datasets()
      .map(ds => ds.classifier)
      .filter((c): c is string => !!c && !standard.includes(c.toLowerCase()));
    return [...new Set(used)].sort();
  });

  ngOnInit() {
    this.loadDatasets();
    // Load project datasets on init if a project is already selected
    this.refreshProjectDatasets(this.projectService.activeDatasetProject());

    // When training creates a cache, update affected dataset cards in-place
    this.ws.on<{ datasets: string[] }>('dataset_cache_ready')
      .pipe(takeUntil(this.destroy$))
      .subscribe(({ datasets: names }) => {
        this.datasets.update(list =>
          list.map(ds => names.includes(ds.name) ? { ...ds, has_cache: true } : ds)
        );
      });
  }

  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
  }

  loadDatasets() {
    this.datasetService.listDatasets().subscribe(data => this.datasets.set(data));
  }

  // --- Viewer Actions ---
  openViewer(ds: Dataset) {
    if (ds.missing) {
      this.toast.warning('Cannot view a missing dataset.');
      return;
    }
    this.currentViewerDataset.set(ds.name);
    this.showViewer.set(true);
  }

  closeViewer() {
    this.showViewer.set(false);
    this.loadDatasets();
  }

  // --- Modal Actions ---
  openCreateModal() {
    this.editingDataset.set(null);
    this.showCreateModal.set(true);
  }

  openEditModal(ds: Dataset) {
    this.editingDataset.set(ds);
    this.showCreateModal.set(true);
  }

  saveDataset(data: { name: string, description: string, classifier: string }) {
    const editDs = this.editingDataset();

    if (editDs) {
      this.datasetService.updateDataset(editDs.name, data.name, data.description, data.classifier).subscribe({
        next: () => {
          this.showCreateModal.set(false);
          this.loadDatasets();
        },
        error: (err) => this.toast.error('Failed to update dataset: ' + err.error.detail)
      });
    } else {
      this.datasetService.createDataset(data.name, data.description, data.classifier).subscribe({
        next: () => {
          this.showCreateModal.set(false);
          this.loadDatasets();
        },
        error: (err) => this.toast.error('Failed to create dataset: ' + err.error.detail)
      });
    }
  }

  // --- Dataset Actions ---
  startSingleScan(name: string, forceFull: boolean) {
    this.rescanPromptTarget.set(null);
    this.activeSingleScan.set({ name, forceFull });
  }

  onSingleRescanComplete() {
    // Don't auto-close the modal, let the user click "Close", but reload
    this.loadDatasets();
  }

  scanAllDatasets(forceFull: boolean = false) {
    this.rescanForceFull.set(forceFull);
    this.showRescanModal.set(true);
  }

  onRescanComplete() {
    this.showRescanModal.set(false);
    // Reload and check for missing
    this.datasetService.listDatasets().subscribe({
      next: (datasets) => {
        this.datasets.set(datasets);
        const missing = datasets.filter(d => d.missing);
        if (missing.length > 0) {
          const names = missing.map(d => d.name).join(', ');
          if (confirm(`The following datasets are missing on disk: ${names}.\nDo you want to remove them from your library?`)) {
            missing.forEach(d => this.deleteDataset(d.name, false, true));
          }
        }
      },
      error: (err) => this.toast.error('Failed to reload library: ' + err.message)
    });
  }

  deleteDataset(name: string, deleteFiles: boolean = false, silent: boolean = false) {
    if (!silent) {
      if (deleteFiles) {
        if (!confirm(`WARNING: Are you sure you want to delete dataset '${name}'? This will PERMANENTLY DELETE the folder and files on disk.`)) return;
      } else {
        if (!confirm(`Remove dataset '${name}' from library? (Files will be kept if they exist)`)) return;
      }
    }

    this.datasetService.deleteDataset(name, deleteFiles).subscribe({
      next: () => this.loadDatasets(),
      error: (err) => {
        if (!silent) this.toast.error('Failed to delete dataset: ' + err.message);
        console.error('Failed to delete', name, err);
      }
    });
  }

  handleUpload(data: { datasetName: string, files: FileList }) {
    let completed = 0;
    const fileArray = Array.from(data.files);

    fileArray.forEach((file: File) => {
      this.datasetService.uploadFile(data.datasetName, file).subscribe({
        next: () => {
          completed++;
          if (completed === fileArray.length) {
            this.startSingleScan(data.datasetName, false);
          }
        },
        error: (err) => console.error('Upload failed', file.name, err)
      });
    });
  }

  downloadDataset(ds: Dataset) {
    const url = this.datasetService.getDownloadUrl(ds.name);
    window.open(url, '_blank');
  }

  addDatasetToProject(ds: Dataset) {
    const pid = this.projectService.activeDatasetProject();
    if (!pid) return;
    this.projectService.addProjectDataset(pid, ds.id).subscribe({
      next: () => {
        this.toast.success(`Dataset '${ds.name}' added to project.`);
        this.projectDatasetIds.update(s => new Set([...s, ds.id]));
        this.projectService.loadProjects();
      },
      error: (err: any) => this.toast.error('Failed to add dataset: ' + (err.error?.detail || err.message))
    });
  }

  removeDatasetFromProject(ds: Dataset) {
    const pid = this.projectService.activeDatasetProject();
    if (!pid) return;
    this.projectService.removeProjectDataset(pid, ds.id).subscribe({
      next: () => {
        this.toast.success(`Dataset '${ds.name}' removed from project.`);
        this.projectDatasetIds.update(s => { const n = new Set(s); n.delete(ds.id); return n; });
        this.projectService.loadProjects();
      },
      error: (err: any) => this.toast.error('Failed to remove dataset: ' + (err.error?.detail || err.message))
    });
  }

  onProjectChange(projectId: string | null) {
    // HTML select [value]="null" emits the string "null" — normalize it
    const pid = (projectId && projectId !== 'null') ? projectId : null;
    this.projectService.activeDatasetProject.set(pid);
    this.refreshProjectDatasets(pid);
  }

  isDatasetInProject(datasetId: string): boolean {
    return this.projectDatasetIds().has(datasetId);
  }

  private refreshProjectDatasets(projectId: string | null) {
    if (!projectId) {
      this.projectDatasetIds.set(new Set());
      return;
    }
    this.projectService.getProjectDatasets(projectId).subscribe({
      next: (datasets) => this.projectDatasetIds.set(new Set(datasets.map(d => d.id))),
      error: () => this.projectDatasetIds.set(new Set())
    });
  }
}
