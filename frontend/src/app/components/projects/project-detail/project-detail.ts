import { Component, OnInit, inject, input, output, computed, signal } from '@angular/core';
import { FormsModule, ReactiveFormsModule, FormBuilder, FormGroup, FormArray, FormControl } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { ProjectService, Project } from '../../../services/project.service';
import { DatasetService, Dataset } from '../../../services/dataset';
import { TemplateService } from '../../../services/template.service';
import { JobService } from '../../../services/job';
import { ToastService } from '../../../services/toast';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { GeneralTemplatesComponent } from '../general-templates/general-templates';
import { DynamicFormGroupComponent } from '../../training/dynamic-form-group/dynamic-form-group';
import { ProjectDialogComponent } from '../project-dialog/project-dialog';

@Component({
  selector: 'app-project-detail',
  standalone: true,
  imports: [FormsModule, ReactiveFormsModule, GeneralTemplatesComponent, DynamicFormGroupComponent, ProjectDialogComponent],
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
                        <button (click)="showEditDialog.set(true)" 
                                class="p-1.5 text-text-disabled hover:text-brand hover:bg-brand/10 rounded-theme-md transition-all" 
                                title="Edit project details">
                            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/></svg>
                        </button>
                      </div>
                      <p class="text-text-muted">{{ project()?.description || 'No description' }}</p>
                  </div>
              </div>
          </div>
      </div>

      <div class="space-y-6">
          
              
              <!-- Stats Row -->
              <div class="grid grid-cols-2 sm:grid-cols-5 gap-4">
                  <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-4">
                      <div class="text-sm text-text-muted mb-1">Datasets</div>
                      <div class="text-2xl font-bold text-white">{{ project()?.stats?.datasets || 0 }}</div>
                  </div>
                  <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-4">
                      <div class="text-sm text-text-muted mb-1">Training Jobs</div>
                      <div class="text-2xl font-bold text-white">{{ project()?.stats?.jobs || 0 }}</div>
                  </div>
                  <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-4">
                      <div class="text-sm text-text-muted mb-1">Training</div>
                      <div class="text-2xl font-bold text-brand-light">{{ project()?.stats?.training_templates || 0 }}</div>
                  </div>
                  <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-4">
                      <div class="text-sm text-text-muted mb-1">Captioning</div>
                      <div class="text-2xl font-bold text-brand-light">{{ project()?.stats?.captioning_templates || 0 }}</div>
                  </div>
                  <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-4">
                      <div class="text-sm text-text-muted mb-1">Masking</div>
                      <div class="text-2xl font-bold text-brand-light">{{ project()?.stats?.masking_templates || 0 }}</div>
                  </div>
              </div>

              <!-- ═══════════════════════════════════════════════════════ -->
              <!-- Training Quick Launch Panel                            -->
              <!-- ═══════════════════════════════════════════════════════ -->
              @if (projectTrainingTemplates().length > 0) {
              <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-6 shadow-lg">
                  <div class="flex items-center gap-3 mb-6">
                    <div class="p-2 bg-brand/10 rounded-theme-md border border-brand/20">
                      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-brand"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                    </div>
                    <div>
                      <h3 class="text-lg font-bold text-white">Training Quick Launch</h3>
                      <p class="text-text-muted text-sm">Select a template, customize per-run settings, and start training.</p>
                    </div>
                  </div>

                  <!-- Template Selector -->
                  <div class="space-y-4">
                    <div>
                      <label class="block text-xs font-bold uppercase tracking-wider text-text-subtle mb-1.5">Training Template</label>
                      <select [ngModel]="selectedTemplateId()" (ngModelChange)="onSelectTemplate($event)"
                              class="w-full bg-surface-mid border border-surface-high text-white text-sm rounded-theme-md px-3 py-2 outline-none focus:border-brand transition-colors">
                        <option value="">Choose a template...</option>
                        @for (tpl of projectTrainingTemplates(); track tpl.id) {
                          <option [value]="tpl.id">{{ tpl.name }} — {{ tpl.definition_id }}</option>
                        }
                      </select>
                    </div>

                    @if (selectedTemplateId()) {
                      <!-- LoRA Name & Trigger Word -->
                      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div>
                          <label class="block text-xs font-bold uppercase tracking-wider text-text-subtle mb-1.5">LoRA Name</label>
                          <input type="text" [(ngModel)]="launchLoraName"
                                 placeholder="my_lora_v1"
                                 class="w-full bg-surface-mid border border-surface-high text-white text-sm rounded-theme-md px-3 py-2 outline-none focus:border-brand transition-colors placeholder:text-text-disabled">
                        </div>
                        <div>
                          <label class="block text-xs font-bold uppercase tracking-wider text-text-subtle mb-1.5">Trigger Word</label>
                          <input type="text" [(ngModel)]="launchTriggerWord"
                                 placeholder="ohwx"
                                 class="w-full bg-surface-mid border border-surface-high text-white text-sm rounded-theme-md px-3 py-2 outline-none focus:border-brand transition-colors placeholder:text-text-disabled">
                        </div>
                      </div>

                      <!-- Dataset Configuration (reused from Training Tab) -->
                      @if (datasetsSchema()) {
                        <app-dynamic-form-group
                            [fieldKey]="'datasets'"
                            [schema]="datasetsSchema()"
                            [rootSchema]="trainingSchema()"
                            [parentForm]="launchForm"
                            [datasetNames]="projectDatasetNames()"
                            (arrayItemAdded)="onArrayItemAdded($event.key, $event.schemaParam)"
                            (arrayItemRemoved)="onArrayItemRemoved($event.key, $event.index)"
                            (helpRequested)="$event">
                        </app-dynamic-form-group>
                      }

                      <!-- Start Training Button -->
                      <button (click)="startTraining()" [disabled]="!canStartTraining()"
                              class="w-full mt-2 bg-gradient-to-r from-brand to-brand-gradient-end hover:from-brand/90 hover:to-brand-gradient-end/80 text-white font-bold py-3 px-6 rounded-theme-lg transition-all shadow-lg hover:shadow-brand/20 disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none transform hover:-translate-y-0.5 active:translate-y-0">
                        🚀 Start Training
                      </button>
                    }
                  </div>
              </div>
              }

              <!-- Datasets Association Area -->
              <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-6 shadow-lg">
                  <div class="flex items-center justify-between mb-4">
                    <div>
                      <h3 class="text-lg font-bold text-white">Datasets</h3>
                      <p class="text-text-muted text-sm mt-1">Associate datasets with this project for quick access.</p>
                    </div>
                    @if (!showDatasetPicker()) {
                      <div class="flex items-center gap-2">
                        @if (projectDatasets().length > 0) {
                          <button (click)="removeAllDatasets()"
                                  class="flex items-center gap-1.5 text-text-muted hover:text-danger border border-surface-high hover:border-danger/40 px-3 py-1.5 rounded-theme-md transition-all text-sm">
                            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                            Remove All
                          </button>
                        }
                        <button (click)="showDatasetPicker.set(true)"
                                class="flex items-center gap-1.5 bg-brand/20 hover:bg-brand/30 border border-brand/40 text-brand-light px-3 py-1.5 rounded-theme-md transition-all text-sm">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                          Add Dataset
                        </button>
                      </div>
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
                        <div class="flex items-center gap-1">
                          @if (selectedTemplateId()) {
                            <button (click)="addDatasetToLaunch(ds)"
                                    class="text-xs text-brand-light hover:text-white bg-brand/10 hover:bg-brand/20 px-2 py-1 rounded-theme-md transition-colors border border-brand/20"
                                    title="Add to Quick Launch datasets">
                              ＋ Launch
                            </button>
                          }
                          <button (click)="removeDataset(ds.id)"
                                  class="text-text-muted hover:text-danger transition-colors p-1" title="Remove from project">
                            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                          </button>
                        </div>
                      </div>
                    } @empty {
                      <div class="text-center text-text-subtle p-4 bg-surface-high rounded-theme-md text-sm">
                        No datasets associated yet.
                      </div>
                    }
                  </div>
              </div>

              <!-- Project Templates Section -->
              <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-6 shadow-lg">
                  <div class="flex items-center justify-between mb-4">
                    <div>
                      <h3 class="text-lg font-bold text-white">Project Templates</h3>
                      <p class="text-text-muted text-sm mt-1">Templates branched into this project.</p>
                    </div>
                    <button (click)="loadProjectTemplatesForTab(projectTemplateTab())" class="text-text-muted hover:text-white transition-colors p-1.5" title="Refresh">
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>
                    </button>
                  </div>

                  <!-- Tabs -->
                  <div class="flex gap-4 mb-4">
                    @for (tab of templateTabs; track tab.key) {
                      <button [class.text-white]="projectTemplateTab() === tab.key"
                              [class.border-brand]="projectTemplateTab() === tab.key"
                              [class.text-text-muted]="projectTemplateTab() !== tab.key"
                              [class.border-transparent]="projectTemplateTab() !== tab.key"
                              class="pb-2 border-b-2 font-medium transition-colors text-sm"
                              (click)="loadProjectTemplatesForTab(tab.key)">{{ tab.label }}</button>
                    }
                  </div>

                  <div class="space-y-2 max-h-[300px] overflow-y-auto pr-1 custom-scrollbar">
                    @for (tpl of projectTemplatesFiltered(); track tpl.id) {
                      <div class="flex items-center justify-between p-3 bg-surface-mid border border-surface-high rounded-theme-md hover:border-brand/30 transition-colors"
                           [class.border-brand]="projectTemplateTab() === 'training' && tpl.id === selectedTemplateId()">
                        <div class="min-w-0 flex-1">
                          <div class="flex items-center gap-2">
                            <span class="font-medium text-white truncate">{{ tpl.name }}</span>
                            <span class="text-xs bg-emerald-500/15 text-emerald-400 px-2 py-0.5 rounded-full shrink-0">📁 Project</span>
                          </div>
                          <div class="text-xs text-text-subtle mt-1 flex items-center gap-2">
                            <span class="truncate max-w-[180px]">{{ tpl.definition_id || tpl.model_id || '' }}</span>
                            @if (tpl.branched_from) {
                              <span class="text-amber-400/80 shrink-0">↳ branched</span>
                            }
                          </div>
                        </div>
                        <div class="flex items-center gap-1.5 shrink-0 ml-2">
                          @if (projectTemplateTab() === 'training') {
                            <button (click)="onSelectTemplate(tpl.id)" 
                                    class="text-xs bg-brand/20 hover:bg-brand/30 text-brand-light px-2.5 py-1.5 rounded-theme-md transition-colors border border-brand/30"
                                    title="Use this template for Quick Launch">
                              Select
                            </button>
                          }
                          <button (click)="deleteProjectTemplate(tpl)"
                                  class="text-xs bg-surface-high hover:bg-danger/20 text-danger p-1.5 rounded-theme-md transition-colors border border-border-default"
                                  title="Delete this project template">
                            <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                          </button>
                        </div>
                      </div>
                    } @empty {
                      <div class="text-center text-text-subtle p-4 bg-surface-high rounded-theme-md text-sm">
                        No project {{ projectTemplateTab() }} templates yet. Branch a global template below.
                      </div>
                    }
                  </div>
              </div>

              <!-- Global Templates Branching Area -->
              <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-6 shadow-lg">
                  <h3 class="text-lg font-bold text-white mb-2">Branch Global Templates</h3>
                  <p class="text-text-muted mb-6 text-sm">
                      Copy "General" (global) templates into this project to customize them without affecting other projects.
                  </p>
                  
                  <app-general-templates [projectId]="projectId()"></app-general-templates>
              </div>

      </div>

      <!-- Edit Project Dialog -->
      @if (showEditDialog()) {
        <app-project-dialog
          [projectId]="projectId()"
          (close)="showEditDialog.set(false)"
          (saved)="onEditSaved()" />
      }
    </div>
  `
})
export class ProjectDetailComponent implements OnInit {
  private projectService = inject(ProjectService);
  private datasetService = inject(DatasetService);
  private templateService = inject(TemplateService);
  private jobService = inject(JobService);
  private toast = inject(ToastService);
  private http = inject(HttpClient);
  private fb = inject(FormBuilder);
  private rtc = inject(RuntimeConfigService);
  
  projectId = input.required<string>();
  back = output<void>();

  // Dataset association state
  allDatasets = signal<Dataset[]>([]);
  projectDatasets = signal<any[]>([]);
  showDatasetPicker = signal(false);
  showEditDialog = signal(false);
  selectedDatasetToAdd = '';

  // Project-scoped templates (all types, filtered by active tab)
  projectTemplatesFiltered = signal<any[]>([]);
  projectTemplateTab = signal<'training' | 'captioning' | 'masking'>('training');
  templateTabs = [
    { key: 'training' as const, label: 'Training' },
    { key: 'captioning' as const, label: 'Captioning' },
    { key: 'masking' as const, label: 'Masking' }
  ];

  // Keep a separate signal for training templates (used by Quick Launch)
  projectTrainingTemplates = signal<any[]>([]);

  // Quick Launch state
  selectedTemplateId = signal<string>('');
  launchLoraName = '';
  launchTriggerWord = '';

  // Schema-driven dataset form (reused from Training Tab)
  trainingSchema = signal<any>(null);
  datasetsSchema = signal<any>(null);
  launchForm: FormGroup = new FormGroup({ datasets: new FormArray<any>([]) });

  // A local signal to hold the fetched project details
  project = computed(() => {
    return this.projectService.allProjects().find(p => p.id === this.projectId()) || null;
  });

  // Compute available datasets (not yet associated with this project)
  availableDatasets = computed(() => {
    const associated = new Set(this.projectDatasets().map(d => d.id));
    return this.allDatasets().filter(d => !associated.has(d.id));
  });

  // Project-scoped dataset names for the DynamicFormGroupComponent autocomplete
  projectDatasetNames = computed(() => this.projectDatasets().map(d => d.name));

  canStartTraining = computed(() => {
    const fa = this.launchForm.get('datasets') as FormArray;
    return this.selectedTemplateId() && this.launchLoraName && fa && fa.length > 0;
  });

  ngOnInit() {
    this.loadDatasets();
    this.loadProjectDatasets();
    this.loadProjectTemplates();
  }

  onEditSaved() {
    this.showEditDialog.set(false);
    this.projectService.loadProjects();
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

  loadProjectTemplates() {
    // Load training templates (always needed for Quick Launch)
    this.templateService.listTrainingTemplates(undefined, this.projectId()).subscribe((res: any) => {
      const filtered = res.filter((t: any) => t.project_id === this.projectId());
      this.projectTrainingTemplates.set(filtered);
      // If currently viewing training tab, also update the filtered list
      if (this.projectTemplateTab() === 'training') {
        this.projectTemplatesFiltered.set(filtered);
      }
    });
    // Also load the current tab if not training
    if (this.projectTemplateTab() !== 'training') {
      this.loadProjectTemplatesForTab(this.projectTemplateTab());
    }
  }

  loadProjectTemplatesForTab(tab: 'training' | 'captioning' | 'masking') {
    this.projectTemplateTab.set(tab);
    const pid = this.projectId();
    const req = tab === 'training'
      ? this.templateService.listTrainingTemplates(undefined, pid)
      : tab === 'captioning'
        ? this.templateService.listCaptioningTemplates(undefined, pid)
        : this.templateService.listMaskingTemplates(undefined, pid);

    req.subscribe((res: any) => {
      const filtered = res.filter((t: any) => t.project_id === pid);
      this.projectTemplatesFiltered.set(filtered);
      // Keep training templates in sync for Quick Launch
      if (tab === 'training') {
        this.projectTrainingTemplates.set(filtered);
      }
    });
  }

  // ── Quick Launch ───────────────────────────────────────────────

  onSelectTemplate(templateId: string) {
    this.selectedTemplateId.set(templateId);
    if (!templateId) return;

    // Load template config and populate fields
    this.templateService.getTemplate('training', templateId).subscribe({
      next: (tpl) => {
        const cfg = tpl.config || {};
        this.launchLoraName = cfg.lora_name || '';
        this.launchTriggerWord = cfg.global_triggerword || '';

        // Fetch the training schema so we can render the exact same dataset form
        this.http.get(`${this.rtc.apiUrl}/plugins/standard/schema?t=${Date.now()}`).subscribe({
          next: (schema: any) => {
            this.trainingSchema.set(schema);

            // Extract the datasets array schema from the full schema
            const props = schema?.properties || {};
            if (props.datasets) {
              // Deep clone to avoid mutating the shared schema object
              const dsSchema = JSON.parse(JSON.stringify(props.datasets));

              // Override dataset_name enum with project-scoped datasets only
              const projectNames = this.projectDatasetNames();
              const items = dsSchema.items;
              if (items?.properties?.dataset_name) {
                items.properties.dataset_name.enum = projectNames;
              }
              // Also patch $defs if the schema uses $ref
              const defs = schema.$defs || schema.definitions || {};
              for (const defVal of Object.values(defs) as any[]) {
                if (defVal?.properties?.dataset_name) {
                  defVal.properties.dataset_name = {
                    ...defVal.properties.dataset_name,
                    enum: projectNames
                  };
                }
              }

              this.datasetsSchema.set(dsSchema);
            }

            // Build the launch form with the datasets FormArray
            this.buildLaunchForm(schema, cfg);
          },
          error: () => this.toast.error('Failed to load training schema.')
        });
      },
      error: () => this.toast.error('Failed to load template details.')
    });
  }

  private resolveSchema(schemaOrRef: any): any {
    if (!schemaOrRef) return {};
    const root = this.trainingSchema() || {};
    const definitions = root.$defs || root.definitions || {};
    if (schemaOrRef.$ref) {
      const refKey = schemaOrRef.$ref.split('/').pop();
      if (definitions[refKey]) return { ...definitions[refKey], ...schemaOrRef };
    }
    return schemaOrRef;
  }

  private buildLaunchForm(schema: any, cfg: any) {
    const datasetsArray = this.fb.array([]);
    this.launchForm = this.fb.group({ datasets: datasetsArray });

    const templateDatasets: any[] = cfg.datasets || [];
    const dsSchemaRef = schema?.properties?.datasets?.items;

    if (templateDatasets.length > 0) {
      // Populate from template config
      for (const ds of templateDatasets) {
        this.addDatasetArrayItem(dsSchemaRef, ds);
      }
    } else {
      // Add one empty entry
      this.addDatasetArrayItem(dsSchemaRef);
    }
  }

  private addDatasetArrayItem(itemSchemaRef: any, values?: any) {
    const itemSchema = this.resolveSchema(itemSchemaRef);
    const fa = this.launchForm.get('datasets') as FormArray;
    if (!itemSchema?.properties) return;

    const group: Record<string, FormControl> = {};
    for (const pKey in itemSchema.properties) {
      const pSchema = this.resolveSchema(itemSchema.properties[pKey]);
      let val = values?.[pKey];
      if (val === undefined) {
        val = pSchema.default !== undefined ? pSchema.default : '';
        if (pSchema.enum?.length && !val) val = pSchema.enum[0];
      }
      group[pKey] = new FormControl(val);
    }
    fa.push(this.fb.group(group));
  }

  // Called by DynamicFormGroupComponent when user clicks "Add Dataset"
  onArrayItemAdded(key: string, itemSchemaRef: any) {
    if (key === 'datasets') {
      this.addDatasetArrayItem(itemSchemaRef);
    }
  }

  // Called by DynamicFormGroupComponent when user clicks the remove button
  onArrayItemRemoved(key: string, index: number) {
    if (key === 'datasets') {
      const fa = this.launchForm.get('datasets') as FormArray;
      fa.removeAt(index);
    }
  }

  addDatasetToLaunch(ds: any) {
    const fa = this.launchForm.get('datasets') as FormArray;
    // Check if already present
    const existing = fa.controls.some((c: any) => c.get('dataset_name')?.value === ds.name);
    if (existing) {
      this.toast.warning(`'${ds.name}' is already in the Quick Launch list.`);
      return;
    }

    // Add with dataset_name pre-filled and prefix derived
    const dsSchemaRef = this.trainingSchema()?.properties?.datasets?.items;
    const prefix = ds.name.toLowerCase().replace(/[_-]/g, ' ').trim();
    this.addDatasetArrayItem(dsSchemaRef, { dataset_name: ds.name, caption_prefix: prefix });
    this.toast.success(`Added '${ds.name}' to Quick Launch.`);
  }

  startTraining() {
    const templateId = this.selectedTemplateId();
    if (!templateId) return;

    // Get the full template to clone its config
    this.templateService.getTemplate('training', templateId).subscribe({
      next: (tpl) => {
        // Clone the config — template stays pristine
        const config = { ...(tpl.config || {}) };

        // Apply overrides
        config.lora_name = this.launchLoraName;
        config.global_triggerword = this.launchTriggerWord;
        config.project_id = this.projectId();

        // Extract dataset values from the reactive FormArray
        const fa = this.launchForm.get('datasets') as FormArray;
        config.datasets = fa.value.filter((ds: any) => ds.dataset_name);

        // Use the template's definition_id as the plugin_id
        const pluginId = tpl.definition_id || config.definition_id || '';

        this.jobService.createJob(pluginId, config).subscribe({
          next: () => {
            this.toast.success('Training job queued successfully! Check the Jobs tab.');
            this.projectService.loadProjects();
          },
          error: (err: any) => {
            this.toast.error('Failed to create training job: ' + (err.error?.detail || err.message));
          }
        });
      },
      error: () => this.toast.error('Failed to load template for training.')
    });
  }

  // ── Project Templates Management ──────────────────────────────

  deleteProjectTemplate(template: any) {
    if (!confirm(`Delete project template '${template.name}'? This cannot be undone.`)) return;
    const domain = this.projectTemplateTab();
    this.templateService.deleteTemplate(domain, template.id).subscribe({
      next: () => {
        this.projectTemplatesFiltered.update(current => current.filter(t => t.id !== template.id));
        // Also update training templates if we're on the training tab
        if (domain === 'training') {
          this.projectTrainingTemplates.update(current => current.filter(t => t.id !== template.id));
          if (this.selectedTemplateId() === template.id) {
            this.selectedTemplateId.set('');
          }
        }
        this.toast.success(`Deleted template '${template.name}'.`);
        this.projectService.loadProjects(); // refresh stats
      },
      error: (err: any) => this.toast.error('Failed to delete template: ' + (err.error?.detail || err.message))
    });
  }

  // ── Datasets Association ──────────────────────────────────────

  addDataset() {
    if (!this.selectedDatasetToAdd) return;
    this.projectService.addProjectDataset(this.projectId(), this.selectedDatasetToAdd).subscribe({
      next: () => {
        this.toast.success('Dataset added to project.');
        this.loadProjectDatasets();
        this.selectedDatasetToAdd = '';
        this.showDatasetPicker.set(false);
        this.projectService.loadProjects();
      },
      error: (err: any) => this.toast.error('Failed to add dataset: ' + (err.error?.detail || err.message))
    });
  }

  removeDataset(datasetId: string) {
    // Resolve the dataset name before removing, so we can purge from launch form
    const dsName = this.projectDatasets().find(d => d.id === datasetId)?.name;
    this.projectService.removeProjectDataset(this.projectId(), datasetId).subscribe({
      next: () => {
        this.toast.success('Dataset removed from project.');
        this.loadProjectDatasets();
        this.projectService.loadProjects();
        // Purge from Quick Launch config if present
        if (dsName) this.purgeFromLaunchForm(dsName);
      },
      error: (err: any) => this.toast.error('Failed to remove dataset: ' + (err.error?.detail || err.message))
    });
  }

  removeAllDatasets() {
    const datasets = this.projectDatasets();
    if (datasets.length === 0) return;
    
    let completed = 0;
    let failed = 0;
    for (const ds of datasets) {
      this.projectService.removeProjectDataset(this.projectId(), ds.id).subscribe({
        next: () => {
          completed++;
          if (completed + failed === datasets.length) {
            this.toast.success(`Removed ${completed} dataset(s) from project.`);
            this.loadProjectDatasets();
            this.projectService.loadProjects();
            // Clear all entries from Quick Launch config
            this.clearLaunchFormDatasets();
          }
        },
        error: () => {
          failed++;
          if (completed + failed === datasets.length) {
            this.toast.error(`Removed ${completed}, failed ${failed}.`);
            this.loadProjectDatasets();
            this.projectService.loadProjects();
          }
        }
      });
    }
  }

  /** Remove any Quick Launch form entries whose dataset_name matches */
  private purgeFromLaunchForm(datasetName: string) {
    const fa = this.launchForm.get('datasets') as FormArray;
    if (!fa) return;
    for (let i = fa.length - 1; i >= 0; i--) {
      if (fa.at(i).get('dataset_name')?.value === datasetName) {
        fa.removeAt(i);
      }
    }
    this.refreshDatasetSchemaEnum();
  }

  /** Clear all entries from the Quick Launch datasets form */
  private clearLaunchFormDatasets() {
    const fa = this.launchForm.get('datasets') as FormArray;
    if (!fa) return;
    while (fa.length > 0) fa.removeAt(0);
    this.refreshDatasetSchemaEnum();
  }

  /** Re-patch the datasetsSchema enum with current project dataset names */
  private refreshDatasetSchemaEnum() {
    const current = this.datasetsSchema();
    if (!current) return;
    const updated = JSON.parse(JSON.stringify(current));
    const names = this.projectDatasetNames();
    if (updated.items?.properties?.dataset_name) {
      updated.items.properties.dataset_name.enum = names;
    }
    // Also patch $defs if present in the root training schema
    const root = this.trainingSchema();
    if (root) {
      const defs = root.$defs || root.definitions || {};
      for (const defVal of Object.values(defs) as any[]) {
        if (defVal?.properties?.dataset_name) {
          defVal.properties.dataset_name.enum = names;
        }
      }
    }
    this.datasetsSchema.set(updated);
  }
}
