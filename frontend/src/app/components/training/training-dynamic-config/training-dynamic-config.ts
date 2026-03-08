import { Component, OnChanges, SimpleChanges, output, input, inject, signal, computed, effect, DestroyRef, ViewChild } from '@angular/core';
import { TitleCasePipe } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, FormControl, FormArray, Validators, FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { DatasetService } from '../../../services/dataset';
import { ToastService } from '../../../services/toast';
import { SystemService, VRAMReport } from '../../../services/system.service';

import { Subject } from 'rxjs';
import { debounceTime } from 'rxjs/operators';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { TrainingTemplateSelectorComponent } from '../training-template-selector/training-template-selector';
import { VramBudgetCardComponent } from '../vram-budget-card/vram-budget-card';
import { DynamicFormFieldComponent } from '../dynamic-form-field/dynamic-form-field';
import { DynamicFormGroupComponent } from '../dynamic-form-group/dynamic-form-group';
import { AdvancedVramCardComponent } from '../advanced-vram-card/advanced-vram-card';
import { TargetLayersCardComponent } from '../target-layers-card/target-layers-card';

export interface TrainingTemplate {
  id: string;
  name: string;
  definition_id: string; // Scoped to model definition
  is_default?: boolean;
  config: any;
}

@Component({
  selector: 'app-training-dynamic-config',
  standalone: true,
  imports: [TitleCasePipe, ReactiveFormsModule, FormsModule, TrainingTemplateSelectorComponent, VramBudgetCardComponent, AdvancedVramCardComponent, DynamicFormFieldComponent, DynamicFormGroupComponent, TargetLayersCardComponent],
  template: `
    @if (schema()) {
      <form [formGroup]="form" (ngSubmit)="onSubmit()" class="flex flex-col gap-6 p-6 bg-surface-low border border-surface-mid rounded-theme-xl shadow-2xl isolate">
        

        <!-- Template Selection Child Component -->
        <app-training-template-selector
          [availableModels]="availableModels()"
          [selectedDefinitionId]="selectedDefinition()?.id || null"
          [currentFormConfig]="form.value"
          (templateApplied)="onTemplateApplied($event)">
        </app-training-template-selector>



        <!-- Model Selection Section (hardcoded) -->
        <div class="space-y-6 mb-10">
          <div class="flex items-center justify-between border-b border-surface-mid/30 pb-2 mb-4">
              <div class="flex items-center gap-4">
                <div class="w-1 h-6 bg-brand rounded-full"></div>
                <h3 class="text-sm font-black text-text-subtle uppercase tracking-[0.2em]">Model Selection</h3>
              </div>
              
              @if (selectedDefinition(); as model) {
                <div class="text-[10px] font-mono text-text-disabled bg-surface-mid/20 px-3 py-1 rounded-theme-md">
                   ID: <span class="text-brand-light">{{ model.id }}</span>
                </div>
              }
          </div>
          
          <div class="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
            <!-- Dynamic quantization and model fields from MODEL_SELECTION group -->
            @for (prop of modelSelectionProps(); track prop.key) {
              @if (!shouldHideField(prop.schema)) {
                <div [class.md:col-span-2]="isLongInput(prop.key, prop.schema)"
                     [class.opacity-40]="isFieldDisabled(prop.schema)"
                     [class.pointer-events-none]="isFieldDisabled(prop.schema)"
                     class="flex flex-col gap-2 transition-opacity duration-200">
                  <label [for]="prop.key" class="text-sm font-medium text-text-secondary flex items-center gap-1.5">
                    {{ prop.schema.title || (prop.key | titlecase) }}
                    @if (hasHelp(prop.key)) {
                      <span class="config-help-icon" [title]="getHelpTip(prop.key)" (click)="openHelpModal(prop.key); $event.preventDefault()">?</span>
                    }
                  </label>
                  
                  @if (isBoolean(prop.schema)) {
                    <label class="relative inline-flex items-center cursor-pointer group">
                      <input type="checkbox" [formControlName]="prop.key"
                             [attr.data-testid]="'config-checkbox-' + prop.key"
                             class="sr-only peer">
                      <div class="w-11 h-6 bg-surface-high peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-brand/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-border-subtle after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-brand group-hover:bg-surface-mid transition-all"></div>
                      <span class="ml-3 text-sm font-medium text-text-muted group-hover:text-text-secondary">Enable</span>
                    </label>
                  } @else if (isString(prop.schema) && prop.schema.enum) {
                    <select [formControlName]="prop.key"
                            [attr.data-testid]="'config-select-' + prop.key"
                            class="bg-surface-high border border-surface-high/50 rounded-theme-lg px-4 py-2 text-white w-full appearance-none focus:ring-2 focus:ring-brand outline-none">
                      @for (opt of getFilteredEnumOptions(prop); track opt.value) {
                        <option [value]="opt.value" [disabled]="opt.disabled">{{ opt.label }}</option>
                      }
                    </select>
                  } @else if (isString(prop.schema)) {
                    <input type="text" [formControlName]="prop.key"
                           [attr.data-testid]="'config-input-' + prop.key"
                           class="bg-surface-mid border border-surface-high rounded-theme-lg px-4 py-2 text-white w-full focus:ring-2 focus:ring-brand outline-none transition-all">
                  }
                  
                  @if (prop.schema.description) {
                    <p class="text-xs text-text-subtle italic">{{ prop.schema.description }}</p>
                  }
                </div>
              }
            }
          </div>
        </div>

        <!-- VRAM Budget Card -->
        <app-vram-budget-card [report]="vramReport()"></app-vram-budget-card>

        <!-- Advanced VRAM Management (Block Swapping) -->
        <app-advanced-vram-card
          [definitionId]="currentDefinitionId()"
          (blockSwapChanged)="onBlockSwapChanged($event)">
        </app-advanced-vram-card>

        <div class="space-y-10">
          @for (group of groups(); track group.name) {
            @if (!isGroupHidden(group)) {
            <div class="space-y-6">
              <!-- Group Header -->
              <div class="flex items-center justify-between border-b border-surface-mid/30 pb-2 mb-4 cursor-pointer select-none"
                    (click)="toggleGroup(group.name)">
                   <div class="flex items-center gap-4">
                     <div class="w-1 h-6 bg-brand rounded-full"></div>
                     <h3 class="text-sm font-black text-text-subtle uppercase tracking-[0.2em]">{{ formatGroupName(group.name) }}</h3>
                   </div>
                   <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                        class="text-text-disabled transition-transform duration-200" [class.rotate-180]="!isGroupCollapsed(group.name)">
                     <path d="m6 9 6 6 6-6"/>
                   </svg>
               </div>

              @if (!isGroupCollapsed(group.name)) {
              <div class="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6 animate-in fade-in slide-in-from-top-2 duration-200">

                  @for (prop of group.props; track prop.key) {

                     <!-- Inline group: render all grouped toggles in one full-width row on first encounter -->
                     @if (prop.schema.inline_group && isFirstInlineGroupProp(prop.key, group.props)) {
                       <div class="md:col-span-2 flex flex-col gap-2">
                         <label class="text-sm font-bold text-text-subtle uppercase tracking-widest flex items-center gap-1.5 mb-1">
                           {{ prop.schema.inline_group === 'masking_toggles' ? 'Enable masking' : (prop.schema.inline_group.replace('_', ' ') | titlecase) }}
                         </label>
                         <div class="grid grid-cols-3 gap-x-8">
                           @for (ip of getInlineGroupProps(prop.schema.inline_group, group.props); track ip.key) {
                             <div class="flex flex-col gap-2">
                               <label class="text-sm font-medium text-text-secondary flex items-center gap-1.5 mb-1">
                                 {{ ip.schema.title || (ip.key | titlecase) }}
                                 @if (hasHelp(ip.key)) {
                                   <span class="config-help-icon" [title]="getHelpTip(ip.key)" (click)="openHelpModal(ip.key); $event.preventDefault()">?</span>
                                 }
                               </label>
                               <label class="relative inline-flex items-center cursor-pointer group">
                                 <input type="checkbox" [formControlName]="ip.key"
                                        [attr.data-testid]="'config-checkbox-' + ip.key"
                                        class="sr-only peer">
                                 <div class="w-11 h-6 bg-surface-high peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-brand/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-border-subtle after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-brand group-hover:bg-surface-mid transition-all"></div>
                                 <span class="ml-3 text-sm font-medium text-text-muted group-hover:text-text-secondary">Enable</span>
                               </label>
                               @if (ip.schema.description) {
                                 <p class="text-xs text-text-subtle italic">{{ ip.schema.description }}</p>
                               }
                             </div>
                           }
                         </div>
                       </div>
                     }

                     <!-- Normal Grouping for non-array types (skip block_swap_sliders — rendered near VRAM card) -->
                     @if (prop.schema.type !== 'array' && !shouldHideField(prop.schema) && !prop.schema.inline_group && prop.schema.ui_type !== 'block_swap_sliders') {
                         <app-dynamic-form-field
                            [control]="getControl(prop.key)"
                            [schema]="prop.schema"
                            [fieldKey]="prop.key"
                            [currentBackend]="form.get('backend')?.value || 'local'"
                            [outputDir]="form.get('output_dir')?.value || 'outputs'"
                            [hasHelp]="hasHelp(prop.key)"
                            [helpTip]="getHelpTip(prop.key)"
                            (helpRequested)="openHelpModal($event)"
                            (checkpointConfigLoaded)="loadExternalConfig($event)">
                         </app-dynamic-form-field>
                     }

                 <!-- Array Types (Delegated to Group Component or Custom Checklists) -->
                 @if (prop.schema.type === 'array' && (prop.schema.ui_type === 'layer_checklist' || !shouldHideField(prop.schema))) {
                    @if (prop.schema.ui_type === 'layer_checklist') {
                        <div class="md:col-span-2 mt-4 mb-2">
                            <app-target-layers-card 
                                [definitionId]="currentDefinitionId()" 
                                [control]="getArrayStringControl(prop.key)">
                            </app-target-layers-card>
                        </div>
                    } @else {
                        <app-dynamic-form-group
                            [fieldKey]="prop.key"
                            [schema]="prop.schema"
                            [rootSchema]="schema()"
                            [parentForm]="form"
                            [currentBackend]="form.get('backend')?.value || 'local'"
                            [outputDir]="form.get('output_dir')?.value || 'outputs'"
                            [configHelp]="configHelp()"
                            (arrayItemAdded)="addArrayItem($event.key, $event.schemaParam)"
                            (arrayItemRemoved)="removeArrayItem($event.key, $event.index)"
                            (helpRequested)="openHelpModal($event)"
                            (checkpointConfigLoaded)="loadExternalConfig($event)">
                        </app-dynamic-form-group>
                    }
                  }
                }
              </div>
              }


            </div>
            }
          }
        </div>

        <button type="submit" 
          [disabled]="!form.valid" 
          data-testid="submit-config-btn"
          class="mt-6 bg-gradient-to-r from-brand to-brand-gradient-end hover:from-brand/90 hover:to-brand-gradient-end/80 text-white font-bold py-3 px-6 rounded-theme-lg transition-all shadow-lg hover:shadow-brand/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none transform hover:-translate-y-0.5 active:translate-y-0">
          Start Training Session
        </button>

      </form>

      @if (activeHelpKey()) {
        <div class="fixed inset-0 bg-overlay backdrop-blur-md z-[100] flex items-center justify-center p-4 animate-in fade-in duration-300"
             (click)="closeHelpModal()">
          <div class="bg-surface-low border border-surface-mid rounded-theme-xl w-full max-w-lg shadow-2xl p-8 transform animate-in slide-in-from-bottom-4 duration-300"
               (click)="$event.stopPropagation()">
            
            <div class="flex items-center gap-4 mb-6">
              <div class="p-3 bg-brand/10 rounded-theme-md border border-brand/20 text-brand">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <circle cx="12" cy="12" r="10"></circle>
                  <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
                  <line x1="12" y1="17" x2="12.01" y2="17"></line>
                </svg>
              </div>
              <div>
                <h3 class="text-xl font-bold text-white">{{ getHelpTitle(activeHelpKey()!) }}</h3>
                <p class="text-xs text-text-subtle mt-0.5">{{ getHelpTip(activeHelpKey()!) }}</p>
              </div>
            </div>

            <div class="help-detail-content text-sm text-text-secondary leading-relaxed max-h-[60vh] overflow-y-auto pr-2"
                 [innerHTML]="renderHelpDetail(activeHelpKey()!)">
            </div>

            <div class="flex justify-end mt-8">
              <button (click)="closeHelpModal()"
                      class="text-text-subtle hover:text-white text-sm font-bold px-6 py-3 transition-colors uppercase tracking-widest">
                Got it
              </button>
            </div>
          </div>
        </div>
      }

      @if (showModelChangeModal()) {
        <div class="fixed inset-0 bg-overlay backdrop-blur-md z-[100] flex items-center justify-center p-4 animate-in fade-in duration-300">
          <div class="bg-surface-low border border-surface-mid rounded-theme-xl w-full max-w-md shadow-2xl p-8 transform animate-in slide-in-from-bottom-4 duration-300"
               (click)="$event.stopPropagation()">
            <div class="flex items-center gap-4 mb-6">
              <div class="p-3 bg-amber-500/10 rounded-theme-md border border-amber-500/20 text-amber-400">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                  <line x1="12" y1="9" x2="12" y2="13"></line>
                  <line x1="12" y1="17" x2="12.01" y2="17"></line>
                </svg>
              </div>
              <div>
                <h3 class="text-lg font-bold text-white">Model Changed</h3>
                <p class="text-xs text-text-subtle mt-0.5">You have targeted layers selected for the previous model</p>
              </div>
            </div>
            <p class="text-sm text-text-secondary mb-6">
              The targeted layer selection is not compatible with the new model. Would you like to <strong class="text-white">keep your current model and layers</strong>, or <strong class="text-white">switch models and reset</strong> the layer selection?
            </p>
            <div class="flex justify-end gap-3">
              <button (click)="onModelChangeKeep()"
                      class="text-text-subtle hover:text-white text-sm font-bold px-5 py-2.5 transition-colors uppercase tracking-widest border border-surface-mid rounded-theme-md hover:border-surface-high">
                Keep Model & Layers
              </button>
              <button (click)="onModelChangeReset()"
                      class="bg-brand hover:bg-brand/80 text-white text-sm font-bold px-5 py-2.5 rounded-theme-md transition-colors uppercase tracking-widest">
                Switch & Reset
              </button>
            </div>
          </div>
        </div>
      }
    }
  `,
  styleUrl: 'training-dynamic-config.css'
})
export class TrainingDynamicConfigComponent {
  schema = input<any>(); // JSON Schema
  availableModels = input<any[]>([]); // New input for model list
  configSubmitted = output<any>();

  private fb = inject(FormBuilder);
  private http = inject(HttpClient);
  private toast = inject(ToastService);
  private systemService = inject(SystemService);

  // VRAM estimation
  vramReport = signal<VRAMReport | null>(null);
  private vramEstimate$ = new Subject<void>();
  Math = Math; // expose to template

  form: FormGroup = new FormGroup({});
  properties = signal<{ key: string, schema: any }[]>([]);
  groups = signal<{ name: string, props: { key: string, schema: any }[] }[]>([]);
  modelSelectionProps = signal<{ key: string, schema: any }[]>([]);
  nestedItemPropsMap = signal<Record<string, { key: string, schema: any }[]>>({});

  // Collapsible groups — ENGINE starts collapsed
  collapsedGroups = signal<Set<string>>(new Set(['Advanced Engine', 'Sampling', 'Expert Features']));

  // Dataset autocomplete
  availableDatasets = signal<string[]>([]);

  @ViewChild(TrainingTemplateSelectorComponent) templateSelector!: TrainingTemplateSelectorComponent;
  @ViewChild(AdvancedVramCardComponent) advancedVramCard!: AdvancedVramCardComponent;
  @ViewChild(TargetLayersCardComponent) targetLayersCard!: TargetLayersCardComponent;

  // Config Help System
  configHelp = signal<Record<string, { tip: string; detail: string }>>({});
  activeHelpKey = signal<string | null>(null);

  // Model selection state
  families = computed(() => [...new Set(this.availableModels().map(m => m.family))]);
  selectedFamily = signal<string>('');
  filteredDefinitions = computed(() => this.availableModels().filter(m => m.family === this.selectedFamily()));

  // Track the current definition_id reactively
  currentDefinitionId = signal<string>('');

  // Model change → target layers modal
  showModelChangeModal = signal(false);
  private _isTemplateApplying = false;
  private _pendingDefinitionId: string | null = null;
  private _previousModelFamily: string | null = null;
  private _previousDefinitionId: string | null = null;
  // Explicit tracking — always in sync, even after emitEvent:false reverts
  private _lastKnownModelFamily = '';
  private _lastKnownDefinitionId = '';

  selectedDefinition = computed(() => {
    const defId = this.currentDefinitionId();
    return this.availableModels().find(m => m.id === defId);
  });

  private datasetService = inject(DatasetService);
  private destroyRef = inject(DestroyRef);

  constructor() {
    this.http.get<Record<string, { tip: string; detail: string }>>('/config_help.json')
      .subscribe(data => this.configHelp.set(data));

    // Debounced VRAM estimation trigger
    this.vramEstimate$.pipe(
      debounceTime(800),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe(() => this.refreshVRAMEstimate());

    effect(() => {
      const schema = this.schema();
      if (schema) {
        this.buildForm();

        // Ensure definition_id is initialized if form is empty
        const currentDef = this.form.get('definition_id')?.value;
        const model = this.availableModels().find(m => m.id === currentDef);

        if (model) {
          this.selectedFamily.set(model.family);
        } else if (this.availableModels().length > 0) {
          const firstModel = this.availableModels()[0];
          this.selectedFamily.set(firstModel.family);
          this.form.get('definition_id')?.setValue(firstModel.id);
        }

        // Trigger initial VRAM estimate
        this.vramEstimate$.next();

        // Re-estimate on any config field change (definition, quantization, LoRA rank, etc.)
        this.form.valueChanges.pipe(
          debounceTime(800),
          takeUntilDestroyed(this.destroyRef),
        ).subscribe(() => this.vramEstimate$.next());

        // Auto-save: persist changes to the active template on every form change
        this.form.valueChanges.pipe(
          debounceTime(1200),
          takeUntilDestroyed(this.destroyRef),
        ).subscribe((newVal) => {
          const defId = this.form.get('definition_id')?.value;
          if (this.templateSelector && defId) {
            this.templateSelector.triggerAutoSave(newVal, defId);
          }
        });
      }
    });
  }

  onFamilyChange(event: any) {
    const family = event.target.value;
    this.selectedFamily.set(family);
    // Auto-select first definition in family
    const defs = this.filteredDefinitions();
    if (defs.length > 0) {
      this.form.get('definition_id')?.setValue(defs[0].id);
    }
    this.vramEstimate$.next();
  }

  // --- VRAM Estimation ---

  refreshVRAMEstimate(): void {
    const defId = this.form.get('definition_id')?.value;
    if (!defId) return;

    const config = this.form.getRawValue();
    this.systemService.estimateVRAM(defId, config).subscribe({
      next: (report) => this.vramReport.set(report),
      error: (err) => {
        console.warn('[VRAM] Estimation failed', err);
        this.vramReport.set(null);
      }
    });
  }

  // --- Config Help System ---

  hasHelp(key: string): boolean {
    return !!this.configHelp()[key];
  }

  getHelpTip(key: string): string {
    return this.configHelp()[key]?.tip || '';
  }

  getHelpTitle(key: string): string {
    const schema = this.schema();
    if (schema?.properties?.[key]?.title) {
      return schema.properties[key].title;
    }
    return key.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase());
  }

  openHelpModal(key: string): void {
    this.activeHelpKey.set(key);
  }

  closeHelpModal(): void {
    this.activeHelpKey.set(null);
  }

  // --- Model Change → Target Layers Modal ---

  onModelChangeKeep(): void {
    // Revert model family and definition silently (emitEvent: false)
    // to bypass the depends_on cascade and prevent re-triggering the modal
    if (this._previousModelFamily != null) {
      this.form.get('model_family')?.setValue(this._previousModelFamily, { emitEvent: false });
      this.selectedFamily.set(this._previousModelFamily);
      this._lastKnownModelFamily = this._previousModelFamily;
    }
    if (this._previousDefinitionId != null) {
      this.form.get('definition_id')?.setValue(this._previousDefinitionId, { emitEvent: false });
      this.currentDefinitionId.set(this._previousDefinitionId);
      this._lastKnownDefinitionId = this._previousDefinitionId;
    }
    this.showModelChangeModal.set(false);
    this._pendingDefinitionId = null;
    this._previousModelFamily = null;
    this._previousDefinitionId = null;
  }

  onModelChangeReset(): void {
    const ctrl = this.form.get('targeted_layers') as FormControl<string[]>;
    if (ctrl) {
      ctrl.setValue([]);
    }
    // Sync signals to the actual form values (depends_on used emitEvent:false,
    // so the signals are still pointing to the OLD model)
    const newFamily = this.form.get('model_family')?.value || '';
    const newDefId = this.form.get('definition_id')?.value || '';
    this.selectedFamily.set(newFamily);
    this.currentDefinitionId.set(newDefId);
    this._lastKnownModelFamily = newFamily;
    this._lastKnownDefinitionId = newDefId;
    // Trigger capabilities reload for advanced VRAM card
    if (newDefId && this.advancedVramCard) {
      this.advancedVramCard.loadCapabilities(newDefId);
    }
    this.showModelChangeModal.set(false);
    this._pendingDefinitionId = null;
    this._previousModelFamily = null;
    this._previousDefinitionId = null;
  }

  private _checkTargetLayersOnModelChange(prevFamily: string, prevDefId: string): void {
    const ctrl = this.form.get('targeted_layers') as FormControl<string[]>;
    if (ctrl && ctrl.value?.length > 0) {
      // Snapshot the previous model so "Keep" can revert
      this._previousModelFamily = prevFamily;
      this._previousDefinitionId = prevDefId;
      this.showModelChangeModal.set(true);
    }
  }

  renderHelpDetail(key: string): string {
    const raw = this.configHelp()[key]?.detail || '';
    return raw
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/^- (.+)$/gm, '<li>$1</li>')
      .replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>')
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n/g, '<br>');
  }

  // --- Template Logic ---

  onTemplateApplied(event: { config: any, isDefault: boolean, definitionId?: string }) {
    this._isTemplateApplying = true;
    if (event.isDefault) {
      this.resetFormToDefaults();
    } else {
      if (event.definitionId) {
        const model = this.availableModels().find(m => m.id === event.definitionId);
        if (model) {
          this.selectedFamily.set(model.family);
          // Explicitly update the form control so valueChanges triggers the tracking signal
          this.form.get('definition_id')?.setValue(event.definitionId);
        }
      }
      this.patchFormRecursive(this.form, event.config);
      // Sync block swap slider values from loaded template
      this._syncBlockSwapFromForm();
      // Rebuild target layers tree from the newly-patched control
      this.targetLayersCard?.refreshFromControl();
    }

    // Release the child's auto-save suppression after patching is done
    setTimeout(() => {
      this._isTemplateApplying = false;
      if (this.templateSelector) {
        this.templateSelector.suppressAutoSave.set(false);
      }
    }, 1500);
  }

  resetFormToDefaults() {
    const schema = this.schema();
    const props = schema.properties || {};

    Object.keys(this.form.controls).forEach(key => {
      if (key === 'definition_id') return;

      const propSchema = this.resolveSchema(props[key]);
      const control = this.form.get(key);

      if (propSchema.type === 'array') {
        const formArray = control as FormArray;
        formArray.clear();
        const defaults = propSchema.default || [];
        defaults.forEach((d: any) => {
          this.addArrayItem(key, propSchema.items);
        });
        if (this.isPrimitiveArray(key)) {
          formArray.patchValue(defaults);
        }
      } else {
        const def = propSchema.default !== undefined ? propSchema.default : '';
        control?.setValue(def);
      }
    });
  }

  patchFormRecursive(formGroup: FormGroup | FormArray, config: any) {
    Object.keys(config).forEach(key => {
      const control = formGroup.get(key);
      if (!control) return;

      const value = config[key];

      if (control instanceof FormArray) {
        while (control.length !== value.length) {
          if (control.length < value.length) {
            const schema = this.getSchemaForKey(key, formGroup);
            if (schema && schema.items) {
              this.addArrayItem(key, schema.items);
            } else {
              return;
            }
          } else {
            control.removeAt(control.length - 1);
          }
        }
        control.patchValue(value);

      } else if (control instanceof FormGroup) {
        this.patchFormRecursive(control, value);
      } else {
        control.setValue(value);
      }
    });
  }

  getControl(key: string): FormControl {
    return this.form.get(key) as FormControl;
  }

  getArrayStringControl(key: string): FormControl<string[]> {
    return this.form.get(key) as FormControl<string[]>;
  }

  getSchemaForKey(key: string, parent: any): any {
    const prop = this.properties().find(p => p.key === key);
    if (prop) return prop.schema;
    return null;
  }

  // --- External Config Import (from Job Queue) ---

  importTemplate(name: string, config: any, definitionId: string) {
    if (this.templateSelector) {
      this.templateSelector.importExternalTemplate(name, config, definitionId);
    }

    // Switch family to match the template's definition
    const model = this.availableModels().find(m => m.id === definitionId);
    if (model) {
      this.selectedFamily.set(model.family);
    }
    this.patchFormRecursive(this.form, config);
  }

  loadExternalConfig(config: any) {
    // Preserve session-specific fields that should not be overwritten by templates
    const resumePath = this.form.get('resume_from_checkpoint')?.value;

    // Switch family + definition to match the config being loaded
    const defId = config['definition_id'];
    if (defId) {
      const model = this.availableModels().find(m => m.id === defId);
      if (model) {
        this.selectedFamily.set(model.family);
      }
    }
    this.patchFormRecursive(this.form, config);

    // Restore session-specific fields
    if (resumePath) {
      this.form.get('resume_from_checkpoint')?.setValue(resumePath);
    }

    // Sync block swap slider values from loaded config
    this._syncBlockSwapFromForm();
    // Rebuild target layers tree from the newly-patched control
    this.targetLayersCard?.refreshFromControl();
  }
  buildForm() {
    const schema = this.schema();
    const props = schema.properties || {};
    const group: any = {};
    const properties: { key: string, schema: any }[] = [];
    const nestedItemPropsMap: Record<string, { key: string, schema: any }[]> = {};

    for (const key in props) {
      if (props.hasOwnProperty(key)) {
        const propSchema = this.resolveSchema(props[key]);
        properties.push({ key, schema: propSchema });

        if (propSchema.type === 'array') {
          if (propSchema.ui_type === 'layer_checklist') {
            // we want a flat FormControl<string[]> instead of a dynamic FormArray of FormControls
            group[key] = new FormControl(propSchema.default || []);
          } else {
            const array = this.fb.array([]);
            group[key] = array;

            if (this.isPrimitiveArrayForKey(key, properties)) {
              const defaults = propSchema.default || [];
              defaults.forEach((d: any) => {
                array.push(new FormControl(d));
              });
            } else {
              nestedItemPropsMap[key] = this.getArrayItemProps(propSchema.items);
            }
          }
        } else if (propSchema.type === 'object') {
          // block_swap_sliders uses a plain FormControl so setValue() works with dynamic keys
          if (propSchema.ui_type === 'block_swap_sliders') {
            group[key] = new FormControl(propSchema.default ?? {});
          } else {
            group[key] = this.fb.group({}); // Simple placeholder
          }
        } else {
          const validators = [];
          if (schema.required && schema.required.includes(key)) {
            validators.push(Validators.required);
          }
          const defaultValue = propSchema.default !== undefined ? propSchema.default : '';
          group[key] = new FormControl(defaultValue, validators);
        }
      }
    }
    this.form = this.fb.group(group);
    this.properties.set(properties);
    this.nestedItemPropsMap.set(nestedItemPropsMap);

    // Group properties for categorization in UI
    this.organizeGroups(properties);

    // Auto-add first dataset if none exist
    if (group['datasets'] && this.getFormArray('datasets').length === 0) {
      const datasetsSchema = props['datasets'];
      if (datasetsSchema && datasetsSchema.items) {
        this.addArrayItem('datasets', datasetsSchema.items);
      }
    }

    // Auto-set LR defaults when optimizer changes
    this.form.get('optimizer_type')?.valueChanges.pipe(
      takeUntilDestroyed(this.destroyRef)
    ).subscribe((optimizer: string) => {
      const lrControl = this.form.get('learning_rate');
      if (!lrControl) return;
      const defaults: Record<string, number> = {
        'AdamW': 0.0001,
        'AdamW8bit': 0.0001,
        'Prodigy': 1.0,
        'ProdigyPlusSF': 1.0,
        'SophiaH': 0.0001,
        'SophiaG': 0.0001,
        'Lion': 0.0001,
        'Adafactor': 0.0001,
        'StableAdamW': 0.0001,
        'Shampoo': 0.0001,
        'RAdam': 0.0001,
        'AdEMAMix': 0.0001,
      };
      if (defaults[optimizer] !== undefined) {
        lrControl.setValue(defaults[optimizer]);
      }
      // Reset LR scaling for adaptive optimizers (they adapt LR internally)
      const scaleControl = this.form.get('lr_scale_mode');
      if (scaleControl && (optimizer === 'Prodigy' || optimizer === 'ProdigyPlusSF')) {
        scaleControl.setValue('none');
      }
    });

    // Auto-set LR when Adafactor relative_step is toggled
    this.form.get('adafactor_relative_step')?.valueChanges.pipe(
      takeUntilDestroyed(this.destroyRef)
    ).subscribe((enabled: boolean) => {
      const lrControl = this.form.get('learning_rate');
      if (!lrControl) return;
      if (this.form.get('optimizer_type')?.value === 'Adafactor') {
        lrControl.setValue(enabled ? 1.0 : 0.0001);
      }
    });

    // Auto-filter dependent fields (e.g. backend_map restrictions)
    properties.forEach(prop => {
      let parentKey = prop.schema.depends_on;
      if (!parentKey && prop.schema.backend_map) {
        // Fallback heuristics if 'depends_on' is missing but backend_map is present
        if (prop.key === 'quantization') parentKey = 'quantization_backend';
        else if (prop.key === 'te_quantization') parentKey = 'te_quantization_backend';
      }

      if (parentKey && this.form.get(parentKey) && this.form.get(prop.key)) {
        this.form.get(parentKey)?.valueChanges.pipe(
          takeUntilDestroyed(this.destroyRef)
        ).subscribe(() => {
          // When parent changes, re-evaluate dropdown options for child
          const childControl = this.form.get(prop.key);
          if (!childControl) return;

          const validOptions = this.getFilteredEnumOptions(prop);
          const currentValue = childControl.value;
          const isCurrentValueValid = validOptions.some(opt => opt.value === currentValue && !opt.disabled);

          if (!isCurrentValueValid) {
            // Get the first valid, non-disabled option, or default to empty string
            const firstValid = validOptions.find(opt => !opt.disabled);
            childControl.setValue(firstValid ? firstValid.value : '', { emitEvent: false });
          }
        });
      }
    });

    // Handle programmatic enable/disable states to avoid Angular [disabled] warnings
    const updateDisabledStates = () => {
      properties.forEach(prop => {
        if (prop.schema.depends_on) {
          const ctrl = this.form.get(prop.key);
          if (ctrl) {
            if (this.isFieldDisabled(prop.schema)) {
              if (ctrl.enabled) ctrl.disable({ emitEvent: false });
            } else {
              if (ctrl.disabled) ctrl.enable({ emitEvent: false });
            }
          }
        }
      });
    };

    // Run once on load
    updateDisabledStates();

    // Sync definition_id to reactive signal for UI updates
    this.currentDefinitionId.set(this.form.get('definition_id')?.value || '');
    this._lastKnownDefinitionId = this.form.get('definition_id')?.value || '';
    this._lastKnownModelFamily = this.form.get('model_family')?.value || '';

    this.form.get('definition_id')?.valueChanges.pipe(
      takeUntilDestroyed(this.destroyRef)
    ).subscribe(val => {
      // Check if user-initiated model change with existing targeted_layers
      if (!this._isTemplateApplying && val) {
        if (this._lastKnownDefinitionId && this._lastKnownDefinitionId !== val) {
          this._checkTargetLayersOnModelChange(this._lastKnownModelFamily, this._lastKnownDefinitionId);
        }
      }
      this._lastKnownDefinitionId = val || '';
      this.currentDefinitionId.set(val || '');
      // Trigger capabilities fetch for advanced VRAM card
      if (val && this.advancedVramCard) {
        this.advancedVramCard.loadCapabilities(val);
      }
    });

    // Watch model_family changes (definition_id auto-set uses emitEvent:false,
    // so we need this separate watcher for family-level changes)
    this.form.get('model_family')?.valueChanges.pipe(
      takeUntilDestroyed(this.destroyRef)
    ).subscribe((newFamily) => {
      if (!this._isTemplateApplying && this._lastKnownModelFamily !== newFamily) {
        this._checkTargetLayersOnModelChange(this._lastKnownModelFamily, this._lastKnownDefinitionId);
      }
      this._lastKnownModelFamily = newFamily || '';
    });

    // Re-run on any form value change
    this.form.valueChanges.pipe(
      takeUntilDestroyed(this.destroyRef)
    ).subscribe(() => {
      updateDisabledStates();
    });
  }

  organizeGroups(properties: { key: string, schema: any }[]) {
    const groupMap: Record<string, { key: string, schema: any }[]> = {};
    const groupOrder = ['MODEL_SELECTION', 'BASE', 'CONCEPTS', 'STRATEGY', 'NETWORK', 'OPTIMIZER', 'OPTIMIZER_EXPERT', 'SAMPLING', 'ENGINE', 'OTHER'];

    properties.forEach(prop => {
      const groupName = prop.schema.group || 'OTHER';
      if (!groupMap[groupName]) groupMap[groupName] = [];
      groupMap[groupName].push(prop);
    });

    // Enforce logical sub-grouping within STRATEGY:
    //   save_every_n_steps → persist_latents → persist_embeddings
    //   resume_from_checkpoint → use_cached_latents → use_cached_embeddings
    const strategyFieldOrder = [
      'max_train_steps', 'train_batch_size', 'gradient_accumulation_steps',
      'gradient_checkpointing',
      'save_every_n_steps', 'keep_last_checkpoints',
      'persist_latents', 'persist_embeddings',
      'resume_from_checkpoint', 'use_cached_latents', 'use_cached_embeddings',
      'resolutions', 'resolution_strategy', 'bucketing_mode',
      'timestep_sampling', 'logit_normal_mu', 'logit_normal_sigma',
    ];
    if (groupMap['STRATEGY']) {
      groupMap['STRATEGY'].sort((a, b) => {
        const ia = strategyFieldOrder.indexOf(a.key);
        const ib = strategyFieldOrder.indexOf(b.key);
        // Unknown keys go to the end, preserving their original relative order
        return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
      });
    }

    // Extract MODEL_SELECTION props for the hardcoded section
    this.modelSelectionProps.set(groupMap['MODEL_SELECTION'] || []);

    const groups = groupOrder
      .filter(name => name !== 'MODEL_SELECTION' && groupMap[name] && groupMap[name].length > 0)
      .map(name => ({
        name: this.formatGroupName(name),
        props: groupMap[name]
      }));

    // Catch-all for groups not in groupOrder
    Object.keys(groupMap).forEach(name => {
      if (!groupOrder.includes(name) && name !== 'MODEL_SELECTION') {
        groups.push({
          name: this.formatGroupName(name),
          props: groupMap[name]
        });
      }
    });

    this.groups.set(groups);
  }

  formatGroupName(name: string): string {
    const labels: Record<string, string> = {
      'BASE': 'General Settings',
      'MODEL_SELECTION': 'Model Selection',
      'STRATEGY': 'Training Dynamics',
      'NETWORK': 'LoRA Parameters',
      'OPTIMIZER': 'Optimizer Settings',
      'OPTIMIZER_EXPERT': 'Expert Features',
      'ENGINE': 'Advanced Engine',
      'CONCEPTS': 'Concepts & Triggerwords',
      'SAMPLING': 'Sampling'
    };
    return labels[name] || name;
  }

  resolveSchema(schema: any): any {
    const definitions = this.schema().$defs || this.schema().definitions || {};
    if (schema && schema.$ref) {
      const refKey = schema.$ref.split('/').pop();
      if (definitions[refKey]) {
        return { ...definitions[refKey], ...schema };
      }
    }
    return schema;
  }

  getFormArray(key: string): FormArray {
    return this.form.get(key) as FormArray;
  }

  addArrayItem(key: string, itemSchemaRef: any) {
    const itemSchema = this.resolveSchema(itemSchemaRef);
    const array = this.getFormArray(key);

    if (itemSchema.properties) {
      // Object
      const itemProps = itemSchema.properties || {};
      const group: any = {};

      for (const pKey in itemProps) {
        const pSchema = this.resolveSchema(itemProps[pKey]);
        let defaultValue = pSchema.default !== undefined ? pSchema.default : '';

        // Pick first enum value as default if currently empty
        if (pSchema.enum && pSchema.enum.length > 0 && (defaultValue === '' || defaultValue === undefined)) {
          defaultValue = pSchema.enum[0];
        }

        group[pKey] = new FormControl(defaultValue);
      }
      array.push(this.fb.group(group));
    } else {
      // Primitive
      let defaultValue = itemSchema.default !== undefined ? itemSchema.default : '';
      if (itemSchema.type === 'integer' || itemSchema.type === 'number') {
        defaultValue = defaultValue || 0;
      }

      const validators = [];
      if (key === 'resolutions') {
        validators.push((control: any) => {
          const val = parseInt(control.value);
          return (val > 0 && val % 32 === 0) ? null : { 'mod32': true };
        });
      }

      array.push(new FormControl(defaultValue, validators));
    }
  }

  // --- Resolution Helpers ---
  isPreset(val: any): boolean {
    return [512, 768, 1024, 1280, 1536].includes(parseInt(val));
  }

  isResolutionSelected(res: number): boolean {
    const array = this.getFormArray('resolutions');
    if (!array) return false;
    return array.value.some((id: any) => parseInt(id) === res);
  }

  toggleResolution(res: number) {
    const array = this.getFormArray('resolutions');
    if (!array) return;

    const index = array.value.findIndex((v: any) => parseInt(v) === res);
    if (index >= 0) {
      array.removeAt(index);
    } else {
      array.push(new FormControl(res, [(control: any) => {
        const val = parseInt(control.value);
        return (val > 0 && val % 32 === 0) ? null : { 'mod32': true };
      }]));
    }
  }

  removeArrayItem(key: string, index: number) {
    this.getFormArray(key).removeAt(index);
  }

  // --- Dynamic Dropdown Filtering ---

  /**
   * Returns a filtered list of `{value, label, disabled}` based on a dynamically injected `backend_map`.
   */
  getFilteredEnumOptions(prop: { key: string, schema: any }): { value: string, label: string, disabled: boolean }[] {
    const defaultOptions = (prop.schema.enum || []).map((opt: string, idx: number) => {
      const label = prop.schema.enum_labels && prop.schema.enum_labels[idx] ? prop.schema.enum_labels[idx] : opt;
      return { value: opt, label: label, disabled: false };
    });

    // If there is no backend_map injected, fallback to standard enum mapping
    if (!prop.schema.backend_map) {
      return defaultOptions;
    }

    // Retrieve the backend value from the form
    let backendKey = prop.schema.depends_on;
    if (!backendKey) {
      // Fallback heuristics if 'depends_on' is missing
      if (prop.key === 'quantization') backendKey = 'quantization_backend';
      else if (prop.key === 'te_quantization') backendKey = 'te_quantization_backend';
    }

    const currentBackend = backendKey ? this.form.get(backendKey)?.value : 'auto';
    const capabilities = prop.schema.backend_map;

    // If backend is 'auto', do not disable anything
    if (!currentBackend || currentBackend === 'auto') {
      return defaultOptions;
    }

    // If a specific backend is selected, check its supported schemes logic
    const backendData = capabilities[currentBackend];
    if (!backendData) {
      return defaultOptions; // Unknown backend, show all just in case
    }

    const supportedSchemes: string[] = Array.isArray(backendData) ? backendData : (backendData.schemes || []);

    return defaultOptions.map((opt: { value: string, label: string, disabled: boolean }) => {
      // 'none' or 'bf16' are universally available across backends
      if (opt.value === 'none' || opt.value === 'bf16') {
        return opt;
      }

      // Check if the current option is explicitly supported by the selected backend
      if (!supportedSchemes.includes(opt.value)) {
        return { ...opt, disabled: true, label: `${opt.label} (Not supported by ${currentBackend})` };
      }

      return opt;
    }).filter((opt: { value: string, label: string, disabled: boolean }) => !(prop.schema.hide_unsupported && opt.disabled));
  }

  /**
   * Same as above, but for arrays of objects where the form control lives inside a FormArray.
   */
  getFilteredNestedEnumOptions(nestedProp: { key: string, schema: any }, arrayKey: string, dsIdx: number): { value: string, label: string, disabled: boolean }[] {
    const defaultOptions = (nestedProp.schema.enum || []).map((opt: string, idx: number) => {
      const label = nestedProp.schema.enum_labels && nestedProp.schema.enum_labels[idx] ? nestedProp.schema.enum_labels[idx] : opt;
      return { value: opt, label: label, disabled: false };
    });

    if (!nestedProp.schema.backend_map || !nestedProp.schema.depends_on) {
      return defaultOptions;
    }

    const backendKey = nestedProp.schema.depends_on;
    const formGroup = this.getFormArray(arrayKey).at(dsIdx);
    const currentBackend = formGroup?.get(backendKey)?.value || 'auto';
    const capabilities = nestedProp.schema.backend_map;

    if (!currentBackend || currentBackend === 'auto') return defaultOptions;

    const backendData = capabilities[currentBackend];
    if (!backendData) return defaultOptions;

    const supportedSchemes: string[] = Array.isArray(backendData) ? backendData : (backendData.schemes || []);

    return defaultOptions.map((opt: { value: string, label: string, disabled: boolean }) => {
      if (opt.value === 'none' || opt.value === 'bf16') return opt;
      if (!supportedSchemes.includes(opt.value)) {
        return { ...opt, disabled: true, label: `${opt.label} (Not supported by ${currentBackend})` };
      }
      return opt;
    }).filter((opt: { value: string, label: string, disabled: boolean }) => !(nestedProp.schema.hide_unsupported && opt.disabled));
  }

  toggleIgnoreFilter(index: number) {
    const ctrl = this.getFormArray('datasets').at(index).get('ignore_filter');
    if (ctrl) ctrl.setValue(!ctrl.value);
  }

  isPrimitiveArrayForKey(key: string, properties: { key: string, schema: any }[]): boolean {
    const prop = properties.find(p => p.key === key);
    if (!prop || (prop.schema && prop.schema.type !== 'array')) return false;
    const items = this.resolveSchema(prop.schema.items);
    return !items.properties;
  }

  isPrimitiveArray(key: string): boolean {
    return this.isPrimitiveArrayForKey(key, this.properties());
  }

  getArrayItemProps(itemSchemaRef: any): { key: string, schema: any }[] {
    const itemSchema = this.resolveSchema(itemSchemaRef);
    const props = itemSchema.properties || {};
    return Object.keys(props)
      .map(key => ({ key, schema: this.resolveSchema(props[key]) }))
      .filter(p => !p.schema.hidden);
  }

  getInlineGroups(props: { key: string, schema: any }[]): { name: string, props: { key: string, schema: any }[] }[] {
    const map: Record<string, { key: string, schema: any }[]> = {};
    for (const prop of props) {
      const ig = prop.schema.inline_group;
      if (ig) {
        if (!map[ig]) map[ig] = [];
        map[ig].push(prop);
      }
    }
    return Object.keys(map).map(name => ({ name, props: map[name] }));
  }

  isFirstInlineGroupProp(key: string, props: { key: string, schema: any }[]): boolean {
    const group = props.find(p => p.key === key)?.schema.inline_group;
    if (!group) return false;
    return props.find(p => p.schema.inline_group === group)?.key === key;
  }

  getInlineGroupProps(groupName: string, props: { key: string, schema: any }[]): { key: string, schema: any }[] {
    return props.filter(p => p.schema.inline_group === groupName);
  }


  isNumber(schema: any): boolean {
    return schema.type === 'number' || schema.type === 'integer';
  }

  isBoolean(schema: any): boolean {
    return schema.type === 'boolean';
  }

  isString(schema: any): boolean {
    return schema.type === 'string' || !schema.type;
  }

  isObject(schema: any): boolean {
    return schema.type === 'object';
  }

  isLongInput(key: string, schema: any): boolean {
    return key === 'lora_name' || key === 'output_dir' || key === 'global_triggerword'
      || key === 'save_every_n_steps' || key === 'resume_from_checkpoint'
      || schema.type === 'array' || schema.input_type === 'path';
  }

  // --- Collapsible Groups ---
  toggleGroup(groupName: string) {
    this.collapsedGroups.update(current => {
      const next = new Set(current);
      if (next.has(groupName)) {
        next.delete(groupName);
      } else {
        next.add(groupName);
      }
      return next;
    });
  }

  isGroupCollapsed(groupName: string): boolean {
    return this.collapsedGroups().has(groupName);
  }

  /** Hide an entire group when ALL its fields are hidden by depends_on */
  isGroupHidden(group: { name: string, props: { key: string, schema: any }[] }): boolean {
    return group.props.length > 0 && group.props.every(p => this.shouldHideField(p.schema));
  }

  // --- Conditional Field Disable ---
  isFieldDisabled(schema: any): boolean {
    if (!schema.depends_on) return false;
    const dep: string = schema.depends_on;

    // OR multi-field: "fieldA:!none|fieldB:!none" — enabled when ANY condition passes
    if (dep.includes('|')) {
      const parts = dep.split('|');
      // Disabled only if ALL parts say "disabled"
      return parts.every(part => this._checkSingleDep(part));
    }

    return this._checkSingleDep(dep);
  }

  private _checkSingleDep(dep: string): boolean {
    // Value-matching: "field:value" or "field:!value" (negation) or "field:val1,val2"
    if (dep.includes(':')) {
      const [field, expectedValues] = dep.split(':', 2);
      const currentValue = this.form.get(field)?.value;
      // Negation support: "field:!0" means disabled when field equals 0
      if (expectedValues.startsWith('!')) {
        const negatedValues = expectedValues.slice(1).split(',');
        return negatedValues.includes(String(currentValue));
      }
      const allowedValues = expectedValues.split(',');
      return !allowedValues.includes(currentValue);
    }
    // Boolean: disabled if parent is falsy
    const parentValue = this.form.get(dep)?.value;
    return !parentValue;
  }



  shouldHideField(schema: any): boolean {
    if (schema.hidden) return true;
    if (!schema.depends_on) return false;
    const dep: string = schema.depends_on;

    // OR multi-field: "fieldA:!none|fieldB:!none" — show when ANY condition passes
    if (dep.includes('|')) {
      const parts = dep.split('|');
      // Hide only if ALL parts say "hide"
      return parts.every(part => this._shouldHideSingleDep(part));
    }

    return this._shouldHideSingleDep(dep);
  }

  private _shouldHideSingleDep(dep: string): boolean {
    // Only hide for value-matching depends_on (enum sub-fields)
    if (dep.includes(':')) {
      const [field, expectedValues] = dep.split(':', 2);
      const currentValue = this.form.get(field)?.value;
      // Negation support: "field:!0" means hide when field equals 0
      if (expectedValues.startsWith('!')) {
        const negatedValues = expectedValues.slice(1).split(',');
        return negatedValues.includes(String(currentValue));
      }
      // Support comma-separated values: "optimizer_type:AdamW,AdamW8bit"
      const allowedValues = expectedValues.split(',');
      return !allowedValues.includes(currentValue);
    }
    // Boolean depends_on: show but disable (don't hide)
    return false;
  }

  // --- Dataset Autocomplete ---

  onSubmit() {
    if (this.form.valid) {
      const raw = this.form.value;
      const schema = this.schema();
      if (schema?.properties) {
        // Strip fields that are hidden due to depends_on mismatch
        // (but NOT fields with hidden:true — those are just visually hidden
        // and rendered by custom components like TargetLayersCard)
        for (const [key, propSchema] of Object.entries(schema.properties)) {
          const ps = propSchema as any;
          if (!ps.hidden && this.shouldHideField(ps)) {
            delete raw[key];
          }
        }
        // Strip targeted_layers when empty (= filtering OFF, train all layers)
        if (Array.isArray(raw['targeted_layers']) && raw['targeted_layers'].length === 0) {
          delete raw['targeted_layers'];
        }
      }
      this.configSubmitted.emit(raw);
    }
  }

  /** Called by AdvancedVramCard when block swap sliders change. */
  onBlockSwapChanged(config: Record<string, number>): void {
    const control = this.form.get('block_swap_config');
    if (control) {
      control.setValue(config);
    }
  }

  /** Push form's block_swap_config values to the AdvancedVramCard sliders. */
  private _syncBlockSwapFromForm(): void {
    const val = this.form.get('block_swap_config')?.value;
    if (val && typeof val === 'object' && Object.keys(val).length > 0 && this.advancedVramCard) {
      this.advancedVramCard.setSwapValues(val);
    }
  }
}
