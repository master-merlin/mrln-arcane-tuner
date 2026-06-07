import { Component, input, output, inject, effect, ChangeDetectionStrategy, ChangeDetectorRef } from '@angular/core';
import { TitleCasePipe, DatePipe } from '@angular/common';
import { ReactiveFormsModule, FormArray, FormGroup, FormControl, AbstractControl } from '@angular/forms';
import { DatasetService, Dataset } from '../../../services/dataset';
import { DatasetStore } from '../../../state/dataset.store';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { StatePillsComponent, StatePillsState, datasetStatePills } from '../../../ui/state-pills/state-pills.component';
import { DynamicFormFieldComponent } from '../dynamic-form-field/dynamic-form-field';
import type { TrainingConfig } from '../../../services/job';
import { SchemaNode, SchemaProp } from '../schema-node';

@Component({
  selector: 'app-dynamic-form-group',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [TitleCasePipe, DatePipe, ReactiveFormsModule, DynamicFormFieldComponent, StatePillsComponent],
  host: { 'class': 'contents' },
  template: `
    <div class="md:col-span-2 space-y-4">
       <div class="flex items-center justify-between">
           <div class="flex items-center gap-2">
             <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-brand-light">
               <path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"></path>
               <polyline points="3.29 7 12 12 20.71 7"></polyline>
               <line x1="12" y1="22" x2="12" y2="12"></line>
             </svg>
             <span class="text-xs font-bold text-text-subtle uppercase tracking-widest">{{ schema().title || (fieldKey() | titlecase) }}</span>
           </div>
           <!-- Resolutions has its own "+ Add Custom" control, so skip the duplicate generic Add here -->
           @if (fieldKey() !== 'resolutions') {
             <button type="button" (click)="addArrayItem()"
                [attr.data-testid]="'config-add-array-item-' + fieldKey()"
                class="bg-brand hover:bg-brand/90 text-white text-xs font-bold py-1 px-3 rounded-full transition-all flex items-center gap-1">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                Add {{ fieldKey() === 'datasets' ? 'Dataset' : 'Item' }}
             </button>
           }
       </div>

       <div [formGroup]="parentForm()" class="space-y-4">
         <div [formArrayName]="fieldKey()" class="space-y-4">
             <!-- Case 1: Array of Primitive Types -->
             @if (isPrimitiveArray()) {

               <!-- Specialized Resolutions Array overrides standard primitive array -->
               @if (fieldKey() === 'resolutions') {
                  <div class="space-y-3">
                     <!-- Presets -->
                     <div class="flex items-center justify-between">
                         <span class="text-[11px] font-medium text-text-muted">Presets</span>
                         <span class="text-[10px] text-text-subtle italic">Must be divisible by 32</span>
                     </div>
                     <div class="grid grid-cols-3 md:grid-cols-5 gap-2.5">
                         @for (res of [512, 768, 1024, 1280, 1536]; track res) {
                           <button type="button"
                                   (click)="toggleResolution(res)"
                                   [class.bg-brand]="isResolutionSelected(res)"
                                   [class.border-brand]="isResolutionSelected(res)"
                                   [class.text-white]="isResolutionSelected(res)"
                                   [class.bg-surface-mid]="!isResolutionSelected(res)"
                                   [class.text-text-subtle]="!isResolutionSelected(res)"
                                   [attr.data-testid]="'config-res-preset-' + res"
                                   class="py-1.5 px-2.5 rounded-theme-md border border-surface-high/50 text-[11px] font-bold font-mono transition-all hover:border-brand">
                               {{ res }}
                           </button>
                         }
                     </div>

                     <!-- Custom List -->
                     <div class="space-y-2">
                         <div class="flex items-center justify-between">
                             <span class="text-[11px] font-medium text-text-muted">Custom resolutions</span>
                             <button type="button" (click)="addArrayItem()"
                                     data-testid="config-add-custom-res-btn"
                                     class="text-brand hover:text-brand/80 text-[10px] font-bold uppercase tracking-tight">
                                 + Add Custom
                             </button>
                         </div>

                         <div class="grid grid-cols-3 md:grid-cols-5 gap-2.5">
                             @for (control of formArray().controls; track $index) {
                                 @if (!isPreset(control.value)) {
                                   <div class="relative group animate-in slide-in-from-bottom-2 duration-200">
                                       <input type="number" [formControlName]="$index"
                                              [attr.data-testid]="'config-custom-res-input-' + $index"
                                              class="w-full text-center py-1.5 px-2.5 rounded-theme-md border border-surface-high/50 bg-surface-mid text-text-secondary text-[11px] font-bold font-mono outline-none transition-all hover:border-brand focus:border-brand [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
                                              placeholder="e.g. 1440">
                                       <button type="button" (click)="removeArrayItem($index)"
                                          [attr.data-testid]="'config-remove-custom-res-btn-' + $index"
                                          class="absolute -top-1.5 -right-1.5 bg-surface-high border border-surface-mid rounded-full p-0.5 text-text-disabled hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
                                          title="Remove">
                                          <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                                       </button>
                                   </div>
                                 }
                             }
                         </div>
                     </div>
                  </div>

               } @else {
                 @for (control of formArray().controls; track $index) {
                    <div class="flex items-center gap-2 animate-in slide-in-from-left-2 duration-200">
                        <input [type]="isNumber(schema().items) ? 'number' : 'text'" 
                               [formControlName]="$index"
                               [attr.data-testid]="'config-array-input-' + fieldKey() + '-' + $index"
                               class="bg-surface-mid border border-surface-mid rounded-lg px-4 py-2 text-white w-full focus:ring-2 focus:ring-brand outline-none transition-all">
                        <button type="button" (click)="removeArrayItem($index)" 
                           [attr.data-testid]="'config-remove-array-item-' + fieldKey() + '-' + $index"
                           class="text-text-subtle hover:text-red-400 transition-colors p-2">
                           <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                        </button>
                    </div>
                 }
               }
             } @else {
               <!-- Case 2: Array of Objects -->
               @for (itemForm of formArray().controls; track $index; let dsIdx = $index) {
                  <div [formGroupName]="$index"
                       [attr.data-testid]="'config-array-object-' + fieldKey() + '-' + $index"
                       class="relative border-t border-surface-mid/40 pt-4 animate-in zoom-in-95 duration-200">

                      <button type="button" (click)="removeArrayItem($index)"
                         [attr.data-testid]="'config-remove-array-object-' + fieldKey() + '-' + $index"
                         class="absolute right-0 top-4 z-10 text-text-subtle hover:text-red-400 transition-colors"
                         title="Remove">
                         <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                      </button>

                      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-3.5 gap-y-3">

                          @for (nestedProp of nestedProps(); track nestedProp.key) {

                            <!-- Inline group: render all grouped toggles in one cell on first encounter -->
                            @if (nestedProp.schema.inline_group && isFirstInlineGroupProp(nestedProp.key)) {
                              <div class="flex flex-col gap-1.5">
                                <label class="field-label">
                                  {{ nestedProp.schema.inline_group === 'masking_toggles' ? 'Enable masking' : (nestedProp.schema.inline_group.replace('_', ' ') | titlecase) }}
                                </label>
                                <div class="flex flex-col gap-2 mt-1">
                                  @for (ip of getInlineGroupProps(nestedProp.schema.inline_group); track ip.key) {
                                    <div class="flex items-center gap-3 transition-opacity duration-200"
                                         [class.opacity-40]="isNestedFieldDisabled(ip.schema, formArray().at(dsIdx))"
                                         [class.pointer-events-none]="isNestedFieldDisabled(ip.schema, formArray().at(dsIdx))">
                                      <label class="relative inline-flex items-center cursor-pointer group">
                                        <input type="checkbox" [formControlName]="ip.key"
                                               [attr.data-testid]="'config-nested-checkbox-' + fieldKey() + '-' + ip.key"
                                               (change)="onNestedToggleChange(dsIdx, ip.key)"
                                               class="sr-only peer">
                                        <div class="w-7 h-4 bg-surface-high/50 border border-surface-mid rounded-full peer peer-focus:ring-2 peer-focus:ring-brand/50 peer-checked:after:translate-x-3 after:content-[''] after:absolute after:top-[1px] after:left-[2px] after:bg-white after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-brand group-hover:bg-surface-mid transition-all relative"></div>
                                      </label>
                                      <span class="text-[11px] text-text-muted">{{ ip.schema.title || (ip.key | titlecase) }}</span>
                                    </div>
                                  }
                                </div>
                              </div>
                            }

                            <!-- Skip non-first inline_group props (already rendered above) -->
                            @if (!nestedProp.schema.inline_group && !shouldHideNestedField(nestedProp.schema, formArray().at(dsIdx))) {
                              <div [class.col-span-full]="nestedProp.key === 'dataset_name'"
                                   [class.opacity-40]="isNestedFieldDisabled(nestedProp.schema, formArray().at(dsIdx))"
                                   [class.pointer-events-none]="isNestedFieldDisabled(nestedProp.schema, formArray().at(dsIdx))"
                                   class="flex flex-col gap-1.5 transition-opacity duration-200">
                                  
                                  <label class="field-label flex items-center gap-1.5">
                                    @if (nestedProp.key === 'dataset_name') {
                                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="text-brand"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>
                                    }
                                    {{ nestedProp.schema.title || (nestedProp.key | titlecase) }}
                                    @if (hasHelp(nestedProp.key)) {
                                      <span class="config-help-icon" [title]="getHelpTip(nestedProp.key)" (click)="requestHelp(nestedProp.key); $event.preventDefault()">?</span>
                                    }
                                    @if (nestedProp.key === 'dataset_name') {
                                      <!-- Status cluster: right-aligned, left of the ✕. Suppressed sits at the
                                           far left so the H·C·M pills keep their position. The pills are the
                                           library's <app-state-pills/> reused verbatim (incl. per-pill tooltips). -->
                                      <span class="flex items-center gap-2 ml-auto pr-7">
                                        @if (excludedCountFor(dsIdx) > 0) {
                                          <span class="chip warning"
                                                [title]="excludedCountFor(dsIdx) + ' image(s) currently excluded from training. Use the toggle below to include them this run.'">
                                            {{ excludedCountFor(dsIdx) }} suppressed
                                          </span>
                                        }
                                        @if (datasetFor(dsIdx)) {
                                          <app-state-pills [state]="datasetStateFor(dsIdx)"></app-state-pills>
                                        }
                                      </span>
                                    }
                                  </label>

                                  <!-- Dynamic Field rendering via standalone component for the nested object props -->
                                  @if (getNestedControl(dsIdx, nestedProp.key)) {
                                    <app-dynamic-form-field
                                        [control]="getNestedControl(dsIdx, nestedProp.key)"
                                        [schema]="nestedProp.schema"
                                        [fieldKey]="nestedProp.key"
                                        [currentBackend]="currentBackend()"
                                        [outputDir]="outputDir()"
                                        [hasHelp]="false"
                                        [hideLabel]="true"
                                        [datasetAutocomplete]="availableDatasets"
                                        (autofillRequested)="autofillCaptionPrefix(dsIdx)"
                                        (checkpointConfigLoaded)="checkpointConfigLoaded.emit($event)">
                                    </app-dynamic-form-field>
                                  } @else {
                                    <div class="text-xs text-red-500">Missing control: {{ nestedProp.key }}</div>
                                  }
                              </div>
                            }

                            <!-- Thumbnail + single info card beneath the dataset dropdown:
                                 a compact preview followed by one metadata card that fills the row. -->
                            @if (nestedProp.key === 'dataset_name' && datasetFor(dsIdx); as ds) {
                              <div class="col-span-full flex items-stretch gap-3">
                                <!-- Thumbnail (compact, fixed width) -->
                                <div class="w-40 shrink-0 rounded-theme-md overflow-hidden border border-surface-high/40 bg-base">
                                  @if (previewUrlFor(dsIdx); as url) {
                                    <img [src]="url" [alt]="ds.name" class="w-full h-full object-cover">
                                  } @else {
                                    <div class="w-full h-full min-h-[88px] flex items-center justify-center text-text-disabled bg-surface-low">
                                      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                                    </div>
                                  }
                                </div>

                                <!-- Info card (fills the remaining width) -->
                                <div class="flex-1 min-w-0 rounded-theme-md bg-surface-low/30 border border-surface-high/30 px-4 py-3 flex flex-col justify-center gap-2.5">
                                  <div class="grid grid-flow-col auto-cols-fr items-center gap-x-4">
                                    <div class="flex flex-col gap-0.5">
                                      <span class="text-[10px] uppercase tracking-wider text-text-subtle">Images</span>
                                      <span class="text-sm font-mono text-text-primary">{{ ds.multimedia_count || 0 }}</span>
                                    </div>
                                    <div class="flex flex-col gap-0.5">
                                      <span class="text-[10px] uppercase tracking-wider text-text-subtle">Captions</span>
                                      <span class="text-sm font-mono text-text-primary">{{ ds.caption_count || 0 }}</span>
                                    </div>
                                    <div class="flex flex-col gap-0.5">
                                      <span class="text-[10px] uppercase tracking-wider text-text-subtle">Masks</span>
                                      <span class="text-sm font-mono text-text-primary">{{ ds.mask_count || 0 }}</span>
                                    </div>
                                    <div class="flex flex-col gap-0.5">
                                      <span class="text-[10px] uppercase tracking-wider text-text-subtle">Files</span>
                                      <span class="text-sm font-mono text-text-primary">{{ ds.file_count || 0 }}</span>
                                    </div>
                                    <div class="flex flex-col gap-0.5">
                                      <span class="text-[10px] uppercase tracking-wider text-text-subtle">Size</span>
                                      <span class="text-sm font-mono text-text-primary">{{ sizeMbFor(dsIdx) }} <span class="text-[10px] text-text-subtle">MB</span></span>
                                    </div>
                                    <div class="flex flex-col gap-0.5">
                                      <span class="text-[10px] uppercase tracking-wider text-text-subtle">Version</span>
                                      <span class="text-sm font-mono text-text-primary">{{ ds.version ? 'v' + ds.version : '—' }}</span>
                                    </div>
                                    <div class="flex flex-col gap-0.5">
                                      <span class="text-[10px] uppercase tracking-wider text-text-subtle">Last Scanned</span>
                                      <span class="text-sm font-mono text-text-primary">{{ ds.last_scanned_at ? ((ds.last_scanned_at * 1000) | date:'MMM d, y') : 'Never' }}</span>
                                    </div>
                                    @if (ds.classifier) {
                                      <div class="flex flex-col gap-0.5">
                                        <span class="text-[10px] uppercase tracking-wider text-text-subtle">Class</span>
                                        <span class="text-sm font-medium text-text-primary capitalize">{{ ds.classifier }}</span>
                                      </div>
                                    }
                                  </div>
                                  <div class="flex items-center gap-2 min-w-0 pt-2 border-t border-surface-high/20">
                                    <span class="text-[10px] uppercase tracking-wider text-text-subtle shrink-0">Path</span>
                                    <span class="text-xs font-mono text-text-muted truncate" [title]="ds.path">{{ ds.path }}</span>
                                  </div>
                                </div>
                              </div>
                            }
                          }

                          @if (fieldKey() === 'datasets' && excludedCountFor(dsIdx) > 0) {
                            <div class="col-span-full flex items-start gap-3 mt-1 p-3 rounded-theme-lg bg-warning/5 border border-warning/20">
                              <label class="relative inline-flex items-center cursor-pointer group mt-0.5">
                                <input type="checkbox" formControlName="ignore_filter"
                                       [attr.data-testid]="'config-ignore-filter-' + dsIdx"
                                       class="sr-only peer">
                                <!-- Geometry matches the standard config toggle
                                     (dynamic-form-field); only the checked accent
                                     differs (warning, for the suppressed theme). -->
                                <div class="w-7 h-4 bg-surface-high peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-brand/20 rounded-full peer peer-checked:after:translate-x-3 peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-border-subtle after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-warning group-hover:bg-surface-mid transition-all relative"></div>
                              </label>
                              <div class="flex flex-col gap-0.5">
                                <span class="text-[11px] font-bold text-text-secondary">Use suppressed images this run</span>
                                <span class="text-[10px] text-text-subtle leading-snug">Includes the {{ excludedCountFor(dsIdx) }} excluded image(s) for this run only — does not change the dataset.</span>
                              </div>
                            </div>
                          }
                      </div>
                  </div>
               }
             }

             @if (formArray().length === 0) {
               <div class="p-8 text-center bg-surface-mid/10 border border-surface-mid border-dashed rounded-theme-xl text-text-subtle text-sm"
                    data-testid="config-array-empty">
                   No items added. Click "Add Item" to begin.
               </div>
             }
         </div>
       </div>
    </div>
  `
})
export class DynamicFormGroupComponent {
  fieldKey = input.required<string>();
  schema = input.required<SchemaNode>();
  parentForm = input.required<FormGroup>();
  rootSchema = input<SchemaNode>();

  // Passed down context
  currentBackend = input<string>('local');
  outputDir = input<string>('outputs');
  configHelp = input<Record<string, { tip: string; detail: string }>>({});
  /** Optional: override the auto-fetched dataset names (e.g. for project-scoped filtering) */
  datasetNames = input<string[] | null>(null);

  // External actions
  arrayItemAdded = output<{ key: string, schemaParam: SchemaNode | undefined }>();
  arrayItemRemoved = output<{ key: string, index: number }>();
  helpRequested = output<string>();
  checkpointConfigLoaded = output<TrainingConfig>();

  // Use dataset service locally for dataset_name autocomplete
  private datasetService = inject(DatasetService);
  // Global store, read-only: used to surface the persisted `excluded_count`
  // ("N suppressed" badge) per dataset row. Never written from here.
  private datasetStore = inject(DatasetStore);
  // Media base URL for per-dataset thumbnails (mirrors the dataset library card).
  private rtc = inject(RuntimeConfigService);
  private cdr = inject(ChangeDetectorRef);
  availableDatasets: string[] = [];
  private _allDatasetNames: string[] = [];
  /** name → full Dataset, for the per-row thumbnail / meta / H·C·M pills. */
  private _datasetByName = new Map<string, Dataset>();

  constructor() {
    // Fetch all datasets as the default pool
    this.datasetService.listDatasets().subscribe((ds: Dataset[]) => {
      this._allDatasetNames = ds.map((d: Dataset) => d.name);
      this._datasetByName = new Map(ds.map((d: Dataset) => [d.name, d]));
      // Only use if no external override is active
      if (!this.datasetNames()) {
        this.availableDatasets = this._allDatasetNames;
      }
    });

    // Reactively watch the datasetNames input — when the parent sets it,
    // override the autocomplete list (project-scoped filtering)
    effect(() => {
      const override = this.datasetNames();
      if (override) {
        this.availableDatasets = override;
      } else if (this._allDatasetNames.length > 0) {
        this.availableDatasets = this._allDatasetNames;
      }
    });
  }

  // Gets the exact FormArray from parent
  formArray(): FormArray {
    return this.parentForm().get(this.fieldKey()) as FormArray;
  }

  // Access a nested control within an array object
  getNestedControl(arrayIndex: number, propKey: string): FormControl {
    const row = this.formArray().at(arrayIndex);
    return row.get(propKey) as FormControl;
  }

  isPrimitiveArray(): boolean {
    const itemsSchema = this.resolveSchema(this.schema().items);
    return !!itemsSchema.type && ['string', 'number', 'integer', 'boolean'].includes(itemsSchema.type);
  }

  isNumber(schema: SchemaNode | undefined): boolean {
    if (!schema) return false;
    const resolved = this.resolveSchema(schema);
    return resolved.type === 'number' || resolved.type === 'integer';
  }

  // Resolves nested objects (like dataset settings) into an iterable array of properties
  /**
   * Preferred render order for specific arrays' nested fields. Keys not listed
   * fall to the end, preserving their schema order. Used to group related
   * sample-prompt fields (e.g. Width/Height on the same row).
   */
  private readonly nestedFieldOrder: Record<string, string[]> = {
    sample_prompts: ['prompt', 'seed', 'guidance_scale', 'width', 'height', 'num_inference_steps'],
  };

  nestedProps(): SchemaProp[] {
    const schema = this.schema();
    if (!schema?.items) return [];
    const itemsSchema = this.resolveSchema(schema.items);
    const itemProps = itemsSchema.properties;
    if (itemsSchema.type !== 'object' || !itemProps) return [];

    let keys = Object.keys(itemProps).filter(propKey => propKey !== 'ignore_filter');
    const order = this.nestedFieldOrder[this.fieldKey()];
    if (order) {
      keys = keys.slice().sort((a, b) => {
        const ia = order.indexOf(a); const ib = order.indexOf(b);
        return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
      });
    }

    return keys.map(propKey => ({
      key: propKey,
      schema: this.resolveSchema(itemProps[propKey])
    }));
  }

  resolveSchema(schemaOrRef: SchemaNode | undefined): SchemaNode {
    if (!schemaOrRef) return {};
    const root: SchemaNode = this.rootSchema?.() || {};
    const definitions = root.$defs || root.definitions || {};
    if (schemaOrRef.$ref) {
      const refKey = schemaOrRef.$ref.split('/').pop();
      if (refKey && definitions[refKey]) {
        return { ...definitions[refKey], ...schemaOrRef };
      }
    }
    return schemaOrRef;
  }

  // Array Actions
  addArrayItem() {
    this.arrayItemAdded.emit({ key: this.fieldKey(), schemaParam: this.schema().items });
  }

  removeArrayItem(index: number) {
    this.arrayItemRemoved.emit({ key: this.fieldKey(), index });
  }

  /**
   * Persisted exclusion count for the dataset selected in row `index`,
   * looked up by `dataset_name` against the global store. Returns 0 when the
   * row has no name, the store isn't loaded yet, or the field is absent.
   * Read-only/presentational — never triggers a network load.
   */
  excludedCountFor(index: number): number {
    const name = this.formArray().at(index)?.get('dataset_name')?.value;
    if (!name) return 0;
    const ds = this.datasetStore.entities().find(d => d.name === name);
    return ds?.excluded_count ?? 0;
  }

  /**
   * Full Dataset record for the dataset selected in row `index`, looked up by
   * name. Prefers the locally-fetched list; falls back to the global store.
   * Read-only/presentational (thumbnail, meta row, H·C·M pills).
   */
  datasetFor(index: number): Dataset | undefined {
    const name = this.formArray().at(index)?.get('dataset_name')?.value;
    if (!name) return undefined;
    return this._datasetByName.get(name) ?? this.datasetStore.entities().find(d => d.name === name);
  }

  /** Thumbnail URL for the row's dataset, or null when there's no usable preview. */
  previewUrlFor(index: number): string | null {
    const ds = this.datasetFor(index);
    if (!ds || ds.missing || !ds.preview_image) return null;
    return `${this.rtc.mediaBaseUrl}/${encodeURIComponent(ds.name)}/${ds.preview_image}`;
  }

  /** Total dataset size in MB (1 decimal) for the meta row. */
  sizeMbFor(index: number): string {
    return (((this.datasetFor(index)?.total_size_bytes || 0) / 1048576)).toFixed(1);
  }

  /**
   * H·C·M readiness flags + per-pill coverage tooltips for `<app-state-pills/>`,
   * mirroring the dataset library's `stateOf(d)` so the pills read identically.
   */
  datasetStateFor(index: number): StatePillsState {
    const d = this.datasetFor(index);
    if (!d) return { harmonized: false, captioned: false, masked: false };
    return datasetStatePills({
      total: d.multimedia_count ?? 0,
      captioned: d.caption_count ?? 0,
      masked: d.mask_count ?? 0,
      harmonizationScore: d.harmonization_score ?? 0,
    });
  }

  autofillCaptionPrefix(index: number): void {
    const row = this.formArray().at(index);
    const datasetName = row.get('dataset_name')?.value || '';
    if (!datasetName) return;
    const prefix = datasetName.toLowerCase().replace(/[_-]/g, ' ').trim();
    row.get('caption_prefix')?.setValue(prefix);
  }

  // Group helpers
  isFirstInlineGroupProp(key: string): boolean {
    const props = this.nestedProps();
    const propSchema = props.find(p => p.key === key)?.schema;
    if (!propSchema?.inline_group) return false;
    const groupProps = props.filter(p => p.schema.inline_group === propSchema.inline_group);
    return groupProps.length > 0 && groupProps[0].key === key;
  }

  getInlineGroupProps(groupName: string): SchemaProp[] {
    return this.nestedProps().filter(p => p.schema.inline_group === groupName);
  }

  /**
   * When a boolean nested toggle is switched OFF, clear any boolean toggle that
   * `depends_on` it so dependents don't linger checked-but-disabled. E.g.
   * disabling `masking_enabled` auto-unchecks `recreate_masks`. No-op on turn-on.
   */
  onNestedToggleChange(dsIdx: number, key: string): void {
    const row = this.formArray().at(dsIdx);
    if (!row || row.get(key)?.value) return; // only cascade when turned off
    for (const p of this.nestedProps()) {
      if (p.schema.depends_on === key && p.schema.type === 'boolean') {
        row.get(p.key)?.setValue(false);
      }
    }
  }

  isNestedFieldDisabled(schema: SchemaNode, itemGroup: AbstractControl | null): boolean {
    // depends_on: boolean parent → disable when parent is false
    if (schema.depends_on) {
      const parentVal = itemGroup?.get(schema.depends_on)?.value;
      if (parentVal === false) return true;
    }
    if (!schema.disabled_if) return false;
    for (const [key, value] of Object.entries(schema.disabled_if)) {
      const parentVal = itemGroup?.get(key)?.value;
      if (Array.isArray(value)) {
        if (value.includes(parentVal)) return true;
      } else if (parentVal === value) {
        return true;
      }
    }
    return false;
  }

  shouldHideNestedField(schema: SchemaNode, itemGroup: AbstractControl | null): boolean {
    // depends_on: boolean parent → hide when parent is false
    if (schema.depends_on) {
      const parentVal = itemGroup?.get(schema.depends_on)?.value;
      if (parentVal === false) return true;
    }
    if (!schema.hidden_if) return false;
    for (const [key, value] of Object.entries(schema.hidden_if)) {
      const parentVal = itemGroup?.get(key)?.value;
      if (Array.isArray(value)) {
        if (value.includes(parentVal)) return true;
      } else if (parentVal === value) {
        return true;
      }
    }
    return false;
  }

  isResolutionSelected(res: number): boolean {
    return this.formArray().value.includes(res);
  }

  toggleResolution(res: number): void {
    const array = this.formArray();
    const idx = array.value.indexOf(res);
    if (idx !== -1) {
      array.removeAt(idx);
    } else {
      this.arrayItemAdded.emit({ key: this.fieldKey(), schemaParam: this.schema().items });
      setTimeout(() => {
        array.at(array.length - 1).setValue(res);
        // OnPush: the async setValue isn't tied to a template event or signal,
        // so nudge CD to refresh the preset-selected state and custom-list filter.
        this.cdr.markForCheck();
      });
    }
  }

  isPreset(res: number): boolean {
    return [512, 768, 1024, 1280, 1536].includes(res);
  }

  hasHelp(key: string): boolean {
    return !!this.configHelp()[key];
  }

  getHelpTip(key: string): string {
    return this.configHelp()[key]?.tip || '';
  }

  requestHelp(key: string) {
    this.helpRequested.emit(key);
  }
}
