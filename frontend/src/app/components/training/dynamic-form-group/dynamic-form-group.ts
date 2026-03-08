import { Component, input, output, inject } from '@angular/core';
import { TitleCasePipe } from '@angular/common';
import { ReactiveFormsModule, FormArray, FormGroup, FormControl } from '@angular/forms';
import { DatasetService, Dataset } from '../../../services/dataset';
import { DynamicFormFieldComponent } from '../dynamic-form-field/dynamic-form-field';

@Component({
  selector: 'app-dynamic-form-group',
  standalone: true,
  imports: [TitleCasePipe, ReactiveFormsModule, DynamicFormFieldComponent],
  host: { 'class': 'contents' },
  template: `
    <div class="md:col-span-2 space-y-4">
       <div class="flex items-center justify-between border-b border-surface-mid/50 pb-2">
           <h3 class="text-lg font-semibold text-brand">{{ schema().title || (fieldKey() | titlecase) }}</h3>
           <button type="button" (click)="addArrayItem()" 
              [attr.data-testid]="'config-add-array-item-' + fieldKey()"
              class="bg-brand hover:bg-brand/90 text-white text-xs font-bold py-1 px-3 rounded-full transition-all flex items-center gap-1">
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
              Add {{ fieldKey() === 'datasets' ? 'Dataset' : 'Item' }}
           </button>
       </div>

       <div [formGroup]="parentForm()" class="space-y-4">
         <div [formArrayName]="fieldKey()" class="space-y-4">
             <!-- Case 1: Array of Primitive Types -->
             @if (isPrimitiveArray()) {

               <!-- Specialized Resolutions Array overrides standard primitive array -->
               @if (fieldKey() === 'resolutions') {
                  <div class="space-y-4 p-4 bg-surface-mid/30 border border-surface-mid/50 rounded-theme-xl">
                     <div class="flex items-center justify-between">
                         <h3 class="text-sm font-bold text-text-secondary uppercase tracking-wider">Target Resolutions</h3>
                         <span class="text-[10px] text-text-subtle italic">Must be divisible by 32</span>
                     </div>
                     
                     <!-- Presets -->
                     <div class="grid grid-cols-3 md:grid-cols-5 gap-3">
                         @for (res of [512, 768, 1024, 1280, 1536]; track res) {
                           <button type="button"
                                   (click)="toggleResolution(res)"
                                   [class.bg-brand]="isResolutionSelected(res)"
                                   [class.border-brand]="isResolutionSelected(res)"
                                   [class.bg-surface-mid]="!isResolutionSelected(res)"
                                   [class.text-text-subtle]="!isResolutionSelected(res)"
                                   [attr.data-testid]="'config-res-preset-' + res"
                                   class="py-2 px-3 rounded-theme-lg border border-surface-high/50 text-xs font-bold transition-all hover:border-brand">
                               {{ res }}
                           </button>
                         }
                     </div>

                     <!-- Custom List -->
                     <div class="space-y-2 mt-4">
                         <div class="flex items-center justify-between">
                             <label class="text-xs text-text-muted">Custom Resolutions</label>
                             <button type="button" (click)="addArrayItem()" 
                                     data-testid="config-add-custom-res-btn"
                                     class="text-brand hover:text-brand/80 text-[10px] font-bold uppercase tracking-tight">
                                 + Add Custom
                             </button>
                         </div>
                         
                         <div class="grid grid-cols-2 md:grid-cols-3 gap-2">
                             @for (control of formArray().controls; track $index) {
                                 @if (!isPreset(control.value)) {
                                   <div class="flex items-center gap-1 group animate-in slide-in-from-bottom-2 duration-200">
                                       <input type="number" [formControlName]="$index"
                                              [attr.data-testid]="'config-custom-res-input-' + $index"
                                              class="bg-surface-low border border-surface-mid rounded-theme-md px-2 py-1 text-xs text-white w-full focus:ring-1 focus:ring-brand outline-none"
                                              placeholder="e.g. 1440">
                                       <button type="button" (click)="removeArrayItem($index)" 
                                          [attr.data-testid]="'config-remove-custom-res-btn-' + $index"
                                          class="text-text-disabled hover:text-red-400 p-1 opacity-0 group-hover:opacity-100 transition-all">
                                          <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
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
                       class="relative p-4 bg-surface-mid/30 border border-surface-mid rounded-theme-xl animate-in zoom-in-95 duration-200">
                      
                      <button type="button" (click)="removeArrayItem($index)" 
                         [attr.data-testid]="'config-remove-array-object-' + fieldKey() + '-' + $index"
                         class="absolute top-2 right-2 text-text-subtle hover:text-red-400 transition-colors">
                         <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                      </button>

                      @if (fieldKey() === 'datasets') {
                          <button type="button" (click)="toggleIgnoreFilter($index)"
                              class="absolute top-2 right-8 transition-colors"
                              [class.text-warning]="formArray().at($index).get('ignore_filter')?.value"
                              [class.text-text-disabled]="!formArray().at($index).get('ignore_filter')?.value"
                              [class.hover:text-warning]="true"
                              [title]="formArray().at($index).get('ignore_filter')?.value 
                                  ? 'Ignore Filter ON — All images will be used regardless of exclusions. Click to respect exclusions.' 
                                  : 'Ignore Filter OFF — Only enabled images will be used. Click to use all images.'">
                              @if (formArray().at($index).get('ignore_filter')?.value) {
                                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
                              } @else {
                                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg>
                              }
                          </button>
                      }

                      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-4">

                          @for (nestedProp of nestedProps(); track nestedProp.key) {

                            <!-- Inline group: render all grouped toggles in one cell on first encounter -->
                            @if (nestedProp.schema.inline_group && isFirstInlineGroupProp(nestedProp.key)) {
                              <div class="flex flex-col gap-1.5">
                                <label class="text-[10px] font-bold text-text-subtle uppercase tracking-widest flex items-center gap-1.5">
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
                                               class="sr-only peer">
                                        <div class="w-9 h-5 bg-surface-high/50 border border-surface-mid rounded-full peer peer-focus:ring-2 peer-focus:ring-brand/50 peer-checked:after:translate-x-[16px] after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-brand group-hover:bg-surface-mid transition-all"></div>
                                      </label>
                                      <span class="text-[10px] font-bold text-text-subtle uppercase tracking-widest">{{ ip.schema.title || (ip.key | titlecase) }}</span>
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
                                  
                                  <label class="text-[10px] font-bold text-text-subtle uppercase tracking-widest flex items-center gap-1.5">
                                    @if (nestedProp.key === 'dataset_name') {
                                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="text-brand"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>
                                    }
                                    {{ nestedProp.schema.title || (nestedProp.key | titlecase) }}
                                    @if (hasHelp(nestedProp.key)) {
                                      <span class="config-help-icon" [title]="getHelpTip(nestedProp.key)" (click)="requestHelp(nestedProp.key); $event.preventDefault()">?</span>
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
                                        [datasetAutocomplete]="availableDatasets"
                                        (autofillRequested)="autofillCaptionPrefix(dsIdx)"
                                        (checkpointConfigLoaded)="checkpointConfigLoaded.emit($event)">
                                    </app-dynamic-form-field>
                                  } @else {
                                    <div class="text-xs text-red-500">Missing control: {{ nestedProp.key }}</div>
                                  }
                              </div>
                            }
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
  schema = input.required<any>();
  parentForm = input.required<FormGroup>();
  rootSchema = input<any>();

  // Passed down context
  currentBackend = input<string>('local');
  outputDir = input<string>('outputs');
  configHelp = input<Record<string, { tip: string; detail: string }>>({});

  // External actions
  arrayItemAdded = output<{ key: string, schemaParam: any }>();
  arrayItemRemoved = output<{ key: string, index: number }>();
  helpRequested = output<string>();
  checkpointConfigLoaded = output<any>();

  // Use dataset service locally for dataset_name autocomplete
  private datasetService = inject(DatasetService);
  availableDatasets: string[] = [];

  constructor() {
    this.datasetService.listDatasets().subscribe((ds: Dataset[]) => {
      this.availableDatasets = ds.map((d: Dataset) => d.name);
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
    return itemsSchema && ['string', 'number', 'integer', 'boolean'].includes(itemsSchema.type);
  }

  isNumber(schema: any): boolean {
    if (!schema) return false;
    const resolved = this.resolveSchema(schema);
    return resolved.type === 'number' || resolved.type === 'integer';
  }

  // Resolves nested objects (like dataset settings) into an iterable array of properties
  nestedProps(): any[] {
    const schema = this.schema();
    if (!schema?.items) return [];
    const itemsSchema = this.resolveSchema(schema.items);
    if (itemsSchema.type !== 'object' || !itemsSchema.properties) return [];

    return Object.keys(itemsSchema.properties)
      .filter(propKey => propKey !== 'ignore_filter')
      .map(propKey => ({
        key: propKey,
        schema: this.resolveSchema(itemsSchema.properties[propKey])
      }));
  }

  resolveSchema(schemaOrRef: any): any {
    if (!schemaOrRef) return {};
    const root = this.rootSchema?.() || {};
    const definitions = root.$defs || root.definitions || {};
    if (schemaOrRef && schemaOrRef.$ref) {
      const refKey = schemaOrRef.$ref.split('/').pop();
      if (definitions[refKey]) {
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

  toggleIgnoreFilter(index: number): void {
    const control = this.formArray().at(index).get('ignore_filter');
    if (control) {
      control.setValue(!control.value);
    }
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

  getInlineGroupProps(groupName: string): any[] {
    return this.nestedProps().filter(p => p.schema.inline_group === groupName);
  }

  isNestedFieldDisabled(schema: any, itemGroup: any): boolean {
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

  shouldHideNestedField(schema: any, itemGroup: any): boolean {
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
