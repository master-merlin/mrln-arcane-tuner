import { Component, ChangeDetectionStrategy, output, input, inject, signal, computed, effect, DestroyRef, ViewChild } from '@angular/core';
import { TitleCasePipe } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, FormControl, FormArray, Validators, FormsModule, type AbstractControl, type ValidationErrors } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { DatasetService } from '../../../services/dataset';
import { DatasetStore } from '../../../state/dataset.store';
import { nextTriggerWord } from '../../../shared/trigger-word';
import { ToastService } from '../../../services/toast';
import { SystemService, VRAMReport } from '../../../services/system.service';
import { JobService, type TrainingEstimate, type TrainingConfig } from '../../../services/job';
import { ModelService } from '../../../services/model.service';
import { RegistryStore } from '../../../state/registry.store';

import { Subject } from 'rxjs';
import { debounceTime } from 'rxjs/operators';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { TrainingTemplateSelectorComponent } from '../training-template-selector/training-template-selector';
import { VramBudgetCardComponent } from '../vram-budget-card/vram-budget-card';
import { DynamicFormFieldComponent } from '../dynamic-form-field/dynamic-form-field';
import { DynamicFormGroupComponent } from '../dynamic-form-group/dynamic-form-group';
import { AdvancedVramCardComponent } from '../advanced-vram-card/advanced-vram-card';
import { TargetLayersCardComponent } from '../target-layers-card/target-layers-card';
import { ModelSourceConfigComponent } from '../model-source-config/model-source-config';
import { ModelSourceOverride } from '../../../services/model.service';
import { ModelCapabilitiesService, ModelCapabilities, isFieldHidden } from '../../../services/model-capabilities.service';
import { SchemaNode, SchemaProp, collapseNullableUnion } from '../schema-node';
import type { ModelDefinition } from '../../../screens/training-screen/training-screen';

export interface TrainingTemplate {
  id: string;
  name: string;
  definition_id: string; // Scoped to model definition
  is_default?: boolean;
  config: TrainingConfig;
}

/**
 * Flat descriptor for one config segment, surfaced to the training-screen
 * shell (B3) so it can render a scroll-spy table of contents. Built in DOM
 * order by the `segments` computed.
 */
export interface TrainingSegment {
  id: string;       // segmentId(label)
  label: string;    // display name
  sub: string;      // one-line summary ('' if none)
  status: 'success' | 'warning' | 'idle' | null;
}

@Component({
  selector: 'app-training-dynamic-config',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [TitleCasePipe, ReactiveFormsModule, FormsModule, TrainingTemplateSelectorComponent, VramBudgetCardComponent, AdvancedVramCardComponent, DynamicFormFieldComponent, DynamicFormGroupComponent, TargetLayersCardComponent, ModelSourceConfigComponent],
  template: `
    @if (schema()) {
      <form [formGroup]="form" (ngSubmit)="onSubmit()" class="flex flex-col gap-3.5 isolate">
        

        <!-- Template Selection Child Component -->
        <app-training-template-selector
          [availableModels]="availableModels()"
          [selectedDefinitionId]="selectedDefinition()?.id || null"
          [currentFormConfig]="form.value"
          [projectId]="projectId()"
          (templateApplied)="onTemplateApplied($event)">
        </app-training-template-selector>



        <!-- Model Selection Section (hardcoded) -->
        <section [id]="segmentId('Model Selection')" class="card ts-segment" [style.scrollMarginTop.px]="24">
          <div class="card-head">
              <div class="card-title" style="padding:0">
                <span class="w-[3px] h-3.5 bg-brand rounded-sm shrink-0"></span>
                Model Selection
                @if (segmentSummary('Model Selection')) {
                  <span class="ts-segment-sub">{{ segmentSummary('Model Selection') }}</span>
                }
              </div>

              @if (selectedDefinition(); as model) {
                <div class="flex items-center gap-2 shrink-0">
                  @if (segmentStatus('Model Selection'); as st) {
                    <span class="chip shrink-0"
                          [class.success]="st === 'success'"
                          [class.warning]="st === 'warning'"
                          [class.solid]="st === 'idle'"
                          style="padding:2px 8px">
                      @if (st === 'success') {
                        <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
                        OK
                      } @else if (st === 'warning') {
                        <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                        attn
                      } @else {
                        idle
                      }
                    </span>
                  }
                  @if (model.architecture_params?.['transformer.type'] === 'unified_transformer') {
                    <span class="badge-architecture"
                          title="Pixel-space Unified Transformer — different VRAM characteristics than diffusion models">
                      Unified Transformer
                    </span>
                  }
                  @if (modelSourceOverride(); as src) {
                    @if (src.source_type !== 'hf_hub') {
                      <span class="text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full cursor-default"
                            [title]="src.local_path || ''"
                            [class]="src.source_type === 'local_diffusers'
                              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                              : 'bg-amber-500/10 text-amber-400 border border-amber-500/30'">
                        {{ src.source_type === 'local_diffusers' ? 'LOCAL' : 'SAFETENSORS' }}
                      </span>
                    } @else if (src.skip_update) {
                      <span class="text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/30 cursor-default">
                        OFFLINE
                      </span>
                    }
                  }
                  <div class="text-[10px] font-mono text-text-disabled bg-surface-mid/20 px-3 py-1 rounded-theme-md">
                     ID: <span class="text-brand-light">{{ model.id }}</span>
                  </div>
                  <button type="button" (click)="showSourceConfigModal.set(true); $event.preventDefault()"
                          data-testid="model-source-config-btn"
                          title="Configure model source"
                          class="p-1.5 bg-surface-mid/40 hover:bg-surface-high text-text-subtle hover:text-brand rounded-theme-md border border-surface-mid/50 transition-all">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>
                      <circle cx="12" cy="12" r="3"/>
                    </svg>
                  </button>
                </div>
              }
          </div>

          <div class="card-body">
          <div class="grid grid-cols-2 gap-3.5">
            <!-- Dynamic quantization and model fields from MODEL_SELECTION group -->
            @for (prop of modelSelectionProps(); track prop.key) {
              @if (!shouldHideField(prop.schema, prop.key)) {
                <div [class.md:col-span-2]="isLongInput(prop.key, prop.schema)"
                     [class.opacity-40]="isFieldDisabled(prop.schema)"
                     [class.pointer-events-none]="isFieldDisabled(prop.schema)"
                     class="flex flex-col gap-2 transition-opacity duration-200">
                  <label [for]="prop.key" class="field-label flex items-center gap-1.5">
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
                      <div class="w-7 h-4 bg-surface-high peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-brand/20 rounded-full peer peer-checked:after:translate-x-3 peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-border-subtle after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-brand group-hover:bg-surface-mid transition-all relative"></div>
                      <span class="ml-3 text-sm font-medium text-text-muted group-hover:text-text-secondary">Enable</span>
                    </label>
                  } @else if (isString(prop.schema) && prop.schema.enum) {
                    <select [formControlName]="prop.key"
                            [attr.data-testid]="'config-select-' + prop.key"
                            class="select">
                      @for (opt of getFilteredEnumOptions(prop); track opt.value) {
                        <option [value]="opt.value" [disabled]="opt.disabled">{{ opt.label }}</option>
                      }
                    </select>
                  } @else if (isString(prop.schema)) {
                    <input type="text" [formControlName]="prop.key"
                           [attr.data-testid]="'config-input-' + prop.key"
                           class="input">
                  }

                  @if (prop.schema.description) {
                    <p class="text-[10.5px] text-text-muted">{{ prop.schema.description }}</p>
                  }
                </div>
              }
            }
          </div>
          </div>
        </section>

        <!-- VRAM Budget — Advanced block-swapping is projected into the same card -->
        <section [id]="segmentId('VRAM Budget')" class="ts-segment" [style.scrollMarginTop.px]="24">
          <app-vram-budget-card [report]="vramReport()">
            <app-advanced-vram-card
              [definitionId]="currentDefinitionId()"
              (blockSwapChanged)="onBlockSwapChanged($event)">
            </app-advanced-vram-card>
          </app-vram-budget-card>
        </section>

        <div class="flex flex-col gap-3.5">
          @for (group of groups(); track group.name) {
            @if (!isGroupHidden(group)) {
            <section [id]="segmentId(group.name)" class="card ts-segment" [style.scrollMarginTop.px]="24">
              <!-- Group Header -->
              <div class="card-head cursor-pointer select-none"
                    (click)="toggleGroup(group.name)">
                   <div class="card-title" style="padding:0">
                     <span class="w-[3px] h-3.5 bg-brand rounded-sm shrink-0"></span>
                     {{ formatGroupName(group.name) }}
                     @if (segmentSummary(group.name)) {
                       <span class="ts-segment-sub">{{ segmentSummary(group.name) }}</span>
                     }
                   </div>
                   <div class="flex items-center gap-2 shrink-0">
                     @if (segmentStatus(group.name); as st) {
                       <span class="chip shrink-0"
                             [class.success]="st === 'success'"
                             [class.warning]="st === 'warning'"
                             [class.solid]="st === 'idle'"
                             style="padding:2px 8px">
                         @if (st === 'success') {
                           <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
                           OK
                         } @else if (st === 'warning') {
                           <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                           attn
                         } @else {
                           idle
                         }
                       </span>
                     }
                     <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                          class="text-text-disabled transition-transform duration-200" [class.rotate-180]="!isGroupCollapsed(group.name)">
                       <path d="m6 9 6 6 6-6"/>
                     </svg>
                   </div>
               </div>

              @if (!isGroupCollapsed(group.name)) {
              <div class="card-body">

              <!-- ═══════════ Custom LoRA Naming Section (General Settings only) ═══════════ -->
              @if (group.name === 'General Settings' && form.get('lora_prefix')) {
              <div class="space-y-3.5 mb-3.5">
                <div class="flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-brand-light">
                    <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
                    <polyline points="14 2 14 8 20 8"/>
                  </svg>
                  <span class="text-xs font-bold text-text-subtle uppercase tracking-widest">LoRA Naming</span>
                </div>

                <!-- Prefix + Suffix row -->
                <div class="grid grid-cols-2 gap-3.5">
                  <!-- Prefix -->
                  <div class="flex flex-col gap-1.5">
                    <label class="field-label flex items-center gap-1.5">
                      LoRA Prefix
                      @if (hasHelp('lora_prefix')) {
                        <span class="config-help-icon" [title]="getHelpTip('lora_prefix')" (click)="openHelpModal('lora_prefix'); $event.preventDefault()">?</span>
                      }
                    </label>
                    <div class="flex gap-2">
                      <input type="text" formControlName="lora_prefix"
                             data-testid="config-input-lora_prefix"
                             placeholder="e.g. MyDataset"
                             class="input flex-1">
                      <button type="button" (click)="autofillLoraField('lora_prefix')" title="Auto-derive from dataset name"
                              data-testid="lora-prefix-wand"
                              class="p-2 bg-surface-mid hover:bg-brand/20 border border-surface-high hover:border-brand/40 rounded-theme-md text-text-muted hover:text-brand transition-all">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                          <path d="m15 4-1 2-2 1 2 1 1 2 1-2 2-1-2-1-1-2Z"/>
                          <path d="m8 11-1.5 3L3 15.5l3.5 1.5L8 20l1.5-3 3-1.5-3-1.5L8 11Z"/>
                        </svg>
                      </button>
                    </div>
                  </div>

                  <!-- Suffix -->
                  <div class="flex flex-col gap-1.5">
                    <label class="field-label flex items-center gap-1.5">
                      LoRA Suffix
                      @if (hasHelp('lora_suffix')) {
                        <span class="config-help-icon" [title]="getHelpTip('lora_suffix')" (click)="openHelpModal('lora_suffix'); $event.preventDefault()">?</span>
                      }
                    </label>
                    <div class="flex gap-2">
                      <input type="text" formControlName="lora_suffix"
                             data-testid="config-input-lora_suffix"
                             placeholder="e.g. v1"
                             class="input flex-1">
                      <button type="button" (click)="autofillLoraField('lora_suffix')" title="Auto-derive from dataset name"
                              data-testid="lora-suffix-wand"
                              class="p-2 bg-surface-mid hover:bg-brand/20 border border-surface-high hover:border-brand/40 rounded-theme-md text-text-muted hover:text-brand transition-all">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                          <path d="m15 4-1 2-2 1 2 1 1 2 1-2 2-1-2-1-1-2Z"/>
                          <path d="m8 11-1.5 3L3 15.5l3.5 1.5L8 20l1.5-3 3-1.5-3-1.5L8 11Z"/>
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>

                <!-- LoRA Name + Trigger Word row -->
                <div class="grid grid-cols-2 gap-3.5">
                  <!-- LoRA Name -->
                  <div class="flex flex-col gap-1.5">
                    <label class="field-label flex items-center gap-1.5">
                      LoRA Name
                      @if (hasHelp('lora_name')) {
                        <span class="config-help-icon" [title]="getHelpTip('lora_name')" (click)="openHelpModal('lora_name'); $event.preventDefault()">?</span>
                      }
                      <span class="text-[10px] font-mono text-text-disabled bg-surface-mid/40 px-1.5 py-0.5 rounded">supports placeholders</span>
                    </label>
                    <input type="text" formControlName="lora_name"
                           data-testid="config-input-lora_name"
                           placeholder="e.g. prefix_flux_suffix"
                           class="input w-full font-mono">
                    <!-- Live Preview -->
                    @if (loraNamePreview() && loraNamePreview() !== form.get('lora_name')?.value) {
                      <div class="flex items-center gap-2 mt-0.5">
                        <span class="text-[10px] text-text-disabled uppercase tracking-wider">Preview:</span>
                        <span class="text-xs text-brand-light font-mono">{{ loraNamePreview() }}.safetensors</span>
                      </div>
                    }
                  </div>

                  <!-- Trigger Word -->
                  <div class="flex flex-col gap-1.5">
                    <label class="field-label flex items-center gap-1.5">
                      Trigger Word
                      @if (hasHelp('global_triggerword')) {
                        <span class="config-help-icon" [title]="getHelpTip('global_triggerword')" (click)="openHelpModal('global_triggerword'); $event.preventDefault()">?</span>
                      }
                    </label>
                    <div class="flex gap-2">
                      <input type="text" formControlName="global_triggerword"
                             data-testid="config-input-global_triggerword"
                             placeholder="e.g. ohwx"
                             class="input flex-1 font-mono">
                      <button type="button" (click)="autofillTriggerWord()" title="Use the dataset's trigger word, or generate one"
                              data-testid="trigger-word-wand"
                              class="p-2 bg-surface-mid hover:bg-brand/20 border border-surface-high hover:border-brand/40 rounded-theme-md text-text-muted hover:text-brand transition-all">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                          <path d="m15 4-1 2-2 1 2 1 1 2 1-2 2-1-2-1-1-2Z"/>
                          <path d="m8 11-1.5 3L3 15.5l3.5 1.5L8 20l1.5-3 3-1.5-3-1.5L8 11Z"/>
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
              }

              <div [class]="fieldGridClass(group.name)">

                  @for (prop of group.props; track prop.key) {

                     <!-- Inline group: render all grouped toggles in one full-width row on first encounter -->
                     @if (prop.schema.inline_group && isFirstInlineGroupProp(prop.key, group.props)) {
                       <div class="md:col-span-2 flex flex-col gap-2">
                         <label class="text-xs font-bold text-text-subtle uppercase tracking-widest flex items-center gap-1.5 mb-0.5">
                           {{ prop.schema.inline_group === 'masking_toggles' ? 'Enable masking' : (prop.schema.inline_group.replace('_', ' ') | titlecase) }}
                         </label>
                         <div class="grid grid-cols-3 gap-x-6 gap-y-2">
                           @for (ip of getInlineGroupProps(prop.schema.inline_group, group.props); track ip.key) {
                             <div class="flex flex-col gap-1">
                               <div class="flex items-center gap-2.5">
                                 <label class="relative inline-flex items-center cursor-pointer group shrink-0">
                                   <input type="checkbox" [formControlName]="ip.key"
                                          [attr.data-testid]="'config-checkbox-' + ip.key"
                                          class="sr-only peer">
                                   <div class="w-7 h-4 bg-surface-high/50 border border-surface-mid rounded-full peer peer-focus:ring-2 peer-focus:ring-brand/50 peer-checked:after:translate-x-3 after:content-[''] after:absolute after:top-[1px] after:left-[2px] after:bg-white after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-brand group-hover:bg-surface-mid transition-all relative"></div>
                                 </label>
                                 <span class="text-[11px] font-medium text-text-secondary flex items-center gap-1.5">
                                   {{ ip.schema.title || (ip.key | titlecase) }}
                                   @if (hasHelp(ip.key)) {
                                     <span class="config-help-icon" [title]="getHelpTip(ip.key)" (click)="openHelpModal(ip.key); $event.preventDefault()">?</span>
                                   }
                                 </span>
                               </div>
                               @if (ip.schema.description) {
                                 <p class="text-[10px] text-text-subtle italic leading-snug">{{ ip.schema.description }}</p>
                               }
                             </div>
                           }
                         </div>
                       </div>
                     }

                     <!-- Normal Grouping for non-array types (skip block_swap_sliders — rendered near VRAM card) -->
                     @if (prop.schema.type !== 'array' && !shouldHideField(prop.schema, prop.key) && !prop.schema.inline_group && prop.schema.ui_type !== 'block_swap_sliders' && !loraCustomKeys.has(prop.key)) {
                         <app-dynamic-form-field
                            [style.grid-column]="isFullWidthField(prop.key, group.name) ? '1 / -1' : null"
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
                 @if (prop.schema.type === 'array' && (prop.schema.ui_type === 'layer_checklist' || !shouldHideField(prop.schema, prop.key))) {
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
                            [isVideoModel]="isVideoModel()"
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
              </div>
              }


            </section>
            }
          }
        </div>

        <button type="submit"
          [disabled]="!form.valid"
          data-testid="submit-config-btn"
          class="btn primary self-start mt-1">
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

      @if (showSourceConfigModal() && currentDefinitionId()) {
        <app-model-source-config
          [definitionId]="currentDefinitionId()"
          [definitionName]="selectedDefinition()?.name || ''"
          [initialBrowsePath]="defaultModelPath()"
          (close)="showSourceConfigModal.set(false)"
          (saved)="onSourceOverrideSaved($event)">
        </app-model-source-config>
      }
    }
  `,
  styleUrl: 'training-dynamic-config.css'
})
export class TrainingDynamicConfigComponent {
  schema = input<SchemaNode>(); // JSON Schema
  availableModels = input<ModelDefinition[]>([]); // New input for model list
  projectId = input<string | null>(null); // Passed down to components
  configSubmitted = output<TrainingConfig>();

  private fb = inject(FormBuilder);
  private http = inject(HttpClient);
  private toast = inject(ToastService);
  private systemService = inject(SystemService);
  private jobService = inject(JobService);
  private modelService = inject(ModelService);
  private registryStore = inject(RegistryStore);
  private modelCapabilitiesService = inject(ModelCapabilitiesService);

  /**
   * Capability descriptor for the currently-selected model definition.
   * `null` => no descriptor (everything visible). Read inside shouldHideField()
   * so template re-evaluation is tracked on OnPush when capabilities load.
   */
  protected capabilities = signal<ModelCapabilities | null>(null);

  /** True when the selected model is a video model — reuses the run-level
   *  `num_frames` capability gate (is_video). Drives `video_only` per-dataset
   *  fields in the dataset config. Fail-open (no descriptor → visible). */
  protected isVideoModel = computed(() => !isFieldHidden(this.capabilities(), 'num_frames'));

  // VRAM estimation
  vramReport = signal<VRAMReport | null>(null);
  // Full data-calibrated estimate (wall time, throughput, output, disk + VRAM).
  estimate = signal<TrainingEstimate | null>(null);
  private vramEstimate$ = new Subject<void>();
  Math = Math; // expose to template

  // Model source override
  showSourceConfigModal = signal(false);
  modelSourceOverride = signal<ModelSourceOverride | null>(null);
  defaultModelPath = signal('');

  form: FormGroup = new FormGroup({});
  properties = signal<SchemaProp[]>([]);
  groups = signal<{ name: string, props: SchemaProp[] }[]>([]);
  modelSelectionProps = signal<SchemaProp[]>([]);
  nestedItemPropsMap = signal<Record<string, SchemaProp[]>>({});

  // Collapsible groups — ENGINE starts collapsed
  collapsedGroups = signal<Set<string>>(new Set(['Advanced Engine', 'Sampling', 'Expert Features']));

  // Dataset autocomplete
  availableDatasets = signal<string[]>([]);

  // LoRA Naming — keys handled by the custom section (skipped from generic @for).
  // global_triggerword is rendered next to LoRA Name (with its own magic wand),
  // so it is excluded from the generic field loop too.
  readonly loraCustomKeys = new Set(['lora_prefix', 'lora_suffix', 'lora_name', 'global_triggerword']);

  /** Live preview of the resolved LoRA filename. */
  loraNamePreview = signal<string>('');

  /** Derive a clean identifier from a dataset name (replace dashes/spaces with underscores, preserve case). */
  cleanDatasetName(): string {
    const dsArray = this.getFormArray('datasets');
    if (!dsArray || dsArray.length === 0) return '';
    const raw = dsArray.at(0)?.get('dataset_name')?.value || '';
    return raw.replace(/[-\s]+/g, '_');
  }

  /** Magic-wand click: auto-fill a lora naming field from the first dataset name. */
  autofillLoraField(fieldKey: string): void {
    const cleaned = this.cleanDatasetName();
    if (!cleaned) {
      this.toast.warning('No dataset configured yet');
      return;
    }
    this.form.get(fieldKey)?.setValue(cleaned);
  }

  // Trigger-word wand cycle state (mirrors the dataset-form modal).
  private nextTriggerStrategy = 0;
  private lastGeneratedTrigger = '';

  /** Raw name of the first configured dataset, untrimmed/uncleaned. */
  private firstDatasetName(): string {
    const dsArray = this.getFormArray('datasets');
    return dsArray && dsArray.length > 0 ? (dsArray.at(0)?.get('dataset_name')?.value || '') : '';
  }

  /**
   * Magic-wand click for the global trigger word:
   *   (a) prefer the configured dataset's stored `trigger_word` (first click on
   *       an empty field), then
   *   (b) fall back to generating one from the dataset name, cycling through the
   *       shared trigger strategies on repeat clicks for alternatives.
   */
  autofillTriggerWord(): void {
    const ctrl = this.form.get('global_triggerword');
    if (!ctrl) return;

    const dsName = this.firstDatasetName();
    const current: string = ctrl.value || '';

    // (a) Use the dataset's own trigger word, if any, on a fresh (empty) field.
    if (!current && dsName) {
      const stored = this.datasetStore.entities().find(d => d.name === dsName)?.trigger_word || '';
      if (stored) {
        ctrl.setValue(stored);
        this.lastGeneratedTrigger = '';
        this.nextTriggerStrategy = 0;
        return;
      }
    }

    // (b) Generate from the dataset name, cycling on repeat clicks.
    const raw = dsName || this.cleanDatasetName();
    if (!raw) {
      this.toast.warning('No dataset configured yet');
      return;
    }
    const continuing = current !== '' && current === this.lastGeneratedTrigger;
    const result = nextTriggerWord(raw, current, continuing ? this.nextTriggerStrategy : 0);
    if (!result) return;
    ctrl.setValue(result.trigger);
    this.lastGeneratedTrigger = result.trigger;
    this.nextTriggerStrategy = result.nextIndex;
  }

  /**
   * CSS class for a group's field grid. General Settings and Training Dynamics
   * use a 2-column grid (matching Model Selection) so their values line up as
   * two equal columns; every other group keeps the responsive 1/2/3 layout.
   */
  private readonly twoColGroups = new Set(['General Settings', 'Training Dynamics', 'Sampling', 'Advanced Engine']);
  fieldGridClass(groupName: string): string {
    const motion = 'gap-3.5 animate-in fade-in slide-in-from-top-2 duration-200';
    return this.twoColGroups.has(groupName)
      ? `grid grid-cols-2 ${motion}`
      : `grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 ${motion}`;
  }

  /** True for fields that should span the full grid width (e.g. Output folder). */
  isFullWidthField(key: string, groupName: string): boolean {
    return groupName === 'General Settings' && key === 'output_dir';
  }

  /** Resolve {placeholder} tokens in a raw lora_name string using current form values. */
  resolveLoraName(raw: string): string {
    const formValues = this.form.getRawValue();
    return raw.replace(/\{(\w+)\}/g, (_: string, key: string) => {
      const val = formValues[key];
      return val !== undefined && val !== null && val !== '' ? String(val) : `{${key}}`;
    });
  }

  /** Refresh the loraNamePreview signal from current form state. */
  private _updateLoraNamePreview(): void {
    const raw = this.form.get('lora_name')?.value || '';
    this.loraNamePreview.set(this.resolveLoraName(raw));
  }

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
  // Set once a Jobs-screen handoff (Reload / Save template) has been applied,
  // so the selector's one-time `auto` template apply yields to it.
  private _externalConfigApplied = false;
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
  private datasetStore = inject(DatasetStore);
  private destroyRef = inject(DestroyRef);

  // Load source override whenever definition changes
  private _sourceOverrideEffect = effect(() => {
    const defId = this.currentDefinitionId();
    if (defId) {
      this.loadSourceOverride(defId);
    } else {
      this.modelSourceOverride.set(null);
    }
  });

  constructor() {
    this.http.get<Record<string, { tip: string; detail: string }>>('/config_help.json')
      .subscribe(data => this.configHelp.set(data));

    // Load global default model path for browse dialog
    this.modelService.getGlobalSettings().subscribe({
      next: (s) => this.defaultModelPath.set(s.default_model_path || ''),
      error: () => {},
    });

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
          this.selectedFamily.set(model.family ?? '');
        } else if (this.availableModels().length > 0) {
          const firstModel = this.availableModels()[0];
          this.selectedFamily.set(firstModel.family ?? '');
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

  /**
   * Fetch the capability descriptor for a definition and store it in the
   * `capabilities` signal. On any failure (or empty id) the signal is cleared
   * to null => every field stays visible (fail-open, never crash).
   * Read-only/additive: does not mutate the form or interfere with the
   * model-change-modal / targeted-layers reset flow.
   */
  private _loadFieldCapabilities(definitionId: string): void {
    if (!definitionId) {
      this.capabilities.set(null);
      return;
    }
    // Snapshot the template-apply guard at INVOCATION time. The capabilities
    // fetch may resolve asynchronously (HTTP) and outlive the 1500ms window in
    // which `_isTemplateApplying` is true. Capturing here ensures a definition
    // change driven by an applied template never clobbers the template's values
    // even if the HTTP response lands after the guard has been released.
    const applyingTemplate = this._isTemplateApplying;
    this.modelCapabilitiesService.getCapabilities(definitionId).subscribe({
      next: (caps) => {
        this.capabilities.set(caps);
        // Fill pristine controls from the definition's per-definition defaults.
        // Skipped entirely when this load was triggered as part of applying a
        // template (the template's values must win).
        if (!applyingTemplate && caps?.defaults) {
          this.applyDefinitionDefaults(caps.defaults);
        }
      },
      error: () => this.capabilities.set(null),
    });
  }

  /**
   * Resolve the schema (JSON-Schema) default for a top-level form key, or
   * `undefined` if the key/schema is unknown. Used to decide whether a control
   * still holds its schema default (safe to overwrite with a definition default)
   * vs. a value the user/template set (must be preserved).
   */
  private _schemaDefaultFor(key: string): unknown {
    const props = this.schema()?.properties;
    if (!props || !props[key]) return undefined;
    return this.resolveSchema(props[key])?.default;
  }

  /**
   * Apply per-definition `defaults` to the form WITHOUT overriding template or
   * user values. Precedence (highest→lowest): user edits / applied template >
   * definition defaults > schema defaults.
   *
   * A control is patched ONLY when ALL of:
   *  (a) we are NOT currently applying a template (template values win — early
   *      return guards the entire operation);
   *  (b) the key maps to a real, non-array/non-group form control (backend key);
   *  (c) the control is pristine (`!control.dirty`) AND its current value still
   *      equals the field's schema default — or, when no schema default exists,
   *      the current value is empty (null / undefined / ''). This conservative
   *      rule errs toward NOT overwriting: any value that differs from the
   *      schema default (a template value, or a user edit) is left untouched.
   *
   * Patches use `{ emitEvent: false }` and never mark controls dirty (defaults
   * are not user edits). The whole batch is wrapped in the child's
   * `suppressAutoSave` guard so it cannot spawn a phantom auto-saved template,
   * and the VRAM estimate is refreshed once afterward via the debounced subject.
   */
  applyDefinitionDefaults(defaults: Record<string, unknown>): void {
    // (a) Template application owns the form right now — its values win.
    if (this._isTemplateApplying) return;
    if (!defaults) return;

    // Suppress auto-save so filling defaults doesn't create/overwrite a template.
    const selector = this.templateSelector;
    const hadSuppress = selector ? selector.suppressAutoSave() : false;
    if (selector) selector.suppressAutoSave.set(true);

    let patchedAny = false;
    try {
      for (const key of Object.keys(defaults)) {
        // (b) Must be a real, primitive backend control. Arrays/groups are
        // handled by their own machinery and are out of scope for defaults.
        const control = this.form.get(key);
        if (!control || control instanceof FormArray || control instanceof FormGroup) continue;

        // (c) Only fill pristine controls that still hold the schema default
        // (or are empty when no schema default is defined). Never overwrite a
        // value that differs — that came from the user or an applied template.
        if (control.dirty) continue;
        const schemaDefault = this._schemaDefaultFor(key);
        const current = control.value;
        const stillSchemaDefault =
          schemaDefault !== undefined
            ? current === schemaDefault
            : current === null || current === undefined || current === '';
        if (!stillSchemaDefault) continue;

        const next = defaults[key];
        if (next === current) continue; // nothing to change

        control.setValue(next, { emitEvent: false });
        patchedAny = true;
      }
    } finally {
      if (selector) selector.suppressAutoSave.set(hadSuppress);
    }

    // Refresh the VRAM estimate / preview once, without re-enabling auto-save
    // (the debounced subject only triggers refreshVRAMEstimate, never a save).
    if (patchedAny) {
      this._updateLoraNamePreview();
      this.vramEstimate$.next();
    }
  }

  onFamilyChange(event: Event) {
    const family = (event.target as HTMLSelectElement).value;
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
    // One call to the full estimator: it returns the calibrated VRAM report
    // (feeding the in-form budget card + the shell's VRAM detail rail) PLUS
    // wall time / throughput / output / disk for the shared estimate wall.
    this.jobService.estimate(defId, config).subscribe({
      next: (est) => {
        this.estimate.set(est);
        this.vramReport.set(est?.vram ?? null);
      },
      error: (err) => {
        console.warn('[Estimate] Estimation failed', err);
        this.estimate.set(null);
        this.vramReport.set(null);
      }
    });
  }

  /** Force a re-estimate (e.g. after the shell backfills calibration stats). */
  refreshEstimate(): void {
    this.vramEstimate$.next();
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
    // Refresh field-visibility descriptor for the newly-applied model
    this._loadFieldCapabilities(newDefId);
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

  onTemplateApplied(event: { config: TrainingConfig, isDefault: boolean, definitionId?: string, auto?: boolean }) {
    // The selector fires a one-time `auto` apply on load so the estimate wall
    // reflects the active template. A Jobs-screen handoff (Reload / Save
    // template) takes precedence — if one already landed, ignore the auto apply
    // so it can't clobber the handed-off config (either ordering is safe: a
    // handoff after this still wins, as it isn't gated).
    if (event.auto && this._externalConfigApplied) return;
    this._isTemplateApplying = true;
    try {
      if (event.isDefault) {
        this.resetFormToDefaults();
      } else {
        if (event.definitionId) {
          const model = this.availableModels().find(m => m.id === event.definitionId);
          if (model) {
            this.selectedFamily.set(model.family ?? '');
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
    } finally {
      // ALWAYS release the child's auto-save suppression, even if the
      // reset/patch above throws — a latched suppressAutoSave silently
      // disables every future auto-save (no template would ever be created).
      setTimeout(() => {
        this._isTemplateApplying = false;
        if (this.templateSelector) {
          this.templateSelector.suppressAutoSave.set(false);
        }
      }, 1500);
    }
  }

  resetFormToDefaults() {
    const schema = this.schema();
    const props = schema?.properties || {};

    Object.keys(this.form.controls).forEach(key => {
      // Default = "default settings for the CURRENT model": the virtual
      // Default selector entry carries the current definition id, so the
      // model selection must survive the reset. Resetting model_family to
      // its schema default silently switched the family, which reloads the
      // schema and rebuilds the whole form mid-flight.
      if (key === 'definition_id' || key === 'model_family') return;

      const propSchema = this.resolveSchema(props[key]);
      const control = this.form.get(key);

      if (propSchema.type === 'array') {
        // Schema-`array` props are not always FormArrays: layer_checklist
        // (targeted_layers) is built as a flat FormControl<string[]> (see
        // buildForm). Calling .clear() on it threw mid-loop, aborting the
        // reset AND leaking the selector's suppressAutoSave=true forever.
        if (!(control instanceof FormArray)) {
          control?.setValue(propSchema.default ?? []);
          return;
        }
        const formArray = control;
        formArray.clear();
        const defaults = Array.isArray(propSchema.default) ? propSchema.default : [];
        defaults.forEach(() => {
          this.addArrayItem(key, propSchema.items);
        });
        if (this.isPrimitiveArray(key)) {
          formArray.patchValue(defaults);
        } else {
          // Object arrays with a minimum row count (`datasets`, minItems: 1)
          // are seeded back to that minimum with blank rows — mirroring
          // buildForm()'s initial auto-add. The schema declares no default for
          // them, so a bare clear left ZERO rows and the Datasets card
          // collapsed to "No items added" on the next change-detection pass.
          const minRows = propSchema.minItems ?? (key === 'datasets' ? 1 : 0);
          while (formArray.length < minRows) {
            this.addArrayItem(key, propSchema.items);
          }
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
            } else if (control.length > 0) {
              // Schema lookup can return null mid-load (properties() not yet
              // populated). Clone an existing row's control structure rather
              // than `return`-ing, which would SKIP this key and leave the
              // array at its pre-seeded default (the bug where a saved
              // `datasets` selection reverted to the first dataset on reload).
              control.push(this.cloneArrayRow(control.at(0)));
            } else {
              // Empty array AND no resolvable schema (worst-case mid-load). For
              // a PRIMITIVE array (e.g. `resolutions`) we can still grow it with
              // plain controls seeded from the target values, so the saved list
              // loads instead of collapsing to its first element ("only 1024
              // selected"). Object rows need a structure we can't synthesise
              // without a schema or an existing row, so those stop here.
              const sample = Array.isArray(value) ? value[control.length] : undefined;
              if (sample !== null && typeof sample === 'object') break;
              control.push(new FormControl(sample ?? null));
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

  /** Build a new array row mirroring an existing row's control structure (keys
   *  + nesting), seeded with that row's current values. Used as a fallback when
   *  growing a FormArray during a patch and the item schema isn't resolvable
   *  yet — `patchValue` then overwrites the seeded values with the real config. */
  private cloneArrayRow(template: AbstractControl): AbstractControl {
    if (template instanceof FormGroup) {
      const group: Record<string, AbstractControl> = {};
      for (const k of Object.keys(template.controls)) {
        group[k] = this.cloneArrayRow(template.get(k)!);
      }
      return this.fb.group(group);
    }
    if (template instanceof FormArray) {
      return this.fb.array(template.controls.map(c => this.cloneArrayRow(c)));
    }
    return new FormControl(template.value);
  }

  getControl(key: string): FormControl {
    return this.form.get(key) as FormControl;
  }

  getArrayStringControl(key: string): FormControl<string[]> {
    return this.form.get(key) as FormControl<string[]>;
  }

  getSchemaForKey(key: string, parent?: unknown): SchemaNode | null {
    const prop = this.properties().find(p => p.key === key);
    if (prop) return prop.schema;
    return null;
  }

  // --- External Config Import (from Job Queue) ---

  importTemplate(name: string, config: TrainingConfig, definitionId: string) {
    this._externalConfigApplied = true;
    if (this.templateSelector) {
      this.templateSelector.importExternalTemplate(name, config, definitionId);
    }

    // Switch family to match the template's definition
    const model = this.availableModels().find(m => m.id === definitionId);
    if (model) {
      this.selectedFamily.set(model.family ?? '');
    }
    this.patchFormRecursive(this.form, config);
  }

  /**
   * Edit an EXISTING template in place (Projects "Edit" → Training, Bug A) or
   * reload a job onto its source template (Bug B). Unlike importTemplate, this
   * never clones a copy: the selector adopts `templateId` as the save-target
   * (recreating it only if it was deleted), and the form is patched with the
   * handed-off `config` (the job's actual run values for a reload). Mirrors
   * loadExternalConfig's auto-save suppression so the patch itself doesn't
   * immediately rewrite the template.
   */
  applyExistingTemplate(templateId: string, name: string, config: TrainingConfig, definitionId: string) {
    this._externalConfigApplied = true;
    this._isTemplateApplying = true;
    if (this.templateSelector) {
      this.templateSelector.suppressAutoSave.set(true);
      this.templateSelector.adoptExternalTemplate(templateId, name, config, definitionId);
    }

    const resumePath = this.form.get('resume_from_checkpoint')?.value;

    const model = this.availableModels().find(m => m.id === definitionId);
    if (model) {
      this.selectedFamily.set(model.family ?? '');
    }
    this.patchFormRecursive(this.form, config);

    if (resumePath) {
      this.form.get('resume_from_checkpoint')?.setValue(resumePath);
    }

    this._syncBlockSwapFromForm();
    this.targetLayersCard?.refreshFromControl();

    setTimeout(() => {
      this._isTemplateApplying = false;
      if (this.templateSelector) {
        this.templateSelector.suppressAutoSave.set(false);
      }
    }, 1500);
  }

  loadExternalConfig(config: TrainingConfig) {
    // Suppress auto-save so patching the form doesn't create a new template
    this._externalConfigApplied = true;
    this._isTemplateApplying = true;
    if (this.templateSelector) {
      this.templateSelector.suppressAutoSave.set(true);
    }

    // Preserve session-specific fields that should not be overwritten by templates
    const resumePath = this.form.get('resume_from_checkpoint')?.value;

    // Switch family + definition to match the config being loaded
    const defId = config['definition_id'];
    if (defId) {
      const model = this.availableModels().find(m => m.id === defId);
      if (model) {
        this.selectedFamily.set(model.family ?? '');
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

    // Release auto-save suppression after debounced valueChanges settle
    setTimeout(() => {
      this._isTemplateApplying = false;
      if (this.templateSelector) {
        this.templateSelector.suppressAutoSave.set(false);
      }
    }, 1500);
  }
  buildForm() {
    const schema = this.schema();
    const props = schema?.properties || {};
    const group: any = {};
    const properties: SchemaProp[] = [];
    const nestedItemPropsMap: Record<string, SchemaProp[]> = {};

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
              const defaults = Array.isArray(propSchema.default) ? propSchema.default : [];
              defaults.forEach((d) => {
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
          if (schema?.required?.includes(key)) {
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

            // When depends_on silently updates definition_id, sync the signal
            // so the header badge + source config modal reflect the new model
            if (prop.key === 'definition_id') {
              const newVal = firstValid ? firstValid.value : '';
              this.currentDefinitionId.set(newVal);
              this._lastKnownDefinitionId = newVal;
            }
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

    // Seed capability descriptor for the initial definition so family-unsupported
    // fields are hidden on first render (valueChanges below only fires on change).
    this._loadFieldCapabilities(this.form.get('definition_id')?.value || '');

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
      // Fetch the capability descriptor to hide family-unsupported fields.
      // Read-only/additive: it does not touch the model-change-modal revert
      // logic above (which uses emitEvent:false and therefore never reaches here).
      this._loadFieldCapabilities(val);
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
      this._updateLoraNamePreview();
      // Bump the form-version signal so the `segments` computed (and the
      // segmentsChanged output) recompute on form value changes — signals do
      // not track FormGroup mutations on their own.
      this.formVersion.update(v => v + 1);
    });

    // Initial preview render
    this._updateLoraNamePreview();
  }

  organizeGroups(properties: SchemaProp[]) {
    const groupMap: Record<string, SchemaProp[]> = {};
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

    // Extract MODEL_SELECTION props for the hardcoded section, ordered into
    // three 2-column rows: (1) family · definition, (2) quant backends
    // (model · TE), (3) quantization (model · TE). Unknown keys fall to the end.
    const modelSelectionFieldOrder = [
      'model_family', 'definition_id',
      'quantization_backend', 'te_quantization_backend',
      'quantization', 'te_quantization',
    ];
    const modelSelectionProps = (groupMap['MODEL_SELECTION'] || []).slice().sort((a, b) => {
      const ia = modelSelectionFieldOrder.indexOf(a.key);
      const ib = modelSelectionFieldOrder.indexOf(b.key);
      return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
    });
    this.modelSelectionProps.set(modelSelectionProps);

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

  // ── Segment anchoring (B1) ──────────────────────────────────────────
  /**
   * Stable DOM id for a config segment, keyed by the group's *display* name
   * (as produced by formatGroupName). Used by the section anchors so the
   * shell TOC (B3) can scroll-spy / jump. Falls back to a slug for any
   * unmapped group so new groups still get a usable id.
   */
  segmentId(groupName: string): string {
    const map: Record<string, string> = {
      'Model Selection': 'model',
      'VRAM Budget': 'vram',
      'Concepts & Triggerwords': 'datasets',
      'General Settings': 'general',
      'Training Dynamics': 'dynamics',
      'LoRA Parameters': 'lora',
      'Optimizer Settings': 'optim',
      'Expert Features': 'expert',
      'Sampling': 'sampling',
      'Advanced Engine': 'advanced',
    };
    return map[groupName] ?? 'seg-' + groupName.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
  }

  // ── Segment summaries + status (B2) ─────────────────────────────────
  /**
   * Version counter bumped on every form valueChanges (see buildForm's
   * subscription). Read inside the `segments` computed so it recomputes when
   * the form mutates — signals do not track FormGroup value changes natively.
   */
  private formVersion = signal(0);

  /** Read a top-level control's value, or undefined if absent. */
  private _val(key: string): unknown {
    return this.form?.get(key)?.value;
  }

  /** True when a value is meaningfully present (not null/undefined/''). */
  private _present(v: unknown): boolean {
    return v !== null && v !== undefined && v !== '';
  }

  /** Resolve the props belonging to a segment label (for validity checks). */
  private _propsForLabel(label: string): SchemaProp[] {
    if (label === 'Model Selection') return this.modelSelectionProps();
    const group = this.groups().find(g => g.name === label);
    return group ? group.props : [];
  }

  /** True if any non-hidden control in the label's group is invalid. */
  private _groupHasInvalid(label: string): boolean {
    return this._propsForLabel(label).some(p => {
      if (this.shouldHideField(p.schema, p.key)) return false;
      return !!this.form.get(p.key)?.invalid;
    });
  }

  /**
   * True if the user has meaningfully configured a group — i.e. any visible
   * field holds a non-default value (a toggle switched on, a number changed, a
   * list populated). Used to flip optional segments (Advanced Engine, Expert
   * Features) from 'idle' to active when e.g. EMA is enabled.
   */
  private _groupIsConfigured(label: string): boolean {
    return this._propsForLabel(label).some(p => {
      if (this.shouldHideField(p.schema, p.key)) return false;
      const v = this.form.get(p.key)?.value;
      const def = p.schema?.default;
      if (typeof v === 'boolean') return v !== (def ?? false);
      if (Array.isArray(v)) return v.length > 0;
      if (v === null || v === undefined || v === '') return false;
      if (def !== undefined && def !== null) return String(v) !== String(def);
      return true;
    });
  }

  /**
   * One-line value summary for a segment header (e.g. "rank 16 / α 8").
   * Derives the Hi-Fi one-liners from current form state. Any missing piece
   * is omitted gracefully — never renders 'undefined'/'null'. Returns '' when
   * a segment has no crisp summary so the slot does not render.
   */
  segmentSummary(groupName: string): string {
    switch (groupName) {
      case 'Model Selection': {
        const name = this.selectedDefinition()?.name;
        if (this._present(name)) return String(name);
        const defId = this._val('definition_id');
        return this._present(defId) ? String(defId) : '';
      }
      case 'VRAM Budget': {
        const report = this.vramReport();
        if (!report) return '';
        const used = (report.peak_mb / 1024).toFixed(1);
        const total = (report.available_mb / 1024).toFixed(1);
        return `${used} / ${total} GB · ${report.fits ? 'fits' : 'over'}`;
      }
      case 'Concepts & Triggerwords': {
        const n = this.getFormArray('datasets')?.length ?? 0;
        if (n <= 0) return '';
        return `${n} dataset${n === 1 ? '' : 's'}`;
      }
      case 'Training Dynamics': {
        const parts: string[] = [];
        const steps = this._val('max_train_steps');
        if (this._present(steps)) parts.push(`${steps} steps`);
        const batch = this._val('train_batch_size');
        if (this._present(batch)) parts.push(`batch ${batch}`);
        return parts.join(' · ');
      }
      case 'LoRA Parameters': {
        const parts: string[] = [];
        const rank = this._val('network_rank');
        if (this._present(rank)) parts.push(`rank ${rank}`);
        const alpha = this._val('network_alpha');
        if (this._present(alpha)) parts.push(`α ${alpha}`);
        return parts.join(' / ');
      }
      case 'Optimizer Settings': {
        const parts: string[] = [];
        const opt = this._val('optimizer_type');
        if (this._present(opt)) parts.push(String(opt));
        const lr = this._val('learning_rate');
        if (this._present(lr)) parts.push(String(lr));
        const sched = this._val('lr_scheduler');
        if (this._present(sched)) parts.push(String(sched));
        return parts.join(' · ');
      }
      case 'Sampling': {
        const every = Number(this._val('sample_every_n_steps') ?? 0);
        if (every <= 0) return '';
        const prompts = this.getFormArray('sample_prompts')?.length ?? 0;
        const parts = [`every ${every} steps`];
        if (prompts > 0) parts.push(`${prompts} prompt${prompts === 1 ? '' : 's'}`);
        return parts.join(' · ');
      }
      default:
        // General Settings, Expert Features, Advanced Engine, etc.
        return '';
    }
  }

  /**
   * Status indicator for a segment header: 'success' | 'warning' | 'idle' | null.
   * Rendered as a chip — success → "✓ OK", warning → "⚠ attn", idle → "idle".
   *
   * The "OK" check, per section:
   *   • VRAM Budget        → success when the estimate fits, warning when over,
   *                          null when no estimate yet.
   *   • Sampling           → success when `sample_every_n_steps` > 0, else idle.
   *   • Advanced Engine /
   *     Expert Features    → success when any field is set to a non-default
   *                          value (e.g. EMA enabled), else idle. (Optional.)
   *   • All other sections → warning if any visible field is invalid, else
   *                          success (they have required values with defaults).
   */
  segmentStatus(groupName: string): 'success' | 'warning' | 'idle' | null {
    if (groupName === 'VRAM Budget') {
      const report = this.vramReport();
      if (!report) return null;
      return report.fits ? 'success' : 'warning';
    }

    if (this._groupHasInvalid(groupName)) return 'warning';

    // Sampling is active (not idle) once a sampling interval is set.
    if (groupName === 'Sampling') {
      return Number(this._val('sample_every_n_steps') ?? 0) > 0 ? 'success' : 'idle';
    }

    // Optional sections become active once the user enables/changes anything.
    if (groupName === 'Advanced Engine' || groupName === 'Expert Features') {
      return this._groupIsConfigured(groupName) ? 'success' : 'idle';
    }

    return 'success';
  }

  /**
   * Source-of-truth segment list for the shell TOC (B3), in DOM order:
   * Model Selection, VRAM Budget, then each non-hidden group. Reads
   * formVersion/vramReport/capabilities to establish reactive dependencies.
   */
  segments = computed<TrainingSegment[]>(() => {
    // Establish dependencies so this recomputes on the relevant changes.
    this.formVersion();
    this.vramReport();
    this.capabilities();

    const build = (label: string): TrainingSegment => ({
      id: this.segmentId(label),
      label,
      sub: this.segmentSummary(label),
      status: this.segmentStatus(label),
    });

    const out: TrainingSegment[] = [build('Model Selection'), build('VRAM Budget')];
    for (const group of this.groups()) {
      if (this.isGroupHidden(group)) continue;
      out.push(build(group.name));
    }
    return out;
  });

  /** Emits the segment list whenever it changes, for the shell to render a TOC. */
  segmentsChanged = output<TrainingSegment[]>();

  private _segmentsEmitEffect = effect(() => {
    this.segmentsChanged.emit(this.segments());
  });

  /** Re-broadcasts the engine's live VRAM report so the shell rail can render it. */
  vramReportChanged = output<VRAMReport | null>();

  private _vramEmitEffect = effect(() => {
    this.vramReportChanged.emit(this.vramReport());
  });

  /** Re-broadcasts the full calibrated estimate for the shell's estimate wall. */
  estimateChanged = output<TrainingEstimate | null>();

  private _estimateEmitEffect = effect(() => {
    this.estimateChanged.emit(this.estimate());
  });

  resolveSchema(schema: SchemaNode | undefined): SchemaNode {
    if (!schema) return {};
    const root = this.schema();
    const definitions = root?.$defs || root?.definitions || {};
    if (schema.$ref) {
      const refKey = schema.$ref.split('/').pop();
      if (refKey && definitions[refKey]) {
        return collapseNullableUnion({ ...definitions[refKey], ...schema });
      }
    }
    return collapseNullableUnion(schema);
  }

  getFormArray(key: string): FormArray {
    return this.form.get(key) as FormArray;
  }

  addArrayItem(key: string, itemSchemaRef: SchemaNode | undefined) {
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
        validators.push((control: AbstractControl): ValidationErrors | null => {
          const val = parseInt(control.value);
          return (val > 0 && val % 32 === 0) ? null : { 'mod32': true };
        });
      }

      array.push(new FormControl(defaultValue, validators));
    }
  }

  // --- Resolution Helpers ---
  isPreset(val: unknown): boolean {
    return [512, 768, 1024, 1280, 1536].includes(parseInt(String(val)));
  }

  isResolutionSelected(res: number): boolean {
    const array = this.getFormArray('resolutions');
    if (!array) return false;
    return array.value.some((id: unknown) => parseInt(String(id)) === res);
  }

  toggleResolution(res: number) {
    const array = this.getFormArray('resolutions');
    if (!array) return;

    const index = array.value.findIndex((v: unknown) => parseInt(String(v)) === res);
    if (index >= 0) {
      array.removeAt(index);
    } else {
      array.push(new FormControl(res, [(control: AbstractControl): ValidationErrors | null => {
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
  getFilteredEnumOptions(prop: SchemaProp): { value: string, label: string, disabled: boolean }[] {
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
  getFilteredNestedEnumOptions(nestedProp: SchemaProp, arrayKey: string, dsIdx: number): { value: string, label: string, disabled: boolean }[] {
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

  isPrimitiveArrayForKey(key: string, properties: SchemaProp[]): boolean {
    const prop = properties.find(p => p.key === key);
    if (!prop || (prop.schema && prop.schema.type !== 'array')) return false;
    const items = this.resolveSchema(prop.schema.items);
    return !items.properties;
  }

  isPrimitiveArray(key: string): boolean {
    return this.isPrimitiveArrayForKey(key, this.properties());
  }

  getArrayItemProps(itemSchemaRef: SchemaNode | undefined): SchemaProp[] {
    const itemSchema = this.resolveSchema(itemSchemaRef);
    const props = itemSchema.properties || {};
    return Object.keys(props)
      .map(key => ({ key, schema: this.resolveSchema(props[key]) }))
      .filter(p => !p.schema.hidden);
  }

  getInlineGroups(props: SchemaProp[]): { name: string, props: SchemaProp[] }[] {
    const map: Record<string, SchemaProp[]> = {};
    for (const prop of props) {
      const ig = prop.schema.inline_group;
      if (ig) {
        if (!map[ig]) map[ig] = [];
        map[ig].push(prop);
      }
    }
    return Object.keys(map).map(name => ({ name, props: map[name] }));
  }

  isFirstInlineGroupProp(key: string, props: SchemaProp[]): boolean {
    const group = props.find(p => p.key === key)?.schema.inline_group;
    if (!group) return false;
    return props.find(p => p.schema.inline_group === group)?.key === key;
  }

  getInlineGroupProps(groupName: string, props: SchemaProp[]): SchemaProp[] {
    return props.filter(p => p.schema.inline_group === groupName);
  }


  isNumber(schema: SchemaNode): boolean {
    return schema.type === 'number' || schema.type === 'integer';
  }

  isBoolean(schema: SchemaNode): boolean {
    return schema.type === 'boolean';
  }

  isString(schema: SchemaNode): boolean {
    return schema.type === 'string' || !schema.type;
  }

  isObject(schema: SchemaNode): boolean {
    return schema.type === 'object';
  }

  isLongInput(key: string, schema: SchemaNode): boolean {
    return key === 'lora_name' || key === 'lora_prefix' || key === 'lora_suffix'
      || key === 'output_dir' || key === 'global_triggerword'
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

  /**
   * Expand the collapsible group addressed by a TOC jump so its body is
   * visible when the shell scrolls to it. Matched by the same `segmentId()`
   * mapping the TOC uses. Model Selection / VRAM Budget are hard-coded,
   * always-open sections (not in `groups()`), so those ids are a safe no-op.
   */
  expandSegment(id: string): void {
    const group = this.groups().find(g => this.segmentId(g.name) === id);
    if (!group || !this.collapsedGroups().has(group.name)) return;
    this.collapsedGroups.update(current => {
      const next = new Set(current);
      next.delete(group.name);
      return next;
    });
  }

  /** Hide an entire group when ALL its fields are hidden by depends_on */
  isGroupHidden(group: { name: string, props: SchemaProp[] }): boolean {
    return group.props.length > 0 && group.props.every(p => this.shouldHideField(p.schema, p.key));
  }

  // --- Conditional Field Disable ---
  isFieldDisabled(schema: SchemaNode): boolean {
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



  shouldHideField(schema: SchemaNode, key?: string): boolean {
    if (schema.hidden) return true;
    // Hide fields the selected model family does not support
    // (capability descriptor: field_visibility[key].supported === false).
    // Reading the capabilities() signal here lets template-invoked calls
    // re-evaluate under OnPush when the descriptor loads.
    const caps = this.capabilities();
    if (key && isFieldHidden(caps, key)) return true;
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
          const ps = propSchema;
          // Strip depends_on-hidden AND family-unsupported (capability-hidden)
          // fields so stale values are never submitted. Passing `key` lets
          // shouldHideField also consult the capability descriptor.
          if (!ps.hidden && this.shouldHideField(ps, key)) {
            delete raw[key];
          }
        }
        // Strip targeted_layers when empty (= filtering OFF, train all layers)
        if (Array.isArray(raw['targeted_layers']) && raw['targeted_layers'].length === 0) {
          delete raw['targeted_layers'];
        }
      }
      // Resolve {placeholder} tokens in lora_name before sending to backend
      if (raw.lora_name) {
        raw.lora_name = raw.lora_name.replace(/\{(\w+)\}/g, (_: string, key: string) => {
          const val = raw[key];
          return val !== undefined && val !== null && val !== '' ? String(val) : '';
        });
      }
      // Attach active project scope so the job is linked to the project
      const pid = this.projectId();
      if (pid) {
        raw.project_id = pid;
      }
      // Link the job back to the template it was built from, so a later
      // "Reload into Training" can re-select that exact template (and recreate
      // it only if it was since deleted). Skip the bare default — there's no
      // real, editable template to link to. Stored in config alongside
      // project_id/job_id (no schema change needed; persisted with the job).
      const selector = this.templateSelector;
      if (selector) {
        const tplId = selector.activeTemplateId();
        const tpl = selector.activeTemplate();
        if (tplId && tplId !== 'default' && tpl && !tpl.is_default && !tpl.readonly) {
          raw.template_id = tplId;
          raw.template_name = tpl.name;
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

  // ── Model Source Override ────────────────────────────────────────────

  /** Fetch the source override for a definition and populate the badge signal. */
  private loadSourceOverride(defId: string): void {
    // Seed the store so cross-tab updates flow through byId; mirror the
    // result into the local badge signal once seeded.
    void this.registryStore.loadFor(defId)
      .then(() => this.modelSourceOverride.set(this.registryStore.byId(defId)() ?? null))
      .catch(() => this.modelSourceOverride.set(null));
  }

  /** Called by the modal after saving or resetting a source override. */
  onSourceOverrideSaved(override: ModelSourceOverride | null): void {
    this.modelSourceOverride.set(override);
    // The modal already persisted the change through RegistryStore;
    // no extra refetch needed — the store row reflects the new value
    // and the badge effect will keep this signal in sync on the next
    // store mutation.
  }
}
