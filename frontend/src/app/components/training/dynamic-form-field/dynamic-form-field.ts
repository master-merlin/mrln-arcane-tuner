import { Component, input, output, inject, signal, OnInit } from '@angular/core';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { TitleCasePipe } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { ToastService } from '../../../services/toast';

@Component({
  selector: 'app-dynamic-form-field',
  standalone: true,
  imports: [ReactiveFormsModule, TitleCasePipe],
  host: { 'class': 'contents' },
  template: `
    <div [class.md:col-span-2]="isLongInput()"
         [class.opacity-40]="isDisabled()"
         [class.pointer-events-none]="isDisabled()"
         class="flex flex-col gap-2 transition-opacity duration-200">
      
      <label [for]="fieldKey()" class="text-sm font-medium text-text-secondary flex items-center gap-1.5">
        {{ schema().title || (fieldKey() | titlecase) }}
        @if (isDisabled()) {
          <span class="text-[10px] text-text-disabled">(disabled)</span>
        }
        @if (hasHelp()) {
          <span class="config-help-icon" [title]="helpTip()" (click)="helpRequested.emit(fieldKey()); $event.preventDefault()">?</span>
        }
      </label>
      
      <!-- Integer/Number -->
      @if (isNumber()) {
           <input [id]="fieldKey()" 
                  [type]="'number'" 
                  [formControl]="control()"
                  [attr.data-testid]="'config-input-' + fieldKey()"
                  [attr.min]="schema().min ?? null"
                  [attr.max]="schema().max ?? null"
                  [attr.step]="schema().step ?? null"
                  class="bg-surface-mid border border-surface-high rounded-theme-lg px-4 py-2 text-white w-full focus:ring-2 focus:ring-brand outline-none transition-all disabled:opacity-50 disabled:cursor-not-allowed">
      }

       <!-- Boolean (Toggle) -->
      @if (isBoolean()) {
          <label class="relative inline-flex items-center cursor-pointer group">
            <input type="checkbox" [formControl]="control()" 
                   [attr.data-testid]="'config-checkbox-' + fieldKey()"
                   class="sr-only peer">
            <div class="w-11 h-6 bg-surface-high peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-brand/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-border-subtle after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-brand group-hover:bg-surface-mid transition-all"></div>
            <span class="ml-3 text-sm font-medium text-text-muted group-hover:text-text-secondary">Enable</span>
          </label>
      }

       <!-- String (Input or Enum Dropdown) -->
       @if (isString()) {
          @if (schema().enum) {
               <select [formControl]="control()" 
                       [attr.data-testid]="'config-select-' + fieldKey()"
                       class="bg-surface-high border border-surface-high/50 rounded-theme-lg px-4 py-2 text-white w-full appearance-none focus:ring-2 focus:ring-brand outline-none disabled:opacity-50 disabled:cursor-not-allowed">
                  @for (opt of getFilteredEnumOptions(); track opt.value) {
                    <option [value]="opt.value" [disabled]="opt.disabled">{{ opt.label }}</option>
                  }
               </select>
          } @else if (schema().input_type === 'path') {
               <div class="relative">
                 <div class="flex gap-2">
                   <div class="relative flex-1">
                     <span class="absolute left-3 top-1/2 -translate-y-1/2 text-text-subtle">
                       <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>
                     </span>
                     <input [type]="'text'" [formControl]="control()" 
                            [attr.data-testid]="'config-input-' + fieldKey()"
                            placeholder="Enter path or click Browse..."
                            class="bg-surface-mid border border-surface-high rounded-theme-lg pl-10 pr-4 py-2 text-white w-full focus:ring-2 focus:ring-brand outline-none transition-all font-mono text-sm disabled:opacity-50 disabled:cursor-not-allowed">
                   </div>
                   <button type="button" (click)="openBrowseDialog()"
                           class="px-3 py-2 bg-surface-mid hover:bg-surface-high border border-surface-high rounded-theme-lg text-xs font-medium text-brand transition-colors whitespace-nowrap">
                     Browse
                   </button>
                       @if (fieldKey() === 'resume_from_checkpoint' && control().value) {
                         <button type="button" (click)="loadCheckpointConfig()"
                                 data-testid="load-checkpoint-config"
                                 title="Load training config from this checkpoint"
                                 class="px-3 py-2 bg-surface-mid hover:bg-surface-high border border-surface-high rounded-theme-lg text-xs font-medium text-emerald-400 hover:text-emerald-300 transition-colors whitespace-nowrap flex items-center gap-1.5">
                           <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                             <path d="M12 3v12"/><path d="m8 11 4 4 4-4"/><path d="M8 5H4a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-4"/>
                           </svg>
                           Load config
                         </button>
                       }
                 </div>
                 
                 <!-- Folder Picker Dropdown (Isolated state) -->
                 @if (browseActive()) {
                   <div class="absolute z-50 mt-1 w-full bg-surface-low border border-surface-mid rounded-theme-lg shadow-2xl overflow-hidden animate-in fade-in slide-in-from-top-1 duration-150">
                     <div class="flex items-center justify-between px-3 py-2 border-b border-surface-mid/50 bg-surface-mid/30">
                       <span class="text-[10px] text-text-muted font-mono truncate flex-1" [title]="browsePath()">{{ browsePath() }}</span>
                       <button type="button" (click)="closeBrowseDialog()" class="text-text-subtle hover:text-white ml-2">
                         <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
                       </button>
                     </div>
                     <div class="max-h-48 overflow-y-auto scrollbar-thin scrollbar-thumb-surface-high">
                       <button type="button" (click)="browseNavigate(browseParent())" 
                               class="w-full text-left px-3 py-1.5 text-xs text-text-muted hover:bg-surface-mid/50 hover:text-white flex items-center gap-2 transition-colors font-mono">
                         <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m18 15-6-6-6 6"/></svg>
                         ..
                       </button>
                       @for (entry of browseEntries(); track entry.path) {
                         <button type="button" 
                                 (click)="entry.type === 'checkpoint' ? selectBrowsePath(entry.path) : browseNavigate(entry.path)"
                                 class="w-full text-left px-3 py-1.5 text-xs hover:bg-surface-mid/50 flex items-center gap-2 transition-colors font-mono"
                                 [class.text-brand]="entry.type === 'checkpoint'"
                                 [class.text-text-secondary]="entry.type !== 'checkpoint'">
                           @if (entry.type === 'checkpoint') {
                             <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-green-400 shrink-0"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>
                           } @else {
                             <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-text-subtle shrink-0"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
                           }
                           {{ entry.name }}
                         </button>
                       }
                       @if (browseEntries().length === 0) {
                         <div class="px-3 py-3 text-xs text-text-subtle italic text-center">No subdirectories</div>
                       }
                     </div>
                   </div>
                 }
               </div>
          } @else {
               <input [type]="'text'" [formControl]="control()" 
                      [attr.data-testid]="'config-input-' + fieldKey()"
                      class="bg-surface-mid border border-surface-high rounded-theme-lg px-4 py-2 text-white w-full focus:ring-2 focus:ring-brand outline-none transition-all disabled:opacity-50 disabled:cursor-not-allowed">
          }
      }

      <!-- Nested Object Placeholder -->
      @if (schema().type === 'object') {
          <div class="pl-4 border-l-2 border-surface-mid space-y-4 py-2">
              <p class="text-xs text-text-subtle">Complex objects handled by Parent Form Groups.</p>
          </div>
      }
      
      @if (schema().description) {
        <p class="text-xs text-text-subtle italic">{{ schema().description }}</p>
      }
    </div>
  `
})
export class DynamicFormFieldComponent {
  control = input.required<FormControl>();
  schema = input.required<any>();
  fieldKey = input.required<string>();
  currentBackend = input<string>('local');
  outputDir = input<string>('outputs'); // for fallback browsing
  hasHelp = input<boolean>(false);
  helpTip = input<string>('');
  datasetAutocomplete = input<string[]>();

  helpRequested = output<string>();
  checkpointConfigLoaded = output<any>();
  autofillRequested = output<void>();

  private http = inject(HttpClient);
  private toast = inject(ToastService);
  private rtc = inject(RuntimeConfigService);

  // Browse Dialog State
  browseActive = signal<boolean>(false);
  browsePath = signal<string>('');
  browseParent = signal<string>('');
  browseEntries = signal<{ name: string, path: string, type: string }[]>([]);

  isDisabled(): boolean {
    return this.schema().readOnly === true;
  }

  isLongInput(): boolean {
    return this.fieldKey().endsWith('_path') || this.fieldKey().endsWith('_dir') || this.schema().input_type === 'path';
  }

  isNumber(): boolean {
    return this.schema().type === 'integer' || this.schema().type === 'number';
  }

  isBoolean(): boolean {
    return this.schema().type === 'boolean';
  }

  isString(): boolean {
    return this.schema().type === 'string' && !this.schema().input_type?.includes('dataset');
  }

  getFilteredEnumOptions() {
    const propSchema = this.schema();
    if (!propSchema.enum) return [];

    let defaultOptions = propSchema.enum.map((opt: any) => ({
      value: opt,
      label: propSchema.options_labels?.[opt] || opt,
      disabled: false
    }));

    if (propSchema.backend_map) {
      const backendMap = propSchema.backend_map;
      const backend = this.currentBackend();
      const supportedSchemes = backendMap[backend] || [];

      return defaultOptions.map((opt: { value: string, label: string, disabled: boolean }) => {
        if (opt.value === 'none' || opt.value === 'bf16') {
          return opt;
        }
        if (!supportedSchemes.includes(opt.value)) {
          return { ...opt, disabled: true, label: `${opt.label} (Not supported by ${backend})` };
        }
        return opt;
      }).filter((opt: { value: string, label: string, disabled: boolean }) => !(propSchema.hide_unsupported && opt.disabled));
    }
    return defaultOptions;
  }

  // --- Folder Browse Dialog ---
  openBrowseDialog() {
    if (this.isDisabled()) return;
    this.browseActive.set(true);
    const initialPath = this.control().value || this.outputDir();
    this.browseNavigate(initialPath);
  }

  closeBrowseDialog() {
    this.browseActive.set(false);
    this.browseEntries.set([]);
  }

  browseNavigate(path: string) {
    this.http.get<any>(`${this.rtc.apiUrl}/filesystem/browse`, { params: { path } }).subscribe({
      next: (result) => {
        this.browsePath.set(result.path);
        this.browseParent.set(result.parent);
        this.browseEntries.set(result.entries || []);
      },
      error: (err) => {
        console.error('[Browse] Failed to browse', err);
        if (err.status === 404 && path !== 'outputs') {
          this.browseNavigate('outputs');
        }
      }
    });
  }

  selectBrowsePath(path: string) {
    this.control().setValue(path);
    this.control().markAsDirty();
    this.closeBrowseDialog();
  }

  loadCheckpointConfig() {
    const checkpointPath = this.control().value;
    if (!checkpointPath) return;

    this.http.get<any>(`${this.rtc.apiUrl}/checkpoints/inspect`, {
      params: { path: checkpointPath }
    }).subscribe({
      next: (result) => {
        if (!result.valid) {
          this.toast.error('Invalid checkpoint: ' + (result.error || 'unknown error'));
          return;
        }
        if (!result.config || Object.keys(result.config).length === 0) {
          this.toast.warning('Checkpoint has no saved training config');
          return;
        }
        this.checkpointConfigLoaded.emit(result.config);
        this.toast.success(`Loaded config from checkpoint (step ${result.global_step})`);
      },
      error: (err) => {
        this.toast.error('Failed to inspect checkpoint: ' + (err.error?.detail || err.message));
      }
    });
  }
}
