import { Component, ChangeDetectionStrategy, input, output, inject, signal, OnInit, DestroyRef } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { TitleCasePipe } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { ToastService } from '../../../services/toast';
import type { TrainingConfig } from '../../../services/job';
import { SchemaNode } from '../schema-node';

/** `GET /filesystem/browse` — directory listing for the path-picker dialog. */
interface BrowseResponse {
  path: string;
  parent: string;
  entries: { name: string; path: string; type: string }[];
}

/** `GET /checkpoints/inspect` — checkpoint validity + its embedded training config. */
interface CheckpointInspectResponse {
  valid: boolean;
  error?: string;
  config?: TrainingConfig;
  global_step?: number;
}

@Component({
  selector: 'app-dynamic-form-field',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [ReactiveFormsModule, TitleCasePipe],
  host: { 'class': 'contents' },
  template: `
    <div [class.md:col-span-2]="isLongInput()"
         [class.opacity-40]="isDisabled()"
         [class.pointer-events-none]="isDisabled()"
         class="flex flex-col gap-1 transition-opacity duration-200">

      @if (!hideLabel()) {
        <label [for]="fieldKey()" class="field-label flex items-center gap-1.5">
          {{ schema().title || (fieldKey() | titlecase) }}
          @if (isDisabled()) {
            <span class="text-[10px] text-text-disabled">(disabled)</span>
          }
          @if (hasHelp()) {
            <span class="config-help-icon" [title]="helpTip()" (click)="helpRequested.emit(fieldKey()); $event.preventDefault()">?</span>
          }
        </label>
      }
      
      <!-- Integer/Number -->
      @if (isNumber()) {
           @if (isScientific()) {
             <!-- Scientific-notation hint overlay (e.g. learning rate 0.0001 → faint "1e-4"
                  right-aligned inside the field). Spin buttons hidden so the hint owns the
                  right edge; pr-16 reserves space so typed digits never slide under it. -->
             <div class="relative">
               <input [id]="fieldKey()"
                      [type]="'number'"
                      [formControl]="control()"
                      [attr.data-testid]="'config-input-' + fieldKey()"
                      [attr.min]="stepMin()"
                      [attr.max]="schema().max ?? null"
                      [attr.step]="schema().step ?? null"
                      class="input mono pr-16 [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none disabled:opacity-50 disabled:cursor-not-allowed">
               @if (sciHint(sciValue()); as hint) {
                 <span class="pointer-events-none select-none absolute right-3 top-1/2 -translate-y-1/2 font-mono text-sm text-brand/60"
                       [attr.data-testid]="'config-sci-hint-' + fieldKey()">{{ hint }}</span>
               }
             </div>
           } @else {
             <input [id]="fieldKey()"
                    [type]="'number'"
                    [formControl]="control()"
                    [attr.data-testid]="'config-input-' + fieldKey()"
                    [attr.min]="stepMin()"
                    [attr.max]="schema().max ?? null"
                    [attr.step]="schema().step ?? null"
                    class="input mono disabled:opacity-50 disabled:cursor-not-allowed">
           }
      }

       <!-- Boolean (Toggle) -->
      @if (isBoolean()) {
          <label class="relative inline-flex items-center cursor-pointer group">
            <input type="checkbox" [formControl]="control()" 
                   [attr.data-testid]="'config-checkbox-' + fieldKey()"
                   class="sr-only peer">
            <div class="w-7 h-4 bg-surface-high peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-brand/20 rounded-full peer peer-checked:after:translate-x-3 peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-border-subtle after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-brand group-hover:bg-surface-mid transition-all relative"></div>
            <span class="ml-2 text-[11px] text-text-muted group-hover:text-text-secondary">Enable</span>
          </label>
      }

       <!-- String (Input or Enum Dropdown) -->
       @if (isString()) {
          @if (schema().enum) {
               <select [formControl]="control()" 
                       [attr.data-testid]="'config-select-' + fieldKey()"
                       class="select disabled:opacity-50 disabled:cursor-not-allowed">
                  @for (opt of getFilteredEnumOptions(); track opt.value) {
                    <option [value]="opt.value" [disabled]="opt.disabled">{{ opt.label }}</option>
                  }
               </select>
          } @else if (schema().input_type === 'path') {
               <div class="relative">
                 <div class="flex gap-2">
                   <div class="relative flex-1">
                     <span class="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-subtle">
                       <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>
                     </span>
                     <input [type]="'text'" [formControl]="control()" 
                            [attr.data-testid]="'config-input-' + fieldKey()"
                            placeholder="Enter path or click Browse..."
                            class="input pl-9 font-mono disabled:opacity-50 disabled:cursor-not-allowed">
                   </div>
                   <button type="button" (click)="openBrowseDialog()"
                           class="btn sm whitespace-nowrap">
                     Browse
                   </button>
                       @if (fieldKey() === 'resume_from_checkpoint' && control().value) {
                         <button type="button" (click)="loadCheckpointConfig()"
                                 data-testid="load-checkpoint-config"
                                 title="Load training config from this checkpoint"
                                 class="btn sm text-emerald-400 hover:text-emerald-300 whitespace-nowrap">
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
               <div class="flex items-center gap-2">
                 <input [type]="'text'" [formControl]="control()" 
                        [attr.data-testid]="'config-input-' + fieldKey()"
                        class="input disabled:opacity-50 disabled:cursor-not-allowed">
                 @if (fieldKey() === 'caption_prefix') {
                   <button type="button" (click)="autofillRequested.emit()"
                           title="Auto-fill prefix from dataset name"
                           class="p-1.5 text-text-subtle hover:text-brand rounded-theme-md transition-colors shrink-0">
                     <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                       <path d="m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72Z"/>
                       <path d="m14 7 3 3"/>
                       <path d="M5 6v4"/><path d="M19 14v4"/>
                       <path d="M10 2v2"/><path d="M7 8H3"/>
                       <path d="M21 16h-4"/><path d="M11 3H9"/>
                     </svg>
                   </button>
                 }
               </div>
          }
      }

      <!-- String array (e.g. control_images): comma-separated paths bound to a
           string[] control. A flat text mirror (signal) keeps native typing
           free of cursor-fights; parse to the array happens on change/blur,
           and external setValue() syncs back via valueChanges. -->
      @if (isStringArray()) {
          <input [type]="'text'"
                 [value]="stringArrayText()"
                 (change)="onStringArrayChange($event)"
                 [attr.data-testid]="'config-input-' + fieldKey()"
                 placeholder="path/to/image.png, path/to/other.png"
                 class="input mono disabled:opacity-50 disabled:cursor-not-allowed">
      }

      <!-- Nested Object Placeholder -->
      @if (schema().type === 'object') {
          <div class="pl-4 border-l-2 border-surface-mid space-y-4 py-2">
              <p class="text-xs text-text-subtle">Complex objects handled by Parent Form Groups.</p>
          </div>
      }
      
      @if (schema().description) {
        <p class="text-[10.5px] text-text-muted">{{ schema().description }}</p>
      }
    </div>
  `
})
export class DynamicFormFieldComponent implements OnInit {
  control = input.required<FormControl>();
  schema = input.required<SchemaNode>();
  fieldKey = input.required<string>();
  currentBackend = input<string>('local');
  outputDir = input<string>('outputs'); // for fallback browsing
  hasHelp = input<boolean>(false);
  helpTip = input<string>('');
  hideLabel = input<boolean>(false);
  datasetAutocomplete = input<string[]>();

  helpRequested = output<string>();
  checkpointConfigLoaded = output<TrainingConfig>();
  autofillRequested = output<void>();

  private http = inject(HttpClient);
  private toast = inject(ToastService);
  private rtc = inject(RuntimeConfigService);
  private destroyRef = inject(DestroyRef);

  /**
   * Live mirror of the control's value, driving the scientific-notation hint.
   * Seeded in ngOnInit and kept in sync via valueChanges so the hint reflects
   * both typing and programmatic sets (archetype defaults, "Load config").
   */
  sciValue = signal<unknown>(null);

  /**
   * Text mirror of a `string[]` control (e.g. control_images), shown as
   * comma-separated values. Seeded in ngOnInit and re-synced on external
   * setValue() (Load config / reset) via valueChanges; user edits flow the
   * other way through {@link onStringArrayChange}.
   */
  stringArrayText = signal<string>('');

  // Browse Dialog State
  browseActive = signal<boolean>(false);
  browsePath = signal<string>('');
  browseParent = signal<string>('');
  browseEntries = signal<{ name: string, path: string, type: string }[]>([]);

  ngOnInit(): void {
    if (this.isScientific()) {
      this.sciValue.set(this.control().value);
      this.control().valueChanges
        .pipe(takeUntilDestroyed(this.destroyRef))
        .subscribe((v) => this.sciValue.set(v));
    }
    if (this.isStringArray()) {
      this.stringArrayText.set(this.joinStringArray(this.control().value));
      this.control().valueChanges
        .pipe(takeUntilDestroyed(this.destroyRef))
        .subscribe((v) => this.stringArrayText.set(this.joinStringArray(v)));
    }
  }

  isDisabled(): boolean {
    return this.schema().readOnly === true;
  }

  /** Field opts into the scientific-notation hint overlay (set on LR-type fields). */
  isScientific(): boolean {
    return this.schema().display === 'scientific';
  }

  /**
   * Compact scientific-notation hint for a numeric value, or '' to hide the
   * overlay. Empty / non-finite / zero → hidden. Values whose exponent is 0
   * (magnitude in [1, 10), e.g. Prodigy's 1.0) are suppressed since scientific
   * notation adds no clarity there. Otherwise: 0.0001 → "1e-4", 0.00015 → "1.5e-4".
   */
  sciHint(value: unknown): string {
    if (value === null || value === undefined || value === '') return '';
    const n = typeof value === 'number' ? value : parseFloat(String(value));
    if (!Number.isFinite(n) || n === 0) return '';
    const exp = n.toExponential();
    const m = /e([+-]?\d+)$/.exec(exp);
    if (!m || parseInt(m[1], 10) === 0) return '';
    return exp;
  }

  isLongInput(): boolean {
    return this.fieldKey().endsWith('_path') || this.fieldKey().endsWith('_dir') || this.schema().input_type === 'path';
  }

  isNumber(): boolean {
    return this.schema().type === 'integer' || this.schema().type === 'number';
  }

  /**
   * The `min` attribute for a number spinner, aligned to the step grid.
   *
   * A native number spinner anchors its step grid to `min`: with `min: 1,
   * step: 100` (max_train_steps) the only reachable values are 1, 101, … 5901,
   * 6001 — so the arrows snap to off-round "…01" numbers. Anchoring the grid to
   * the largest step-multiple ≤ `min` makes the arrows land on clean multiples
   * (…900 → …000). Mins already on the grid (e.g. resolution min 256 / step 64,
   * weight min 0.50 / step 0.05) are returned unchanged. No reactive min
   * validator exists, so this only affects the spinner anchor + HTML5 floor.
   */
  stepMin(): number | null {
    const min = this.schema().min ?? null;
    const step = this.schema().step;
    if (min == null || !step) return min;
    const ratio = min / step;
    if (Math.abs(ratio - Math.round(ratio)) < 1e-9) return min; // already on the grid
    return Math.floor(ratio) * step;
  }

  isBoolean(): boolean {
    return this.schema().type === 'boolean';
  }

  isString(): boolean {
    return this.schema().type === 'string' && !this.schema().input_type?.includes('dataset');
  }

  /**
   * A plain `string[]` field (e.g. control_images) — rendered as a
   * comma-separated text input. Excludes the specialised `layer_checklist`
   * arrays (those are handled by the parent group's custom card).
   */
  isStringArray(): boolean {
    const s = this.schema();
    return s.type === 'array' && s.items?.type === 'string' && s.ui_type !== 'layer_checklist';
  }

  /** Render a `string[]` (or legacy '' / null seed) as comma-separated text. */
  private joinStringArray(value: unknown): string {
    if (Array.isArray(value)) return value.join(', ');
    return typeof value === 'string' ? value : '';
  }

  /** Parse comma-separated text back to a trimmed, empty-stripped `string[]`. */
  onStringArrayChange(event: Event): void {
    const arr = (event.target as HTMLInputElement).value
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
    this.control().setValue(arr);
    this.control().markAsDirty();
  }

  getFilteredEnumOptions() {
    const propSchema = this.schema();
    if (!propSchema.enum) return [];

    let defaultOptions = propSchema.enum.map((opt) => ({
      value: opt,
      label: propSchema.options_labels?.[opt] || opt,
      disabled: false
    }));

    if (propSchema.backend_map) {
      const backendMap = propSchema.backend_map;
      const backend = this.currentBackend();
      const rawSchemes = backendMap[backend];
      const supportedSchemes = Array.isArray(rawSchemes) ? rawSchemes : (rawSchemes?.schemes || []);

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
    this.http.get<BrowseResponse>(`${this.rtc.apiUrl}/filesystem/browse`, { params: { path } }).subscribe({
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

    this.http.get<CheckpointInspectResponse>(`${this.rtc.apiUrl}/checkpoints/inspect`, {
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
