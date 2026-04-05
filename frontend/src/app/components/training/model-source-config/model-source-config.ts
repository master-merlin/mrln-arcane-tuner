import { Component, input, output, inject, signal, computed, OnInit, effect } from '@angular/core';
import { UpperCasePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ToastService } from '../../../services/toast';
import {
  ModelService,
  ModelSourceOverride,
  ModelSourceType,
  PathValidationResult,
} from '../../../services/model.service';

@Component({
  selector: 'app-model-source-config',
  standalone: true,
  imports: [FormsModule, UpperCasePipe],
  template: `
    <!-- Backdrop -->
    <div class="fixed inset-0 bg-overlay backdrop-blur-md z-[100] flex items-center justify-center p-4 animate-in fade-in duration-300"
         (click)="close.emit()">

      <!-- Modal Panel -->
      <div class="bg-surface-low border border-surface-mid rounded-theme-xl w-full max-w-lg shadow-2xl p-6 transform animate-in slide-in-from-bottom-4 duration-300"
           (click)="$event.stopPropagation()">

        <!-- Header -->
        <div class="flex items-center gap-4 mb-6">
          <div class="p-3 bg-brand/10 rounded-theme-md border border-brand/20 text-brand">
            <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
              <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
              <line x1="12" y1="22.08" x2="12" y2="12"/>
            </svg>
          </div>
          <div>
            <h3 class="text-lg font-bold text-white">Model Source</h3>
            <p class="text-xs text-text-subtle mt-0.5 truncate max-w-[350px]"
               [title]="definitionId()">{{ definitionName() || definitionId() }}</p>
          </div>
        </div>

        <!-- Source Type Selector -->
        <div class="space-y-4">
          <div class="flex flex-col gap-1.5">
            <label class="text-[10px] uppercase tracking-wider text-text-subtle font-bold">Source Type</label>
            <select [ngModel]="sourceType()" (ngModelChange)="onSourceTypeChange($event)"
                    data-testid="model-source-type-select"
                    class="w-full bg-surface-mid border border-surface-high text-white text-sm rounded-theme-md px-3 py-2 outline-none focus:border-brand transition-colors">
              <option value="hf_hub">HuggingFace Hub (Default)</option>
              <option value="local_diffusers">Local Diffusers Copy</option>
              <option value="local_safetensors">Local Safetensors (Advanced)</option>
            </select>
          </div>

          <!-- HF Hub: Skip Update Toggle -->
          @if (sourceType() === 'hf_hub') {
            <label class="flex items-center gap-3 cursor-pointer group p-3 bg-surface-high/30 rounded-theme-md border border-surface-mid/50">
              <input type="checkbox" [ngModel]="skipUpdate()" (ngModelChange)="skipUpdate.set($event)"
                     data-testid="model-source-skip-update"
                     class="sr-only peer">
              <div class="w-10 h-5 bg-surface-high peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-brand/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-border-subtle after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-brand group-hover:bg-surface-mid transition-all relative flex-shrink-0"></div>
              <div>
                <span class="text-sm font-medium text-text-secondary block">Skip HF Updates</span>
                <span class="text-[11px] text-text-subtle">Only use locally cached files — never contact HuggingFace</span>
              </div>
            </label>
          }

          <!-- Local Path Input -->
          @if (sourceType() !== 'hf_hub') {
            <div class="flex flex-col gap-1.5">
              <label class="text-[10px] uppercase tracking-wider text-text-subtle font-bold">Local Path</label>
              <div class="flex gap-2">
                <input type="text" [ngModel]="localPath()" (ngModelChange)="localPath.set($event)"
                       data-testid="model-source-local-path"
                       placeholder="D:\\Models\\sdxl-base"
                       class="flex-1 bg-surface-mid border border-surface-high text-white text-sm rounded-theme-md px-3 py-2 outline-none focus:border-brand transition-colors font-mono">
                <button type="button" (click)="browseFolder()"
                        data-testid="model-source-browse-btn"
                        [disabled]="browsing()"
                        title="Browse for folder"
                        class="px-2.5 py-2 bg-surface-high hover:bg-surface-mid text-text-secondary hover:text-white text-sm rounded-theme-md border border-surface-high transition-colors disabled:opacity-40 font-bold">
                  @if (browsing()) {
                    <svg class="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
                      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                    </svg>
                  } @else {
                    ···
                  }
                </button>
                <button type="button" (click)="validateCurrentPath()"
                        data-testid="model-source-validate-btn"
                        [disabled]="!localPath() || validating()"
                        class="px-3 py-2 bg-surface-high hover:bg-surface-mid text-brand text-sm rounded-theme-md border border-surface-high transition-colors disabled:opacity-40 flex items-center gap-1.5 whitespace-nowrap">
                  @if (validating()) {
                    <svg class="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
                      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                    </svg>
                  }
                  Validate
                </button>
              </div>
            </div>

            <!-- Validation Result -->
            @if (validationResult(); as v) {
              <div class="rounded-theme-md border p-3 text-sm"
                   [class]="v.valid
                     ? 'bg-emerald-500/5 border-emerald-500/30 text-emerald-400'
                     : 'bg-danger/5 border-danger/30 text-danger'">
                @if (v.valid) {
                  <div class="font-bold mb-1">✓ Valid — {{ v.type | uppercase }}</div>
                  @if (v.components_found.length) {
                    <div class="text-xs text-text-subtle">
                      Components: {{ v.components_found.join(', ') }}
                    </div>
                  }
                } @else {
                  <div class="font-bold">✗ Path not found or not a valid model directory</div>
                }
                @for (w of v.warnings; track w) {
                  <div class="mt-1.5 text-xs text-amber-400 flex items-start gap-1.5">
                    <span class="mt-0.5">⚠</span>
                    <span>{{ w }}</span>
                  </div>
                }
              </div>
            }
          }

          <!-- Safetensors Warning -->
          @if (sourceType() === 'local_safetensors') {
            <div class="rounded-theme-md bg-amber-500/5 border border-amber-500/30 p-3 text-sm text-amber-400 flex items-start gap-2">
              <svg class="w-5 h-5 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
              </svg>
              <div>
                <span class="font-bold block mb-0.5">Advanced Mode</span>
                <span class="text-xs text-text-subtle">
                  Requires a fully enriched model definition with architecture_params.
                  Each component (VAE, Tokenizer, TE, etc.) must be present as
                  named .safetensors files or Diffusers sub-directories.
                </span>
              </div>
            </div>
          }
        </div>

        <!-- Actions -->
        <div class="flex justify-between items-center mt-8">
          @if (hasExistingOverride()) {
            <button type="button" (click)="resetToDefault()"
                    data-testid="model-source-reset-btn"
                    class="text-xs text-danger hover:text-danger/80 font-bold uppercase tracking-widest transition-colors">
              Reset to Default
            </button>
          } @else {
            <div></div>
          }
          <div class="flex gap-3">
            <button type="button" (click)="close.emit()"
                    class="text-text-subtle hover:text-white text-sm font-bold px-5 py-2.5 transition-colors uppercase tracking-widest">
              Cancel
            </button>
            <button type="button" (click)="save()"
                    data-testid="model-source-save-btn"
                    [disabled]="!canSave()"
                    class="bg-brand hover:bg-brand/80 text-white text-sm font-bold px-5 py-2.5 rounded-theme-md transition-colors uppercase tracking-widest disabled:opacity-40">
              Save
            </button>
          </div>
        </div>
      </div>
    </div>
  `,
})
export class ModelSourceConfigComponent implements OnInit {
  definitionId = input.required<string>();
  definitionName = input<string>('');
  initialBrowsePath = input<string>('');  // Global default model path for browse dialog
  close = output<void>();
  saved = output<ModelSourceOverride | null>();

  private modelService = inject(ModelService);
  private toast = inject(ToastService);

  sourceType = signal<ModelSourceType>('hf_hub');
  localPath = signal<string>('');
  skipUpdate = signal(false);
  validating = signal(false);
  browsing = signal(false);
  validationResult = signal<PathValidationResult | null>(null);
  hasExistingOverride = signal(false);

  canSave = computed(() => {
    const st = this.sourceType();
    if (st === 'hf_hub') return true;
    return !!this.localPath();
  });

  ngOnInit(): void {
    this.modelService.getModelSource(this.definitionId()).subscribe({
      next: (override) => {
        this.sourceType.set(override.source_type);
        this.localPath.set(override.local_path || '');
        this.skipUpdate.set(override.skip_update);
        this.hasExistingOverride.set(
          override.source_type !== 'hf_hub' || override.skip_update,
        );
      },
      error: () => {
        // No override exists — defaults applied
      },
    });
  }

  onSourceTypeChange(value: ModelSourceType): void {
    this.sourceType.set(value);
    this.validationResult.set(null);
  }

  browseFolder(): void {
    this.browsing.set(true);
    this.modelService.pickFolder(this.localPath() || this.initialBrowsePath() || '').subscribe({
      next: (result) => {
        this.browsing.set(false);
        if (result.path) {
          this.localPath.set(result.path);
          this.validationResult.set(null);
          // Auto-validate the selected path
          this.validateCurrentPath();
        }
      },
      error: () => {
        this.browsing.set(false);
        this.toast.error('Folder picker failed');
      },
    });
  }

  validateCurrentPath(): void {
    const path = this.localPath();
    if (!path) return;
    this.validating.set(true);
    this.modelService.validatePath(this.definitionId(), path).subscribe({
      next: (result) => {
        this.validationResult.set(result);
        this.validating.set(false);
      },
      error: () => {
        this.validating.set(false);
        this.toast.error('Validation failed');
      },
    });
  }

  save(): void {
    const override: ModelSourceOverride = {
      source_type: this.sourceType(),
      local_path: this.sourceType() !== 'hf_hub' ? this.localPath() : null,
      skip_update: this.sourceType() === 'hf_hub' ? this.skipUpdate() : true,
    };

    this.modelService.setModelSource(this.definitionId(), override).subscribe({
      next: (result) => {
        this.toast.success('Model source updated');
        this.saved.emit(result);
        this.close.emit();
      },
      error: (err) => {
        const msg = err.error?.detail || 'Failed to save source override';
        this.toast.error(msg);
      },
    });
  }

  resetToDefault(): void {
    this.modelService.deleteModelSource(this.definitionId()).subscribe({
      next: () => {
        this.toast.success('Source reset to HF Hub default');
        this.saved.emit(null);
        this.close.emit();
      },
      error: () => this.toast.error('Failed to reset source'),
    });
  }
}
