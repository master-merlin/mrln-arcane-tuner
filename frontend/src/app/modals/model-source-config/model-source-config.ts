import { Component, ChangeDetectionStrategy, inject, signal, computed, OnInit } from '@angular/core';
import { UpperCasePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IcoComponent } from '../../icons/ico.component';
import { ToastService } from '../../services/toast';
import {
  ModelService,
  ModelSourceOverride,
  ModelSourceType,
  PathValidationResult,
} from '../../services/model.service';
import { RegistryStore } from '../../state/registry.store';
import { OverlayStore } from '../../state/overlay.store';

/**
 * Payload passed via `overlay.openModal('model-source-config', …)`. The opener
 * (training-dynamic-config) supplies the definition context and a callback that
 * receives the saved override (or null on reset).
 */
export interface ModelSourceConfigData {
  definitionId: string;
  definitionName?: string;
  /** Global default model path used to seed the folder browser. */
  initialBrowsePath?: string;
  onSaved?: (override: ModelSourceOverride | null) => void;
}

/**
 * Model-source config modal — pick where a model definition's weights come from
 * (HF Hub / local Diffusers copy / local safetensors) and persist the override
 * through RegistryStore.
 *
 * Registered in modal-layer and opened via OverlayStore (previously rendered
 * inline by training-dynamic-config with its own backdrop). modal-layer owns the
 * backdrop / focus-trap / Esc chrome; this component renders only the dialog body.
 */
@Component({
  selector: 'app-model-source-config',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FormsModule, UpperCasePipe, IcoComponent],
  template: `
    <div class="modal-head">
      <div class="msc-head-main">
        <span class="msc-badge"><app-ico name="Box" [size]="18"/></span>
        <div class="msc-head-text">
          <div class="eyebrow">MODEL SOURCE</div>
          <div class="msc-title" [title]="definitionId">{{ definitionName || definitionId }}</div>
        </div>
      </div>
      <button type="button" class="icon-btn" (click)="close()"
              data-testid="model-source-close-btn" aria-label="Close">×</button>
    </div>

    <div class="modal-body msc-body">
      <!-- Source Type Selector -->
      <div class="msc-field">
        <label class="field-label" for="msc-source-type">Source Type</label>
        <select id="msc-source-type" class="input" [ngModel]="sourceType()"
                (ngModelChange)="onSourceTypeChange($event)"
                data-testid="model-source-type-select">
          <option value="hf_hub">HuggingFace Hub (Default)</option>
          <option value="local_diffusers">Local Diffusers Copy</option>
          <option value="local_safetensors">Local Safetensors (Advanced)</option>
        </select>
      </div>

      <!-- HF Hub: Skip Update Toggle -->
      @if (sourceType() === 'hf_hub') {
        <div class="msc-toggle-row">
          <button type="button" class="toggle" [class.on]="skipUpdate()"
                  (click)="skipUpdate.set(!skipUpdate())"
                  role="switch" [attr.aria-checked]="skipUpdate()"
                  data-testid="model-source-skip-update"></button>
          <div class="msc-toggle-text">
            <span class="msc-toggle-title">Skip HF Updates</span>
            <span class="msc-toggle-desc">Only use locally cached files — never contact HuggingFace</span>
          </div>
        </div>
      }

      <!-- Local Path Input -->
      @if (sourceType() !== 'hf_hub') {
        <div class="msc-field">
          <label class="field-label" for="msc-local-path">Local Path</label>
          <div class="msc-path-row">
            <input id="msc-local-path" type="text" class="input mono"
                   [ngModel]="localPath()" (ngModelChange)="localPath.set($event)"
                   data-testid="model-source-local-path"
                   placeholder="D:\\Models\\sdxl-base">
            <button type="button" class="btn sm" (click)="browseFolder()"
                    data-testid="model-source-browse-btn"
                    [disabled]="browsing()"
                    title="Browse for folder">
              @if (browsing()) {
                <app-ico name="LoaderCircle" [size]="14" class="msc-spin"/>
              } @else {
                <app-ico name="FolderOpen" [size]="14"/>
              }
            </button>
            <button type="button" class="btn sm" (click)="validateCurrentPath()"
                    data-testid="model-source-validate-btn"
                    [disabled]="!localPath() || validating()">
              @if (validating()) {
                <app-ico name="LoaderCircle" [size]="14" class="msc-spin"/>
              }
              Validate
            </button>
          </div>
        </div>

        <!-- Validation Result -->
        @if (validationResult(); as v) {
          <div class="msc-note" [class.ok]="v.valid" [class.err]="!v.valid">
            @if (v.valid) {
              <div class="msc-note-title">✓ Valid — {{ v.type | uppercase }}</div>
              @if (v.components_found.length) {
                <div class="msc-note-sub">Components: {{ v.components_found.join(', ') }}</div>
              }
            } @else {
              <div class="msc-note-title">✗ Path not found or not a valid model directory</div>
            }
            @for (w of v.warnings; track w) {
              <div class="msc-note-warn">
                <span>⚠</span>
                <span>{{ w }}</span>
              </div>
            }
          </div>
        }
      }

      <!-- Safetensors Warning -->
      @if (sourceType() === 'local_safetensors') {
        <div class="msc-note warn msc-note-flex">
          <app-ico name="TriangleAlert" [size]="18"/>
          <div>
            <span class="msc-note-title">Advanced Mode</span>
            <span class="msc-note-sub">
              Requires a fully enriched model definition with architecture_params.
              Each component (VAE, Tokenizer, TE, etc.) must be present as
              named .safetensors files or Diffusers sub-directories.
            </span>
          </div>
        </div>
      }
    </div>

    <!-- Actions -->
    <div class="modal-foot msc-foot">
      @if (hasExistingOverride()) {
        <button type="button" class="btn danger-out" (click)="resetToDefault()"
                data-testid="model-source-reset-btn">
          Reset to Default
        </button>
      } @else {
        <span></span>
      }
      <span class="msc-foot-actions">
        <button type="button" class="btn ghost" (click)="close()"
                data-testid="model-source-cancel-btn">
          Cancel
        </button>
        <button type="button" class="btn primary" (click)="save()"
                data-testid="model-source-save-btn"
                [disabled]="!canSave()">
          Save
        </button>
      </span>
    </div>
  `,
  styles: [`
    .msc-head-main { display: flex; align-items: center; gap: 12px; min-width: 0; }
    .msc-badge {
      display: inline-flex; align-items: center; justify-content: center;
      width: 38px; height: 38px; flex-shrink: 0;
      border-radius: var(--radius-theme-md);
      background: color-mix(in oklab, var(--color-brand) 12%, transparent);
      border: 1px solid color-mix(in oklab, var(--color-brand) 22%, transparent);
      color: var(--color-brand);
    }
    .msc-head-text { min-width: 0; }
    .msc-title {
      font-size: 15px; font-weight: 700; color: var(--color-text-primary);
      max-width: 340px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .msc-body { display: flex; flex-direction: column; gap: 16px; }
    .msc-field { display: flex; flex-direction: column; }
    .msc-path-row { display: flex; gap: 8px; }
    .msc-path-row .input { flex: 1; min-width: 0; }
    .msc-path-row .btn { flex-shrink: 0; white-space: nowrap; }
    .msc-spin { animation: msc-spin 0.8s linear infinite; }
    @keyframes msc-spin { to { transform: rotate(360deg); } }

    .msc-toggle-row {
      display: flex; align-items: center; gap: 12px;
      padding: 12px; border-radius: var(--radius-theme-md);
      background: var(--color-surface-mid);
      border: 1px solid var(--color-border-subtle);
    }
    .msc-toggle-text { display: flex; flex-direction: column; }
    .msc-toggle-title { font-size: 13px; font-weight: 500; color: var(--color-text-secondary); }
    .msc-toggle-desc { font-size: 11px; color: var(--color-text-subtle); }

    .msc-note {
      border-radius: var(--radius-theme-md);
      border: 1px solid var(--color-border-subtle);
      padding: 12px; font-size: 13px; color: var(--color-text-secondary);
    }
    .msc-note.ok {
      background: color-mix(in oklab, var(--color-success) 8%, transparent);
      border-color: color-mix(in oklab, var(--color-success) 30%, transparent);
      color: var(--color-success);
    }
    .msc-note.err {
      background: color-mix(in oklab, var(--color-danger) 8%, transparent);
      border-color: color-mix(in oklab, var(--color-danger) 30%, transparent);
      color: var(--color-danger);
    }
    .msc-note.warn {
      background: color-mix(in oklab, var(--color-warning) 8%, transparent);
      border-color: color-mix(in oklab, var(--color-warning) 30%, transparent);
      color: var(--color-warning);
    }
    .msc-note-flex { display: flex; align-items: flex-start; gap: 10px; }
    .msc-note-flex app-ico { flex-shrink: 0; margin-top: 1px; }
    .msc-note-title { font-weight: 700; display: block; }
    .msc-note-sub {
      display: block; margin-top: 4px; font-size: 12px;
      color: var(--color-text-subtle); font-weight: 400;
    }
    .msc-note-warn {
      display: flex; align-items: flex-start; gap: 6px;
      margin-top: 6px; font-size: 12px; color: var(--color-warning);
    }

    .msc-foot { justify-content: space-between; align-items: center; }
    .msc-foot-actions { display: inline-flex; gap: 8px; }
  `],
})
export class ModelSourceConfigComponent implements OnInit {
  private overlay = inject(OverlayStore);
  private modelService = inject(ModelService);
  private toast = inject(ToastService);
  private registryStore = inject(RegistryStore);

  private data = (this.overlay.topModal()?.data ?? {}) as ModelSourceConfigData;
  protected definitionId = this.data.definitionId ?? '';
  protected definitionName = this.data.definitionName ?? '';
  private initialBrowsePath = this.data.initialBrowsePath ?? '';

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
    // Seed the store so subsequent reads via byId reflect the persisted
    // state. Once seeded, the byId signal also picks up cross-tab updates
    // via entity.changed:registry_model broadcasts.
    void this.registryStore.loadFor(this.definitionId).then(() => {
      const override = this.registryStore.byId(this.definitionId)();
      if (override) {
        this.sourceType.set(override.source_type);
        this.localPath.set(override.local_path || '');
        this.skipUpdate.set(override.skip_update);
        this.hasExistingOverride.set(
          override.source_type !== 'hf_hub' || override.skip_update,
        );
      }
      // No override → defaults remain in place.
    });
  }

  onSourceTypeChange(value: ModelSourceType): void {
    this.sourceType.set(value);
    this.validationResult.set(null);
  }

  browseFolder(): void {
    this.browsing.set(true);
    this.modelService.pickFolder(this.localPath() || this.initialBrowsePath || '').subscribe({
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
    this.modelService.validatePath(this.definitionId, path).subscribe({
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

    // Optimistic mutation through the store. The store toasts on
    // failure and rolls the row back; success path emits the override
    // and closes. Closing immediately is safe — the store's rollback
    // will surface via entity.changed broadcasts to any other consumer.
    this.close();
    this.data.onSaved?.(override);
    this.toast.success('Model source updated');
    void this.registryStore.setOverride(this.definitionId, override);
  }

  resetToDefault(): void {
    // Optimistic clear through the store. The store toasts on failure
    // and restores the row; the parent receives null immediately so the
    // badge updates this tick.
    this.close();
    this.data.onSaved?.(null);
    this.toast.success('Source reset to HF Hub default');
    void this.registryStore.clearOverride(this.definitionId);
  }

  protected close(): void {
    this.overlay.closeModal();
  }
}
