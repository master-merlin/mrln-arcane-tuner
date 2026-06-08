import { Component, ChangeDetectionStrategy, input, output, inject, signal, computed, OnInit, effect } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { forkJoin, of } from 'rxjs';
import { catchError, switchMap } from 'rxjs/operators';
import { ToastService } from '../../../services/toast';
import { TemplateService, Template } from '../../../services/template.service';
import { ProjectService } from '../../../services/project.service';
import { OverlayStore } from '../../../state/overlay.store';
import { TemplateInfoCardComponent, TemplateInfoRow } from '../../../ui/template-info-card/template-info-card.component';
import type { TrainingConfig } from '../../../services/job';
import type { ModelDefinition } from '../../../screens/training-screen/training-screen';

/** training_selections key under which the active training template id is
 *  persisted, so a reload returns to the exact template the user was editing
 *  (instead of falling back to the first one in the list). */
const ACTIVE_TPL_PREF_KEY = 'active_training_template';

@Component({
  selector: 'app-training-template-selector',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FormsModule, TemplateInfoCardComponent],
  template: `
    <section class="card mb-3.5">
      <!-- Header: accent + title + summary + active-template chip -->
      <div class="card-head">
        <div class="card-title min-w-0" style="padding:0">
          <span class="w-[3px] h-3.5 bg-brand rounded-sm shrink-0"></span>
          <span class="shrink-0">Template Selection</span>
          <span class="ml-2.5 normal-case tracking-normal text-[12px] font-medium text-text-secondary truncate">apply a saved configuration · {{ filteredTemplates().length }} available</span>
        </div>
        @if (activeTemplate(); as tpl) {
          <span class="chip shrink-0">
            <span class="dot bg-brand"></span>
            {{ tpl.name }} · {{ (tpl.is_default || tpl.readonly) ? 'default' : 'custom' }}
          </span>
        }
      </div>

      <!-- Body: template picker + actions on one row, meta below -->
      <div class="card-body">
        <label class="field-label">Settings template</label>
        <div class="flex items-center gap-2">
          <select [ngModel]="activeTemplateId()" (ngModelChange)="applyTemplate($event)"
                  data-testid="training-template-select" class="select flex-1 min-w-0">
            @for (tpl of filteredTemplates(); track tpl.id) {
              <option [value]="tpl.id">{{ tpl.name }}{{ tpl.is_default ? ' (Default)' : '' }}{{ !tpl.is_default ? getDefinitionLabel(tpl.definition_id) : '' }}</option>
            }
          </select>

          <!-- Actions -->
          <button type="button" (click)="saveAsNewTemplate()"
                  data-testid="add-training-template-btn"
                  class="icon-btn text-brand shrink-0" title="Clone as New Template">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>
          </button>
          <button type="button" (click)="renameTemplate()"
                  data-testid="rename-training-template-btn"
                  [disabled]="isDefaultTemplate()" [class.opacity-40]="isDefaultTemplate()"
                  class="icon-btn text-warning shrink-0" title="Rename Template">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
          </button>
          <button type="button" (click)="deleteTemplate()"
                  data-testid="delete-training-template-btn"
                  [disabled]="isDefaultTemplate()" [class.opacity-40]="isDefaultTemplate()"
                  class="icon-btn text-danger shrink-0" title="Delete Template">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
          </button>
          <button type="button" (click)="exportCurrentTemplate()"
                  data-testid="export-training-template-btn"
                  [disabled]="isDefaultTemplate()" [class.opacity-40]="isDefaultTemplate()"
                  class="icon-btn shrink-0" title="Export current template">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 15V3"></path></svg>
          </button>
          <button type="button" (click)="importTemplate()"
                  data-testid="import-training-template-btn"
                  class="icon-btn shrink-0" title="Import template / project / dataset">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 8l5-5 5 5M12 3v12"></path></svg>
          </button>
        </div>
        @if (activeTemplate(); as tpl) {
          <app-template-info-card class="block mt-2.5" [name]="tpl.name" [rows]="activeTemplateInfo()"></app-template-info-card>
        }
      </div>
    </section>
  `
})
export class TrainingTemplateSelectorComponent implements OnInit {
  // Inputs from Main Orchestrator
  availableModels = input<ModelDefinition[]>([]);
  selectedDefinitionId = input<string | null>(null);
  currentFormConfig = input<TrainingConfig>({});
  projectId = input<string | null>(null);

  // Outputs to Main Orchestrator. `auto` flags the one-time apply fired on
  // load (vs a user dropdown selection) so the parent can yield to a handoff.
  templateApplied = output<{ config: TrainingConfig, isDefault: boolean, definitionId?: string, auto?: boolean }>();

  private templateService = inject(TemplateService);
  private projects = inject(ProjectService);
  private toast = inject(ToastService);
  private overlay = inject(OverlayStore);

  allTemplates = signal<Template[]>([]);
  activeTemplateId = signal<string>('default');

  /** Persisted active-template id for this project (from preferences), used by
   *  _maybeAutoApply to restore the user's selection across a reload. */
  private _preferredActiveId: string | null = null;

  // Internal flag to prevent recursive auto-saving when applying a template
  suppressAutoSave = signal<boolean>(false);

  // Project context for which the initial template has already been auto-applied
  // (`undefined` = never). Dedupes the duplicate load calls on startup and
  // re-arms the one-time apply when the project changes.
  private _autoAppliedForProject: string | null | undefined = undefined;

  // True once a template list has finished loading at least once, so an
  // `adoptExternalTemplate` call that arrives before/after the load both work.
  private _loaded = false;
  // A pending "adopt this existing template by id" request (from a Projects
  // "Edit" action or a job reload), applied as soon as templates are loaded.
  private _pendingAdopt: { id: string; name: string; config: TrainingConfig; definitionId: string } | null = null;

  filteredTemplates = computed(() => {
    const all = this.allTemplates();
    const defId = this.selectedDefinitionId();

    // Ensure there is always a default entry if none exists from the backed
    const hasDefault = all.some(t => t.is_default);
    
    let sorted = [...all].sort((a, b) => a.name.localeCompare(b.name));
    
    if (!hasDefault) {
       const defaultEntry: Template = {
          id: 'default', name: 'Default', definition_id: defId ?? '', is_default: true, config: {},
          project_id: null, created_at: Date.now(), updated_at: Date.now(), used_count: 0, readonly: true
       };
       sorted = [defaultEntry, ...sorted];
    }
    
    return sorted;
  });

  isDefaultTemplate() {
      const id = this.activeTemplateId();
      if (id === 'default') return true;
      const tpl = this.allTemplates().find(t => t.id === id);
      return tpl ? tpl.is_default || tpl.readonly : false;
  }

  /** The currently-selected template (for the header chip + meta line). */
  activeTemplate = computed<Template | undefined>(() =>
    this.filteredTemplates().find(t => t.id === this.activeTemplateId()));

  /**
   * Key/value rows for the active template's info card — derived from its saved
   * config (model name resolved via availableModels, falling back to the raw
   * definition_id). Only present fields are emitted, so the bare Default
   * template (no config yet) shows just its name. Mirrors the Projects →
   * Quick Train `selectedTemplateInfo` builder so the two cards read alike.
   */
  activeTemplateInfo = computed<TemplateInfoRow[]>(() => {
    const tpl = this.activeTemplate();
    if (!tpl) return [];
    const cfg = (tpl.config ?? {}) as Record<string, unknown>;
    const rows: TemplateInfoRow[] = [];
    const push = (key: string, value: unknown, fmt?: (v: unknown) => string) => {
      if (value === undefined || value === null || value === '') return;
      rows.push({ key, value: fmt ? fmt(value) : String(value) });
    };
    const defId = (cfg['definition_id'] as string) || tpl.definition_id;
    const model = this.availableModels().find(m => m.id === defId);
    push('Base model', model?.name || defId);
    push('Training steps', cfg['max_train_steps']);
    push('Epochs', cfg['max_train_epochs']);
    push('Optimizer', cfg['optimizer_type']);
    push('Learning rate', cfg['learning_rate'], v => this.formatLr(v));
    push('Batch size', cfg['train_batch_size']);
    push('Network rank', cfg['network_rank'] ?? cfg['network_dim'] ?? cfg['lora_rank'] ?? cfg['rank']);
    push('Network alpha', cfg['network_alpha']);
    push('Resolution', cfg['resolution']);
    push('Scheduler', cfg['lr_scheduler']);
    push('Timestep sampling', cfg['timestep_sampling']);
    return rows;
  });

  /** Compact LR rendering: scientific notation for the usual tiny values. */
  private formatLr(lr: unknown): string {
    const n = Number(lr);
    if (Number.isNaN(n) || n === 0) return String(lr);
    return n < 0.0001 ? n.toExponential(1) : n.toString();
  }

  constructor() {
    effect(() => {
       const pId = this.projectId(); // Read signal to trigger re-execution on project change
       this.loadTrainingSettings();
    });
  }

  ngOnInit() {
    this.loadTrainingSettings();
  }

  loadTrainingSettings() {
    // Load ALL templates for this project context (no definition_id filter).
    // When a template is applied, the parent component handles model switching.
    // Preferences are fetched alongside so we can restore the active template
    // the user last had selected (survives a reload); a prefs failure is
    // non-fatal — we just fall back to the default resolution.
    forkJoin({
      templates: this.templateService.listTrainingTemplates(undefined, this.projectId()),
      prefs: this.projects.getPreferences(this.projectId()).pipe(catchError(() => of(null))),
    }).subscribe({
      next: ({ templates, prefs }) => {
        this.allTemplates.set(templates);
        const sel = (prefs?.training_selections ?? {}) as Record<string, unknown>;
        const pref = sel[ACTIVE_TPL_PREF_KEY];
        this._preferredActiveId = typeof pref === 'string' ? pref : null;
        this._loaded = true;
        // A pending external adopt (edit-in-place / job reload) wins over the
        // generic one-time auto-apply so we never clobber the chosen template.
        if (this._pendingAdopt) this._applyPendingAdopt();
        else this._maybeAutoApply();
      },
      error: (err: unknown) => console.error('[Templates] Failed to load training templates', err)
    });
  }

  /**
   * Select an EXISTING training template by id as the active save-target —
   * used when editing a template from Projects (Bug A) or reloading a job into
   * Training (Bug B). Edits then auto-save to this template instead of spawning
   * a duplicate. If the id no longer exists (template deleted after the job was
   * created), it is recreated once from the supplied name + config so there is
   * still a real, project-scoped row to edit. Deliberately does NOT patch the
   * form — the caller applies the handed-off config, which for a job reload is
   * the job's actual run config (it may differ from the template).
   */
  public adoptExternalTemplate(id: string, name: string, config: TrainingConfig, definitionId: string): void {
    this._pendingAdopt = { id, name, config, definitionId };
    // Claim the one-time auto-apply slot so _maybeAutoApply doesn't fight us.
    this._autoAppliedForProject = this.projectId();
    if (this._loaded) this._applyPendingAdopt();
  }

  private _applyPendingAdopt(): void {
    const p = this._pendingAdopt;
    if (!p) return;
    const existing = this.allTemplates().find(t => t.id === p.id);
    if (existing) {
      this._pendingAdopt = null;
      this._setActiveTemplateQuietly(existing.id);
    } else {
      // Template was deleted after the job ran — recreate it from the snapshot.
      this.templateService.createTrainingTemplate({
        definition_id: p.definitionId,
        name: p.name,
        project_id: this.projectId(),
        config: p.config,
      }).subscribe(newTpl => {
        this.allTemplates.update(current => [...current, newTpl]);
        this._pendingAdopt = null;
        this._setActiveTemplateQuietly(newTpl.id);
        this.toast.success(`Template "${p.name}" was missing — recreated it.`);
      });
    }
  }

  /** Set the active template without triggering an auto-save of the (about to
   *  be patched) form. The caller's own patch settles inside the suppression
   *  window, matching the dynamic-config's loadExternalConfig timing. */
  private _setActiveTemplateQuietly(id: string): void {
    this.suppressAutoSave.set(true);
    this.activeTemplateId.set(id);
    setTimeout(() => this.suppressAutoSave.set(false), 1500);
  }

  /**
   * Apply the initially-active template ONCE per project, on load. The Training
   * screen drives its estimate wall from the live form, so without this the
   * wall reflects bare defaults until the user manually re-picks a template.
   * Resolves the active id to a real, selectable option (the hardcoded
   * `'default'` may not match when a real default template exists) and routes
   * through the same `applyTemplate` path a manual selection uses, tagged
   * `auto` so the parent can defer to a Jobs-screen handoff.
   */
  private _maybeAutoApply(): void {
    const pid = this.projectId();
    if (this._autoAppliedForProject === pid) return;
    const templates = this.filteredTemplates();
    if (templates.length === 0) return;
    this._autoAppliedForProject = pid;
    const current = this.activeTemplateId();
    // Priority: the persisted active template (restores the user's selection
    // across reload) → the current selection if still valid → the first template.
    const preferred = this._preferredActiveId;
    const resolvedId =
      preferred && templates.some(t => t.id === preferred) ? preferred
      : templates.some(t => t.id === current) ? current
      : templates[0].id;
    this.applyTemplate(resolvedId, true);
  }

  /** Persist the active template id into project preferences (merged into the
   *  existing training_selections so other keys — masking concepts, Quick Train
   *  inputs — are preserved). Best-effort; failures are swallowed. */
  private _persistActiveTemplate(tplId: string): void {
    if (!tplId || tplId === 'default') return;
    const pid = this.projectId();
    this.projects.getPreferences(pid).pipe(
      switchMap(prefs => {
        const sel = { ...((prefs?.training_selections ?? {}) as Record<string, unknown>) };
        sel[ACTIVE_TPL_PREF_KEY] = tplId;
        return this.projects.updatePreferences(pid, { training_selections: sel });
      }),
      catchError(() => of(null)),
    ).subscribe();
  }

  getDefinitionLabel(definitionId?: string): string {
    if (!definitionId) return '';
    const model = this.availableModels().find(m => m.id === definitionId);
    return model ? ` · ${model.name}` : '';
  }

  applyTemplate(tplId: string, auto = false) {
    this.suppressAutoSave.set(true);
    this.activeTemplateId.set(tplId);

    const tpl = this.filteredTemplates().find(t => t.id === tplId);
    if (!tpl) { this.suppressAutoSave.set(false); return; }

    // A user-driven selection updates the persisted active template so a later
    // reload restores it. The one-time `auto` apply on load must NOT persist —
    // it's restoring, not choosing.
    if (!auto) this._persistActiveTemplate(tplId);

    this.templateApplied.emit({
      config: tpl.config,
      isDefault: !!tpl.is_default,
      definitionId: tpl.definition_id,
      auto
    });
  }

  saveAsNewTemplate() {
    const defId = this.selectedDefinitionId();
    if (!defId) {
      this.toast.error("Please select a model definition first.");
      return;
    }

    const name = prompt("Template Name:");
    if (!name) return;

    this.templateService.createTrainingTemplate({
        definition_id: defId,
        name: name,
        project_id: this.projectId(),
        config: this.currentFormConfig()
    }).subscribe(newTpl => {
        this.allTemplates.update(current => [...current, newTpl]);
        this.activeTemplateId.set(newTpl.id);
        this.toast.success('Template cloned!');
    });
  }

  renameTemplate() {
    const id = this.activeTemplateId();
    if (this.isDefaultTemplate()) return;
    
    const tpl = this.allTemplates().find(t => t.id === id);
    if (!tpl) return;

    const newName = prompt('Rename Template:', tpl.name);
    if (!newName || newName === tpl.name) return;

    this.templateService.updateTemplate('training', id, { name: newName }).subscribe(updatedTpl => {
        this.allTemplates.update(current => current.map(t => t.id === id ? updatedTpl : t));
        this.toast.success('Template renamed!');
    });
  }

  deleteTemplate() {
    const id = this.activeTemplateId();
    if (this.isDefaultTemplate()) return;
    if (!confirm("Delete current template?")) return;

    this.templateService.deleteTemplate('training', id).subscribe(() => {
        this.allTemplates.update(current => current.filter(t => t.id !== id));
        const remaining = this.filteredTemplates();
        if (remaining.length > 0) {
            this.activeTemplateId.set(remaining[0].id);
        }
        this.toast.success('Template deleted!');
    });
  }

  public triggerAutoSave(newFormValue: TrainingConfig, currentDefId: string) {
    if (this.suppressAutoSave()) return;
    // An external adopt (edit-in-place / job reload) is resolving — don't let a
    // form patch save against the previously-active id and spawn a copy.
    if (this._pendingAdopt) return;

    const id = this.activeTemplateId();

    const tpl = this.allTemplates().find(t => t.id === id);
    
    if (id === 'default' || (tpl && tpl.readonly)) {
      this.templateService.createTrainingTemplate({
          definition_id: currentDefId,
          name: 'Default by User',
          project_id: this.projectId(),
          config: newFormValue
      }).subscribe(newTpl => {
          this.allTemplates.update(current => [...current, newTpl]);
          this.activeTemplateId.set(newTpl.id);
      });
      return;
    }

    // Pending auto-save logic
    this.templateService.updateTemplate('training', id, { definition_id: currentDefId, config: newFormValue }).subscribe(updatedTpl => {
        this.allTemplates.update(current => current.map(t => t.id === id ? updatedTpl : t));
    });
  }

  /** Export the active training template as a `.template.zip` (body-less GET). */
  exportCurrentTemplate(): void {
    const id = this.activeTemplateId();
    if (!id || id === 'default' || this.isDefaultTemplate()) {
      this.toast.warning('Save the template before exporting.');
      return;
    }
    window.open(this.templateService.getTemplateExportUrl('training', id), '_blank');
  }

  /** Open the generic import wizard (routes by the dropped archive's kind). */
  importTemplate(): void {
    this.overlay.openModal('import-archive', {
      projectId: this.projectId() ?? undefined,
      onImported: () => this.loadTrainingSettings(),
    });
  }

  public importExternalTemplate(name: string, config: TrainingConfig, definitionId: string) {
    this.templateService.createTrainingTemplate({
        definition_id: definitionId,
        name: name,
        project_id: this.projectId(),
        config: config
    }).subscribe(newTpl => {
        this.allTemplates.update(current => [...current, newTpl]);
        this.activeTemplateId.set(newTpl.id);
    });
  }
}

