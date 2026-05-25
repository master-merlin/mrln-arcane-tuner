import { Component, input, output, inject, signal, computed, OnInit, effect } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ToastService } from '../../../services/toast';
import { TemplateService, Template } from '../../../services/template.service';

@Component({
  selector: 'app-training-template-selector',
  standalone: true,
  imports: [FormsModule],
  template: `
    <div class="space-y-6 mb-4">
      <div class="flex items-center justify-between border-b border-surface-mid/30 pb-2 mb-4">
          <div class="flex items-center gap-4">
            <div class="w-1 h-6 bg-brand rounded-full"></div>
            <h3 class="text-sm font-black text-text-subtle uppercase tracking-[0.2em]">Template Selection</h3>
          </div>
      </div>

    <!-- Template Header & Actions -->
    <div class="bg-surface-high/40 rounded-theme-lg border border-surface-mid/50 overflow-hidden mb-4">
        <div class="p-3 bg-surface-low/50 border-b border-surface-mid/50 flex items-end gap-2">
            <div class="flex-1">
                <label class="text-[10px] uppercase tracking-wider text-text-subtle font-bold mb-1 block">Settings Template</label>
                <select [ngModel]="activeTemplateId()" (ngModelChange)="applyTemplate($event)"
                        data-testid="training-template-select"
                    class="w-full bg-surface-low border border-surface-high text-white text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand transition-colors">
                    @for (tpl of filteredTemplates(); track tpl.id) {
                        <option [value]="tpl.id">{{ tpl.name }}{{ tpl.is_default ? ' (Default)' : '' }}{{ !tpl.is_default ? getDefinitionLabel(tpl.definition_id) : '' }}</option>
                    }
                </select>
            </div>
            
            <!-- Actions -->
            <button type="button" (click)="saveAsNewTemplate()" 
                    data-testid="add-training-template-btn"
                    class="p-1.5 bg-surface-mid hover:bg-surface-high text-brand rounded-theme-md border border-surface-high transition-colors" title="Clone as New Template">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>
            </button>
            <button type="button" (click)="renameTemplate()" 
                    data-testid="rename-training-template-btn"
                    [disabled]="isDefaultTemplate()" [class.opacity-50]="isDefaultTemplate()" class="p-1.5 bg-surface-mid hover:bg-surface-high text-yellow-500 rounded-theme-md border border-surface-high transition-colors" title="Rename Template">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
            </button>
            <button type="button" (click)="deleteTemplate()" 
                    data-testid="delete-training-template-btn"
                    [disabled]="isDefaultTemplate()" [class.opacity-50]="isDefaultTemplate()" class="p-1.5 bg-surface-mid hover:bg-danger/20 text-danger rounded-theme-md border border-surface-high transition-colors" title="Delete Template">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
            </button>
        </div>
    </div>
    </div>
  `
})
export class TrainingTemplateSelectorComponent implements OnInit {
  // Inputs from Main Orchestrator
  availableModels = input<any[]>([]);
  selectedDefinitionId = input<string | null>(null);
  currentFormConfig = input<any>({});
  projectId = input<string | null>(null);

  // Outputs to Main Orchestrator
  templateApplied = output<{ config: any, isDefault: boolean, definitionId?: string }>();

  private templateService = inject(TemplateService);
  private toast = inject(ToastService);

  allTemplates = signal<Template[]>([]);
  activeTemplateId = signal<string>('default');

  // Internal flag to prevent recursive auto-saving when applying a template
  suppressAutoSave = signal<boolean>(false);

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
    this.templateService.listTrainingTemplates(undefined, this.projectId()).subscribe({
      next: (templates) => {
        this.allTemplates.set(templates);
      },
      error: (err: any) => console.error('[Templates] Failed to load training templates', err)
    });
  }

  getDefinitionLabel(definitionId?: string): string {
    if (!definitionId) return '';
    const model = this.availableModels().find(m => m.id === definitionId);
    return model ? ` · ${model.name}` : '';
  }

  applyTemplate(tplId: string) {
    this.suppressAutoSave.set(true);
    this.activeTemplateId.set(tplId);

    const tpl = this.filteredTemplates().find(t => t.id === tplId);
    if (!tpl) { this.suppressAutoSave.set(false); return; }

    this.templateApplied.emit({
      config: tpl.config,
      isDefault: !!tpl.is_default,
      definitionId: tpl.definition_id
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

  public triggerAutoSave(newFormValue: any, currentDefId: string) {
    if (this.suppressAutoSave()) return;

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

  public importExternalTemplate(name: string, config: any, definitionId: string) {
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

