import { Component, input, output, inject, signal, computed, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DatasetService } from '../../../services/dataset';
import { ToastService } from '../../../services/toast';

export interface TrainingTemplate {
  id: string;
  name: string;
  definition_id: string;
  is_default?: boolean;
  config: any;
}

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
                    [disabled]="activeTemplateId() === 'default'" [class.opacity-50]="activeTemplateId() === 'default'" class="p-1.5 bg-surface-mid hover:bg-surface-high text-yellow-500 rounded-theme-md border border-surface-high transition-colors" title="Rename Template">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
            </button>
            <button type="button" (click)="deleteTemplate()" 
                    data-testid="delete-training-template-btn"
                    [disabled]="activeTemplateId() === 'default'" [class.opacity-50]="activeTemplateId() === 'default'" class="p-1.5 bg-surface-mid hover:bg-danger/20 text-danger rounded-theme-md border border-surface-high transition-colors" title="Delete Template">
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

  // Outputs to Main Orchestrator
  templateApplied = output<{ config: any, isDefault: boolean, definitionId?: string }>();

  private datasetService = inject(DatasetService);
  private toast = inject(ToastService);

  allTemplates = signal<TrainingTemplate[]>([]);
  activeTemplateId = signal<string>('default');

  // Internal flag to prevent recursive auto-saving when applying a template
  suppressAutoSave = signal<boolean>(false);

  filteredTemplates = computed(() => {
    const all = this.allTemplates();
    const defId = this.selectedDefinitionId();
    const defaultEntry: TrainingTemplate = {
      id: 'default', name: 'Default', definition_id: defId ?? '', is_default: true, config: {}
    };
    const sorted = [...all].sort((a, b) => a.name.localeCompare(b.name));
    return [defaultEntry, ...sorted];
  });

  ngOnInit() {
    this.loadTrainingSettings();
  }

  loadTrainingSettings() {
    this.datasetService.getSettings('training').subscribe({
      next: (settings: any) => {
        if (settings && settings.templates && Array.isArray(settings.templates)) {
          this.allTemplates.set(settings.templates);
        }
      },
      error: (err: any) => console.error('[Templates] Failed to load training settings', err)
    });
  }

  saveTrainingSettings() {
    this.datasetService.saveSettings('training', { templates: this.allTemplates() }).subscribe({
      error: (err: any) => console.error('[Templates] Failed to save training settings:', err)
    });
  }

  getDefinitionLabel(definitionId: string): string {
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

    const newTpl: TrainingTemplate = {
      id: `tpl_${Date.now()}`,
      name: name,
      definition_id: defId,
      is_default: false,
      config: this.currentFormConfig()
    };

    this.allTemplates.update(current => [...current, newTpl]);
    this.activeTemplateId.set(newTpl.id);
    this.saveTrainingSettings();
    this.toast.success('Template cloned!');
  }

  renameTemplate() {
    const id = this.activeTemplateId();
    if (id === 'default') return;
    const tpl = this.allTemplates().find(t => t.id === id);
    if (!tpl) return;

    const newName = prompt('Rename Template:', tpl.name);
    if (!newName || newName === tpl.name) return;

    this.allTemplates.update(current => current.map(t => t.id === id ? { ...t, name: newName } : t));
    this.saveTrainingSettings();
    this.toast.success('Template renamed!');
  }

  deleteTemplate() {
    const id = this.activeTemplateId();
    if (id === 'default') return;
    if (!confirm("Delete current template?")) return;

    this.allTemplates.update(current => current.filter(t => t.id !== id));
    this.activeTemplateId.set('default');
    this.saveTrainingSettings();
    this.toast.success('Template deleted!');
  }

  public triggerAutoSave(newFormValue: any, currentDefId: string) {
    if (this.suppressAutoSave()) return;

    const id = this.activeTemplateId();

    if (id === 'default') {
      const newTpl: TrainingTemplate = {
        id: `tpl_${Date.now()}`,
        name: 'Default by User',
        definition_id: currentDefId,
        is_default: false,
        config: newFormValue
      };
      this.allTemplates.update(current => [...current, newTpl]);
      this.activeTemplateId.set(newTpl.id);
      this.saveTrainingSettings();
      return;
    }

    this.allTemplates.update(current => current.map(t => {
      if (t.id === id) {
        return { ...t, definition_id: currentDefId, config: newFormValue };
      }
      return t;
    }));
    this.saveTrainingSettings();
  }

  public importExternalTemplate(name: string, config: any, definitionId: string) {
    const newTpl: TrainingTemplate = {
      id: `tpl_${Date.now()}`,
      name,
      definition_id: definitionId,
      is_default: false,
      config
    };
    this.allTemplates.update(current => [...current, newTpl]);
    this.activeTemplateId.set(newTpl.id);
    this.saveTrainingSettings();
  }
}

