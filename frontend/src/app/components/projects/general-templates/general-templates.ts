import { Component, inject, input, OnInit, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { TemplateService } from '../../../services/template.service';
import { ToastService } from '../../../services/toast';

@Component({
  selector: 'app-general-templates',
  standalone: true,
  imports: [FormsModule],
  template: `
    <div class="space-y-4">
      <div class="flex gap-4 mb-4">
        <button [class.text-white]="activeTab() === 'training'"
                [class.border-brand]="activeTab() === 'training'"
                [class.text-text-muted]="activeTab() !== 'training'"
                [class.border-transparent]="activeTab() !== 'training'"
                class="pb-2 border-b-2 font-medium transition-colors"
                (click)="loadTemplates('training')">Training</button>
        <button [class.text-white]="activeTab() === 'captioning'"
                [class.border-brand]="activeTab() === 'captioning'"
                [class.text-text-muted]="activeTab() !== 'captioning'"
                [class.border-transparent]="activeTab() !== 'captioning'"
                class="pb-2 border-b-2 font-medium transition-colors"
                (click)="loadTemplates('captioning')">Captioning</button>
        <button [class.text-white]="activeTab() === 'masking'"
                [class.border-brand]="activeTab() === 'masking'"
                [class.text-text-muted]="activeTab() !== 'masking'"
                [class.border-transparent]="activeTab() !== 'masking'"
                class="pb-2 border-b-2 font-medium transition-colors"
                (click)="loadTemplates('masking')">Masking</button>
      </div>

      <div class="space-y-2 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
        @for (tpl of templates(); track tpl.id) {
          <div class="flex items-center justify-between p-3 bg-surface-mid border border-surface-high rounded-theme-md hover:border-brand/30 transition-colors">
            <div class="min-w-0 flex-1">
              <div class="flex items-center gap-2 flex-wrap">
                <span class="font-medium text-white truncate">{{ tpl.name }}</span>
                @if (tpl.is_default) {
                  <span class="text-xs bg-brand/20 text-brand-light px-2 py-0.5 rounded-full shrink-0">System</span>
                }
                <!-- Scope badge -->
                @if (tpl.project_id) {
                  <span class="text-xs bg-emerald-500/15 text-emerald-400 px-2 py-0.5 rounded-full shrink-0">📁 Project</span>
                } @else {
                  <span class="text-xs bg-sky-500/15 text-sky-400 px-2 py-0.5 rounded-full shrink-0">🌐 Global</span>
                }
              </div>
              <!-- Model / Definition info -->
              <div class="text-xs text-text-subtle mt-1 flex items-center gap-2">
                <span class="text-ellipsis overflow-hidden whitespace-nowrap max-w-[180px]">
                  {{ tpl.definition_id || tpl.model_id || 'All Models' }}
                </span>
                <!-- Lineage badge -->
                @if (tpl.branched_from) {
                  <span class="text-amber-400/80 shrink-0" title="Branched from template {{ tpl.branched_from }}">
                    ↳ branched
                  </span>
                }
              </div>
            </div>
            <div class="flex items-center gap-2 shrink-0 ml-2">
              @if (!tpl.is_default && !tpl.readonly) {
                <button (click)="deleteTemplate(tpl)"
                        class="text-sm bg-surface-high hover:bg-danger/20 text-danger border border-border-default px-2 py-1.5 rounded-theme-md transition-colors"
                        title="Delete this template permanently">
                  <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                </button>
              }
              <button (click)="branchTemplate(tpl)" 
                      class="text-sm bg-surface-high hover:bg-brand hover:text-white border border-border-default px-3 py-1.5 rounded-theme-md transition-colors"
                      title="Branch this template into your project">
                Branch
              </button>
            </div>
          </div>
        } @empty {
          <div class="text-center text-text-subtle p-4 bg-surface-high rounded-theme-md">
            No global templates found for this domain.
          </div>
        }
      </div>
    </div>
  `
})
export class GeneralTemplatesComponent implements OnInit {
  private templateService = inject(TemplateService);
  private toast = inject(ToastService);

  projectId = input.required<string>();

  activeTab = signal<'training' | 'captioning' | 'masking'>('training');
  templates = signal<any[]>([]);

  ngOnInit() {
    this.loadTemplates('training');
  }

  loadTemplates(type: 'training' | 'captioning' | 'masking') {
    this.activeTab.set(type);
    
    // Pass undefined for model_id/definition_id to get ALL templates for this domain.
    // Pass undefined for project_id to get only General (global) templates.
    const req = type === 'training' 
      ? this.templateService.listTrainingTemplates(undefined, undefined)
      : type === 'captioning'
        ? this.templateService.listCaptioningTemplates(undefined, undefined)
        : this.templateService.listMaskingTemplates(undefined, undefined);

    req.subscribe((res: any) => {
      // Filter to only General (global) templates — no project_id
      this.templates.set(res.filter((t: any) => !t.project_id));
    });
  }

  branchTemplate(template: any) {
    const domain = this.activeTab();
    this.templateService.branchTemplate(domain, template.id, this.projectId()).subscribe({
      next: () => {
        this.toast.success(`Branched template '${template.name}' into project.`);
      },
      error: (err: any) => {
        this.toast.error(`Failed to branch template: ${err.message}`);
      }
    });
  }

  deleteTemplate(template: any) {
    if (!confirm(`Delete global template '${template.name}'? This cannot be undone.`)) return;
    const domain = this.activeTab();
    this.templateService.deleteTemplate(domain, template.id).subscribe({
      next: () => {
        this.templates.update(current => current.filter(t => t.id !== template.id));
        this.toast.success(`Deleted template '${template.name}'.`);
      },
      error: (err: any) => {
        this.toast.error(`Failed to delete template: ${err.message}`);
      }
    });
  }
}
