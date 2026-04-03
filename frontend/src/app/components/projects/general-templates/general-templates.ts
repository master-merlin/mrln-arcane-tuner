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
            <div>
              <div class="flex items-center gap-2">
                <span class="font-medium text-white">{{ tpl.name }}</span>
                @if (tpl.is_default) {
                  <span class="text-xs bg-brand/20 text-brand-light px-2 py-0.5 rounded-full">System</span>
                }
              </div>
              <div class="text-xs text-text-subtle mt-1 text-ellipsis overflow-hidden whitespace-nowrap max-w-[250px]">
                {{ tpl.model_id || tpl.definition_id || 'All Models' }}
              </div>
            </div>
            <button (click)="branchTemplate(tpl)" 
                    class="text-sm bg-surface-high hover:bg-brand hover:text-white border border-border-default px-3 py-1.5 rounded-theme-md transition-colors"
                    title="Branch this template into your project">
              Branch
            </button>
          </div>
        } @empty {
          <div class="text-center text-text-subtle p-4 bg-surface-high rounded-theme-md">
            No global templates found.
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
    
    // We pass undefined for projectId to get only General (global) templates
    const req = type === 'training' 
      ? this.templateService.listTrainingTemplates(undefined, undefined) // get all
      : type === 'captioning'
        ? this.templateService.listCaptioningTemplates(undefined, undefined)
        : this.templateService.listMaskingTemplates(undefined, undefined);

    req.subscribe((res: any) => {
      // Filter out project-specific templates just in case, though passing null should handle it
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
}
