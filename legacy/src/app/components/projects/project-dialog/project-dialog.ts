import { Component, output, input, inject, signal, effect, computed } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ProjectService } from '../../../services/project.service';
import { ToastService } from '../../../services/toast';

@Component({
  selector: 'app-project-dialog',
  standalone: true,
  imports: [FormsModule],
  template: `
    <div class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-overlay backdrop-blur-sm animate-in fade-in duration-200"
         (click)="close.emit()">
      <div class="bg-surface-low border border-border-default rounded-theme-xl shadow-2xl w-full max-w-lg overflow-hidden animate-in zoom-in-95 duration-200"
           (click)="$event.stopPropagation()">
        
        <!-- Header -->
        <div class="px-6 py-4 flex items-center justify-between border-b border-surface-mid">
          <h3 class="text-xl font-bold text-white">{{ isEdit() ? 'Edit Project' : 'New Project' }}</h3>
          <button (click)="close.emit()" class="text-text-muted hover:text-white transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
          </button>
        </div>

        <!-- Body -->
        <div class="p-6 space-y-6">
          <div>
            <label class="block text-sm font-medium text-white mb-2">Project Name <span class="text-brand">*</span></label>
            <input type="text" [(ngModel)]="name"
                   placeholder="e.g. My Awesome LoRA"
                   class="w-full bg-surface-mid border border-surface-high hover:border-border-default focus:border-brand rounded-theme-lg px-4 py-2.5 text-white placeholder:text-text-subtle transition-colors focus:outline-none focus:ring-1 focus:ring-brand">
          </div>

          <div>
            <label class="block text-sm font-medium text-white mb-2">Description</label>
            <textarea [(ngModel)]="description"
                      rows="3"
                      placeholder="What is this project about?"
                      class="w-full bg-surface-mid border border-surface-high hover:border-border-default focus:border-brand rounded-theme-lg px-4 py-2.5 text-white placeholder:text-text-subtle transition-colors focus:outline-none focus:ring-1 focus:ring-brand"></textarea>
          </div>

          <div>
            <label class="block text-sm font-medium text-white mb-2">Project Color</label>
            <div class="flex items-center gap-4">
              <input type="color" [(ngModel)]="color"
                     class="w-12 h-12 rounded-theme-md cursor-pointer border-0 bg-transparent p-0">
              <div class="flex gap-2">
                @for (preset of colorPresets; track preset) {
                  <button (click)="color.set(preset)"
                          class="w-8 h-8 rounded-full border-2 transition-transform hover:scale-110"
                          [class.border-white]="color() === preset"
                          [class.border-transparent]="color() !== preset"
                          [style.backgroundColor]="preset"></button>
                }
              </div>
            </div>
          </div>
        </div>

        <!-- Footer -->
        <div class="px-6 py-4 bg-surface-mid border-t border-surface-high flex items-center justify-end gap-3">
          <button (click)="close.emit()" class="px-4 py-2 text-text-muted hover:text-white transition-colors text-sm font-medium">
            Cancel
          </button>
          <button (click)="save()"
                  [disabled]="!name().trim() || isSaving()"
                  class="bg-brand/90 hover:bg-brand text-white px-6 py-2 rounded-theme-lg transition-all font-medium flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed">
            @if (isSaving()) {
              <div class="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin"></div>
              Saving...
            } @else {
              {{ isEdit() ? 'Save Changes' : 'Create Project' }}
            }
          </button>
        </div>

      </div>
    </div>
  `
})
export class ProjectDialogComponent {
  private projectService = inject(ProjectService);
  private toast = inject(ToastService);

  // Inputs
  projectId = input<string | null>(null);
  
  // Outputs
  close = output<void>();
  saved = output<void>();

  // State
  name = signal('');
  description = signal('');
  color = signal('#6366f1');
  
  isSaving = signal(false);
  isEdit = computed(() => !!this.projectId());

  colorPresets = [
    '#ef4444', '#f97316', '#f59e0b', '#84cc16', 
    '#10b981', '#06b6d4', '#3b82f6', '#8b5cf6', 
    '#d946ef', '#f43f5e'
  ];

  constructor() {
    effect(() => {
      const pid = this.projectId();
      if (pid) {
        // Find existing project in service state and populate fields
        const proj = this.projectService.allProjects().find(p => p.id === pid);
        if (proj) {
          this.name.set(proj.name);
          this.description.set(proj.description || '');
          this.color.set(proj.color || '#3b82f6');
        }
      } else {
        this.name.set('');
        this.description.set('');
        this.color.set('#6366f1');
      }
    });
  }

  save() {
    if (!this.name().trim()) return;
    
    this.isSaving.set(true);
    
    if (this.isEdit()) {
      this.projectService.updateProject(this.projectId()!, {
        name: this.name().trim(),
        description: this.description().trim(),
        color: this.color()
      }).subscribe({
        next: () => {
          this.toast.success('Project updated.');
          this.projectService.loadProjects();
          this.saved.emit();
          this.isSaving.set(false);
          this.close.emit();
        },
        error: (err: any) => {
          this.toast.error(`Error updating project: ${err.message}`);
          this.isSaving.set(false);
        }
      });
    } else {
      this.projectService.createProject(
        this.name().trim(),
        this.description().trim(),
        this.color()
      ).subscribe({
        next: () => {
          this.toast.success('Project created.');
          this.projectService.loadProjects();
          this.saved.emit();
          this.isSaving.set(false);
          this.close.emit();
        },
        error: (err: any) => {
          this.toast.error(`Error creating project: ${err.message}`);
          this.isSaving.set(false);
        }
      });
    }
  }
}
