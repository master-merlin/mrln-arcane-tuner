import { Component, OnInit, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ProjectService } from '../../../services/project.service';
import { ProjectDetailComponent } from '../project-detail/project-detail';
import { ProjectDialogComponent } from '../project-dialog/project-dialog';

@Component({
  selector: 'app-projects-view',
  standalone: true,
  imports: [FormsModule, ProjectDetailComponent, ProjectDialogComponent],
  template: `
    <div class="space-y-8 animate-in fade-in duration-300">
      
      @if (selectedProjectId()) {
          <app-project-detail [projectId]="selectedProjectId()!" (back)="selectedProjectId.set(null)"></app-project-detail>
      } @else {
          <!-- Header -->
          <div class="bg-surface-low/50 border border-border-default rounded-theme-xl p-8 shadow-xl">
              <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
              <div>
                  <h2 class="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-brand">
                          <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
                      </svg>
                      Projects
                  </h2>
                  <p class="text-text-muted">Manage your projects, organize datasets, and maintain localized templates.</p>
              </div>
              <button (click)="openNewProjectDialog()" 
                  class="flex items-center gap-2 bg-brand/20 hover:bg-brand/30 border border-brand/40 text-brand-light px-4 py-2 rounded-theme-lg transition-all h-10">
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                  New Project
              </button>
          </div>
      </div>

      <!-- Projects Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        @for (project of projectService.allProjects(); track project.id) {
            <div class="bg-surface-low border border-surface-mid hover:border-brand/40 rounded-theme-xl p-6 shadow-lg transition-all cursor-pointer group"
                 (click)="viewProject(project.id)">
                <div class="flex items-start justify-between mb-4">
                    <div class="flex flex-col">
                        <div class="flex items-center gap-3 mb-1">
                            <div class="w-3 h-3 rounded-full" [style.backgroundColor]="project.color || '#3b82f6'"></div>
                            <h3 class="text-lg font-bold text-white group-hover:text-brand-light transition-colors">{{ project.name }}</h3>
                        </div>
                        <p class="text-sm text-text-muted line-clamp-2">{{ project.description || 'No description provided.' }}</p>
                    </div>
                </div>
                
                <div class="mt-4 pt-4 border-t border-surface-mid flex justify-between items-center text-sm">
                    <span class="text-text-subtle group-hover:text-text-muted transition-colors">
                        Updated {{ formatDate(project.updated_at) }}
                    </span>
                    <div class="flex items-center gap-2">
                        <button class="text-text-muted hover:text-brand-light transition-colors p-1" 
                                (click)="editProject(project.id, $event)" title="Edit Project">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
                        </button>
                        <button class="text-text-muted hover:text-danger transition-colors p-1" 
                                (click)="$event.stopPropagation(); deleteProject(project.id)" title="Delete Project">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"></path><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path></svg>
                        </button>
                    </div>
                </div>
            </div>
        } @empty {
            <div class="col-span-full p-12 text-center text-text-subtle bg-surface-low/30 rounded-theme-xl border border-border-default border-dashed">
                <p>No projects found. Create one to get started.</p>
            </div>
        }
      </div>
      }

      @if (showDialog()) {
        <app-project-dialog
          [projectId]="editingProjectId()"
          (close)="closeDialog()">
        </app-project-dialog>
      }
    </div>
  `
})
export class ProjectsViewComponent implements OnInit {
  protected projectService = inject(ProjectService);
  selectedProjectId = signal<string | null>(null);
  
  // Dialog state
  showDialog = signal(false);
  editingProjectId = signal<string | null>(null);

  ngOnInit() {
    this.projectService.loadProjects();
  }

  openNewProjectDialog() {
    this.editingProjectId.set(null);
    this.showDialog.set(true);
  }

  editProject(id: string, event: Event) {
    event.stopPropagation();
    this.editingProjectId.set(id);
    this.showDialog.set(true);
  }

  closeDialog() {
    this.showDialog.set(false);
    this.editingProjectId.set(null);
  }

  viewProject(id: string) {
    this.selectedProjectId.set(id);
  }

  deleteProject(id: string) {
    if (!confirm('Are you sure you want to delete this project? This will not delete your datasets or images, but will remove project-specific settings.')) return;
    this.projectService.deleteProject(id).subscribe(() => {
      this.projectService.loadProjects();
    });
  }

  formatDate(timestamp?: number) {
    if (!timestamp) return 'Never';
    return new Date(timestamp * 1000).toLocaleDateString();
  }
}
