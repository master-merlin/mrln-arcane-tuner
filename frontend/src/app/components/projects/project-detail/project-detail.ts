import { Component, OnInit, inject, input, output, computed, effect } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ProjectService, Project } from '../../../services/project.service';
import { GeneralTemplatesComponent } from '../general-templates/general-templates';

@Component({
  selector: 'app-project-detail',
  standalone: true,
  imports: [FormsModule, GeneralTemplatesComponent],
  template: `
    <div class="space-y-8 animate-in fade-in duration-300">
      
      <!-- Header -->
      <div class="bg-surface-low/50 border border-border-default rounded-theme-xl p-8 shadow-xl">
          <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
              <div class="flex items-center gap-4">
                  <button (click)="back.emit()" class="p-2 -ml-2 text-text-muted hover:text-white hover:bg-surface-mid rounded-theme-md transition-colors">
                      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline></svg>
                  </button>
                  <div>
                      <div class="flex items-center gap-3 mb-2">
                        <div class="w-4 h-4 rounded-full" [style.backgroundColor]="project()?.color || '#3b82f6'"></div>
                        <h2 class="text-2xl font-bold text-white leading-none">{{ project()?.name || 'Project Details' }}</h2>
                      </div>
                      <p class="text-text-muted">{{ project()?.description || 'No description' }}</p>
                  </div>
              </div>
          </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          <!-- Left Column: Datasets & Actions -->
          <div class="lg:col-span-2 space-y-6">
              
              <!-- Stats Row -->
              <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-4">
                      <div class="text-sm text-text-muted mb-1">Datasets</div>
                      <div class="text-2xl font-bold text-white">{{ project()?.stats?.datasets || 0 }}</div>
                  </div>
                  <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-4">
                      <div class="text-sm text-text-muted mb-1">Training Jobs</div>
                      <div class="text-2xl font-bold text-white">{{ project()?.stats?.jobs || 0 }}</div>
                  </div>
                  <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-4">
                      <div class="text-sm text-text-muted mb-1">Training Templates</div>
                      <div class="text-2xl font-bold text-brand-light">{{ project()?.stats?.training_templates || 0 }}</div>
                  </div>
                  <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-4">
                      <div class="text-sm text-text-muted mb-1">Caption Templates</div>
                      <div class="text-2xl font-bold text-brand-light">{{ project()?.stats?.captioning_templates || 0 }}</div>
                  </div>
              </div>

              <!-- General Templates Branching Area -->
              <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-6 shadow-lg">
                  <h3 class="text-lg font-bold text-white mb-2">Branch Global Templates</h3>
                  <p class="text-text-muted mb-6 text-sm">
                      Copy "General" (global) templates into this project to customize them without affecting other projects.
                  </p>
                  
                  <app-general-templates [projectId]="projectId()"></app-general-templates>
              </div>

          </div>

          <!-- Right Column: Settings -->
          <div class="space-y-6">
              <div class="bg-surface-low border border-surface-mid rounded-theme-xl p-6 shadow-lg">
                  <h3 class="text-lg font-bold text-white mb-4">Project Settings</h3>
                  
                  <div class="space-y-4">
                      <div>
                          <label class="block text-sm font-medium text-text-muted mb-1">Name</label>
                          <input type="text" [ngModel]="project()?.name" disabled
                                 class="w-full bg-surface-mid border border-border-default rounded-theme-md px-3 py-2 text-white focus:outline-none focus:border-brand/50 disabled:opacity-50">
                      </div>
                      <div>
                          <label class="block text-sm font-medium text-text-muted mb-1">Description</label>
                          <textarea [ngModel]="project()?.description" disabled rows="3"
                                    class="w-full bg-surface-mid border border-border-default rounded-theme-md px-3 py-2 text-white focus:outline-none focus:border-brand/50 disabled:opacity-50"></textarea>
                      </div>
                      <div>
                          <label class="block text-sm font-medium text-text-muted mb-1">Color</label>
                          <div class="flex items-center gap-3">
                              <input type="color" [ngModel]="project()?.color" disabled
                                     class="w-10 h-10 rounded border-0 bg-transparent disabled:opacity-50">
                          </div>
                      </div>
                      
                      <!-- Editable mode toggle button would go here -->
                  </div>
              </div>
          </div>
      </div>
    </div>
  `
})
export class ProjectDetailComponent {
  private projectService = inject(ProjectService);
  
  projectId = input.required<string>();
  back = output<void>();

  // A local signal to hold the fetched project details
  project = computed(() => {
    return this.projectService.allProjects().find(p => p.id === this.projectId()) || null;
  });
}
