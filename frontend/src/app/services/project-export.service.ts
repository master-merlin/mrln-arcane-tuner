import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { OverlayStore } from '../state/overlay.store';
import { ProjectService } from './project.service';
import { TemplateService, Template, TemplateDomain } from './template.service';
import { ImportArchiveService } from './import-archive.service';
import { ToastService } from './toast';
import type {
  ExportGroup, ExportDatasetChoice, ExportSelection,
} from '../modals/export-options/export-options.component';

const DOMAINS: { domain: TemplateDomain; label: string }[] = [
  { domain: 'training', label: 'Training templates' },
  { domain: 'captioning', label: 'Caption templates' },
  { domain: 'masking', label: 'Mask templates' },
];

/**
 * Opens the generic `export-options` modal for a project, then runs the export
 * request and downloads the resulting `.project.zip`. Shared by the Projects
 * screen card menu and the Project-detail header so both surfaces present the
 * same check/uncheck modal (templates per domain + per-dataset embed/reference/
 * exclude) over a single code path.
 */
@Injectable({ providedIn: 'root' })
export class ProjectExportService {
  private overlay = inject(OverlayStore);
  private projects = inject(ProjectService);
  private templates = inject(TemplateService);
  private archive = inject(ImportArchiveService);
  private toast = inject(ToastService);

  async open(projectId: string, projectName: string): Promise<void> {
    try {
      const [datasets, cap, mask, train] = await Promise.all([
        firstValueFrom(this.projects.getProjectDatasets(projectId)),
        firstValueFrom(this.templates.listCaptioningTemplates(null, projectId)),
        firstValueFrom(this.templates.listMaskingTemplates(null, projectId)),
        firstValueFrom(this.templates.listTrainingTemplates(undefined, projectId)),
      ]);
      const byDomain: Record<TemplateDomain, Template[]> = {
        training: (train ?? []).filter(t => t.project_id === projectId),
        captioning: (cap ?? []).filter(t => t.project_id === projectId),
        masking: (mask ?? []).filter(t => t.project_id === projectId),
      };
      const groups: ExportGroup[] = DOMAINS
        .filter(d => byDomain[d.domain].length > 0)
        .map(d => ({
          key: d.domain,
          label: d.label,
          items: byDomain[d.domain].map(t => ({ id: t.id, label: t.name, checked: true })),
        }));
      const dsChoices: ExportDatasetChoice[] = (datasets ?? []).map(
        d => ({ name: (d as { name: string }).name, mode: 'reference' as const }),
      );

      this.overlay.openModal('export-options', {
        title: `Export "${projectName}"`,
        confirmLabel: 'Export project',
        groups,
        datasets: dsChoices,
        onExport: (sel: ExportSelection) => this.runExport(projectId, projectName, sel),
      });
    } catch (err) {
      this.toast.error('Could not prepare export: ' + this.msg(err));
    }
  }

  private runExport(projectId: string, projectName: string, sel: ExportSelection): void {
    const templates: { domain: string; id: string }[] = [];
    for (const [domain, ids] of Object.entries(sel.groups)) {
      for (const id of ids) templates.push({ domain, id });
    }
    this.projects.exportProject(projectId, { templates, datasets: sel.datasets }).subscribe({
      next: blob => this.archive.downloadBlob(blob, `${this.safe(projectName)}.project.zip`),
      error: err => this.toast.error('Export failed: ' + this.msg(err)),
    });
  }

  private safe(name: string): string {
    const cleaned = [...name].filter(c => /[A-Za-z0-9 _-]/.test(c)).join('').trim();
    return cleaned || 'project';
  }

  private msg(err: unknown): string {
    const e = err as { error?: { detail?: string }; message?: string } | undefined;
    return e?.error?.detail ?? e?.message ?? 'unknown error';
  }
}
