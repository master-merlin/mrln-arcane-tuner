import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { TrainingDynamicConfigComponent } from '../../components/training/training-dynamic-config/training-dynamic-config';
import { ScopeStore } from '../../state/scope.store';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { JobService } from '../../services/job';
import { ToastService } from '../../services/toast';

interface ModelDefinition {
    id: string;
    name?: string;
    family?: string;
    [key: string]: unknown;
}

/**
 * Training screen — IDE 3-pane layout that wraps the existing
 * `training-dynamic-config` form component.
 *
 *   LEFT (264px)  · Sections TOC (static stub; scroll-spy + status dots come
 *                   in a follow-up — see TODO below).
 *   CENTER        · Page head with model-definition picker, then the dynamic
 *                   config form. The form's own internal sections still
 *                   render top-to-bottom; the TOC will eventually scroll to
 *                   anchors generated from those sections.
 *   RIGHT (320px) · Live Estimate rail — placeholder tiles until the
 *                   `POST /training/estimate` backend endpoint exists.
 *
 * Model fetching + schema loading + job queuing all migrate here from the
 * retired `AppComponent` (see git history at 50bde31:frontend/src/app/app.ts).
 * The `pluginId` is hard-coded to `'standard'` to match the old behaviour;
 * once multiple training plugins exist this becomes user-selectable.
 *
 * TODO(frontend): scroll-spy + per-section status dots in the TOC. Requires
 *   either parsing the dynamic schema for groups up here or having
 *   training-dynamic-config emit anchor/health info.
 * TODO(backend): POST /training/estimate — wire the Live Estimate rail to
 *   real wall-time / VRAM / warning data once the endpoint exists.
 */
@Component({
    selector: 'app-training-screen',
    standalone: true,
    imports: [TrainingDynamicConfigComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './training-screen.html',
    styleUrl: './training-screen.css',
})
export class TrainingScreen {
    private http = inject(HttpClient);
    private rtc = inject(RuntimeConfigService);
    private jobs = inject(JobService);
    private toast = inject(ToastService);
    protected scope = inject(ScopeStore);

    protected availableModels = signal<ModelDefinition[]>([]);
    protected selectedDefinitionId = signal<string | null>(null);
    protected currentSchema = signal<unknown>(null);

    /** Static TOC entries — a stub until scroll-spy lands. */
    protected readonly sections = [
        { id: 'model', label: 'Model' },
        { id: 'dataset', label: 'Dataset' },
        { id: 'hyperparameters', label: 'Hyperparameters' },
        { id: 'output', label: 'Output' },
    ];

    private readonly pluginId = 'standard';

    constructor() {
        this.fetchModels();
    }

    protected fetchModels(): void {
        this.http.get<ModelDefinition[]>(`${this.rtc.apiUrl}/models/definitions`).subscribe({
            next: defs => {
                this.availableModels.set(defs);
                if (defs.length > 0) this.selectDefinition(defs[0].id);
            },
            error: (err: { message?: string }) =>
                this.toast.error('Failed to load model definitions: ' + (err?.message ?? 'unknown error')),
        });
    }

    protected selectDefinition(id: string): void {
        this.selectedDefinitionId.set(id);
        this.http.get(`${this.rtc.apiUrl}/plugins/${this.pluginId}/schema?t=${Date.now()}`).subscribe({
            next: (s: unknown) => this.currentSchema.set(s),
            error: (err: { message?: string }) =>
                this.toast.error('Failed to load training schema: ' + (err?.message ?? 'unknown error')),
        });
    }

    protected onModelChange(event: Event): void {
        const id = (event.target as HTMLSelectElement).value;
        if (id) this.selectDefinition(id);
    }

    protected queueJob(config: unknown): void {
        this.jobs.createJob(this.pluginId, config).subscribe({
            next: () => this.toast.success('Training job queued.'),
            error: (err: { error?: { detail?: string }; message?: string }) =>
                this.toast.error('Failed to create job: ' + (err?.error?.detail || err?.message || 'unknown error')),
        });
    }
}
