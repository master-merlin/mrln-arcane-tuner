import {
    afterNextRender,
    ChangeDetectionStrategy,
    Component,
    effect,
    ElementRef,
    inject,
    signal,
    viewChild,
} from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { TrainingDynamicConfigComponent } from '../../components/training/training-dynamic-config/training-dynamic-config';
import { TrainingHandoffService } from '../../state/training-handoff.service';
import type { TrainingSegment } from '../../components/training/training-dynamic-config/training-dynamic-config';
import { TrainingToc } from '../../components/training/training-toc/training-toc';
import { TrainingEstimateRail } from '../../components/training/training-estimate-rail/training-estimate-rail';
import type { VRAMReport } from '../../services/system.service';
import { DatasetStore } from '../../state/dataset.store';
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
 *   LEFT (340px)  · Sections TOC (`app-training-toc`) driven by the dynamic
 *                   config's `segmentsChanged` output, with scroll-spy +
 *                   smooth jump-to-section.
 *   CENTER        · Page head, then the dynamic config form. Model family +
 *                   definition are chosen inside the form's Model Selection
 *                   segment, so the shell has no separate model picker. This
 *                   column is its own scroll container so the TOC can track the
 *                   active section by viewport position.
 *   RIGHT (280px) · Live Estimate rail fed by the engine's real VRAM report.
 *
 * Model fetching + schema loading + job queuing all migrate here from the
 * retired `AppComponent` (see git history at 50bde31:frontend/src/app/app.ts).
 * The `pluginId` is hard-coded to `'standard'` to match the old behaviour;
 * once multiple training plugins exist this becomes user-selectable. The
 * training schema is plugin-scoped (identical for every definition), so it is
 * loaded once after the model list arrives.
 */
@Component({
    selector: 'app-training-screen',
    standalone: true,
    imports: [TrainingDynamicConfigComponent, TrainingToc, TrainingEstimateRail],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './training-screen.html',
    styleUrl: './training-screen.css',
})
export class TrainingScreen {
    private http = inject(HttpClient);
    private rtc = inject(RuntimeConfigService);
    private jobs = inject(JobService);
    private toast = inject(ToastService);
    private datasetStore = inject(DatasetStore);
    private handoff = inject(TrainingHandoffService);
    protected scope = inject(ScopeStore);

    /** The config form, so a job's config handed off from Jobs can be applied. */
    private configEditor = viewChild(TrainingDynamicConfigComponent);

    protected availableModels = signal<ModelDefinition[]>([]);
    protected currentSchema = signal<unknown>(null);

    /** Config segments emitted by the dynamic config form (DOM order). */
    protected segments = signal<TrainingSegment[]>([]);
    /** Section currently in view (scroll-spy). */
    protected activeSection = signal<string | null>(null);

    /** Live VRAM report re-broadcast by the dynamic-config engine. */
    protected vramReport = signal<VRAMReport | null>(null);

    /** The center scroll container — the scroll-spy reference frame. */
    private formPane = viewChild<ElementRef<HTMLElement>>('formPane');

    private readonly pluginId = 'standard';

    constructor() {
        this.fetchModels();
        // Hydrate the global dataset store so the per-dataset "N suppressed"
        // exclusion badge has excluded_count data when the Training screen is
        // opened directly (idempotent; other screens seed it too).
        void this.datasetStore.loadAll();
        // Highlight an initial section once the form DOM exists.
        afterNextRender(() => this.onPaneScroll());

        // Apply a config handed off from the Jobs screen ("Reload" / "Save
        // template") once the config form component is live. Mirrors the legacy
        // app-shell pendingConfig effect; loadExternalConfig suppresses
        // auto-template so reloading doesn't spawn a phantom template.
        effect(() => {
            const editor = this.configEditor();
            const pending = this.handoff.pending();
            if (!editor || !pending) return;
            const h = this.handoff.consume();
            if (!h) return;
            // Defer so the reactive form has finished building from the schema.
            setTimeout(() => {
                if (h.mode === 'reload') {
                    editor.loadExternalConfig(h.config);
                    this.toast.success('Configuration loaded into Training settings.');
                } else if (h.templateName && h.definitionId) {
                    editor.importTemplate(h.templateName, h.config, h.definitionId);
                }
            }, 200);
        });
    }

    protected onSegmentsChanged(s: TrainingSegment[]): void {
        this.segments.set(s);
    }

    protected onVramReport(r: VRAMReport | null): void {
        this.vramReport.set(r);
    }

    /** Scroll-spy: pick the section whose top sits closest above the fold. */
    protected onPaneScroll(): void {
        const pane = this.formPane()?.nativeElement;
        const segs = this.segments();
        if (!pane || segs.length === 0) return;

        const paneTop = pane.getBoundingClientRect().top;
        let current: string | null = segs[0]?.id ?? null;
        for (const seg of segs) {
            const el = document.getElementById(seg.id);
            if (el && el.getBoundingClientRect().top - paneTop < 80) current = seg.id;
        }
        this.activeSection.set(current);
    }

    protected onJump(id: string): void {
        document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    protected fetchModels(): void {
        this.http.get<ModelDefinition[]>(`${this.rtc.apiUrl}/models/definitions`).subscribe({
            next: defs => {
                this.availableModels.set(defs);
                if (defs.length > 0) this.loadSchema();
            },
            error: (err: { message?: string }) =>
                this.toast.error('Failed to load model definitions: ' + (err?.message ?? 'unknown error')),
        });
    }

    /**
     * Load the (plugin-scoped) training schema. The schema is identical for
     * every definition — the active family/definition is chosen inside the
     * form's Model Selection segment — so it is fetched once.
     */
    protected loadSchema(): void {
        this.http.get(`${this.rtc.apiUrl}/plugins/${this.pluginId}/schema?t=${Date.now()}`).subscribe({
            next: (s: unknown) => this.currentSchema.set(s),
            error: (err: { message?: string }) =>
                this.toast.error('Failed to load training schema: ' + (err?.message ?? 'unknown error')),
        });
    }

    protected queueJob(config: unknown): void {
        this.jobs.createJob(this.pluginId, config).subscribe({
            next: () => this.toast.success('Training job queued.'),
            error: (err: { error?: { detail?: string }; message?: string }) =>
                this.toast.error('Failed to create job: ' + (err?.error?.detail || err?.message || 'unknown error')),
        });
    }
}
