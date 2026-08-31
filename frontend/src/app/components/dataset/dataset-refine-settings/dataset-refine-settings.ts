import { ChangeDetectionStrategy, Component, OnInit, computed, effect, inject, output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DatasetService } from '../../../services/dataset';
import { LlmSettingsService } from '../../../services/llm-settings.service';
import { CaptionContextService } from '../../../services/caption-context.service';
import { ModelContextStore, DefinitionRef } from '../../../state/model-context.store';
import { ToastService } from '../../../services/toast';

/** Caption-style template for refinement. "auto" derives from the model's text
 *  encoder (CLIP/SDXL → tags, T5/large-context → natural language); the other
 *  two are explicit user overrides. */
export type RefineStyle = 'auto' | 'natural_language' | 'tags';

export interface RefineSettingsState {
    definitionId: string;
    preset: string;
    model: string;
    style: RefineStyle;
}

const PRESETS = ['standardize', 'synonym_merge'];

@Component({
    selector: 'app-dataset-refine-settings',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [FormsModule],
    template: `
        <div class="flex flex-col gap-3 text-xs">
            <!-- Ollama availability -->
            @if (available() === false) {
                <div class="p-2 rounded-theme-md bg-danger/10 border border-danger/30 text-danger">
                    Ollama unavailable — start it or configure the endpoint in settings.
                </div>
            }

            <!-- Definition -->
            <div>
                <label class="block mb-1 text-text-subtle uppercase tracking-wide text-[10px]">Target definition</label>
                @if (modelContext.activeDefinition(); as def) {
                    <div class="flex items-center justify-between p-2 rounded-theme-md bg-surface-high">
                        <span class="font-mono">{{ def.name }}</span>
                        <button type="button" class="text-[10px] text-brand" (click)="picking.set(true)">change</button>
                    </div>
                }
                @if (!modelContext.activeDefinition() || picking()) {
                    <select class="w-full mt-1 p-2 rounded-theme-md bg-surface-high"
                            [ngModel]="modelContext.activeDefinitionId()"
                            (ngModelChange)="onPickDefinition($event)">
                        <option [ngValue]="null" disabled>Select a definition…</option>
                        @for (d of definitions(); track d.id) {
                            <option [ngValue]="d.id">{{ d.family }} — {{ d.name }}</option>
                        }
                    </select>
                }
            </div>

            <!-- Model -->
            <div>
                <label class="block mb-1 text-text-subtle uppercase tracking-wide text-[10px]">Refinement model</label>
                <select class="w-full p-2 rounded-theme-md bg-surface-high" [ngModel]="model()" (ngModelChange)="model.set($event)">
                    @for (m of installed(); track m) { <option [ngValue]="m">{{ m }}</option> }
                </select>
                @for (c of pullable(); track c) {
                    <div class="flex items-center justify-between mt-1 text-[10px]">
                        <span class="font-mono text-text-subtle">{{ c }}</span>
                        @if (pulling() === c) {
                            <span class="text-brand">Pulling…</span>
                        } @else {
                            <button type="button" class="text-brand" (click)="pull(c)">Pull</button>
                        }
                    </div>
                }
                <input class="w-full mt-1 p-2 rounded-theme-md bg-surface-high font-mono text-[10px]"
                       placeholder="or type a model tag…" [ngModel]="model()" (ngModelChange)="model.set($event)">
            </div>

            <!-- Refinement template (caption style) -->
            <div>
                <label class="block mb-1 text-text-subtle uppercase tracking-wide text-[10px]">Refinement template</label>
                <select class="w-full p-2 rounded-theme-md bg-surface-high" [ngModel]="style()" (ngModelChange)="style.set($event)"
                        data-testid="refine-style">
                    <option [ngValue]="'auto'">Auto — match model ({{ autoStyleLabel() }})</option>
                    <option [ngValue]="'natural_language'">Natural-language caption</option>
                    <option [ngValue]="'tags'">Booru tags</option>
                </select>
                <p class="mt-1 text-[10px] text-text-subtle">Caption captioning models in prose; tag SDXL-style models. Always fit the model's token budget.</p>
            </div>

            <!-- Preset -->
            <div>
                <label class="block mb-1 text-text-subtle uppercase tracking-wide text-[10px]">Operation</label>
                <select class="w-full p-2 rounded-theme-md bg-surface-high" [ngModel]="preset()" (ngModelChange)="preset.set($event)">
                    @for (p of presets; track p) { <option [ngValue]="p">{{ p }}</option> }
                </select>
            </div>
        </div>
    `,
})
export class DatasetRefineSettingsComponent implements OnInit {
    settingsChanged = output<RefineSettingsState | null>();

    private api = inject(DatasetService);
    private llmSettings = inject(LlmSettingsService);
    private captionContext = inject(CaptionContextService);
    private toast = inject(ToastService);
    protected modelContext = inject(ModelContextStore);

    protected readonly presets = PRESETS;
    protected available = signal<boolean | null>(null);
    protected installed = signal<string[]>([]);
    protected curated = signal<string[]>([]);
    /** `llm_refine.model` — the Server-screen default, '' when unset. */
    protected defaultModel = signal<string>('');
    /** The settings request has answered (successfully or not). Gates seeding. */
    protected settingsResolved = signal<boolean>(false);
    protected pulling = signal<string | null>(null);
    protected picking = signal<boolean>(false);
    protected definitions = signal<DefinitionRef[]>([]);

    model = signal<string>('');
    protected preset = signal<string>('standardize');
    protected style = signal<RefineStyle>('auto');

    /** Curated models not yet installed. */
    protected pullable = computed(() => this.curated().filter(c => !this.installed().includes(c)));

    /** What "Auto" resolves to for the active definition's family — mirrors the
     *  backend `caption_style_for` (CLIP/SDXL → tags, else natural language). */
    protected autoStyleLabel = computed<'tags' | 'natural language'>(() =>
        this.modelContext.activeDefinition()?.family === 'sdxl' ? 'tags' : 'natural language',
    );

    constructor() {
        // Recompute + emit whenever inputs change.
        effect(() => {
            const defId = this.modelContext.activeDefinitionId();
            const model = this.model();
            const preset = this.preset();
            const style = this.style();
            const available = this.available();
            if (available === false || !defId || !model) {
                this.settingsChanged.emit(null);
                return;
            }
            this.settingsChanged.emit({ definitionId: defId, preset, model, style });
        });
    }

    ngOnInit(): void {
        this.api.listRefineModels().subscribe({
            next: r => {
                this.available.set(r.available);
                this.installed.set(r.installed ?? []);
                this.curated.set(r.curated ?? []);
                this.seedModel();
            },
            error: () => { this.available.set(false); },
        });
        // The configured default (Server screen → LLM Refine Endpoint). Both
        // requests are in flight at once, so seeding never assumes an order —
        // see `seedModel`, which refuses to guess until this one has answered.
        this.llmSettings.get().subscribe({
            next: s => { this.defaultModel.set(s.model?.trim() ?? ''); this.settingsResolved.set(true); this.seedModel(); },
            // A failed settings load must still release the gate, or the panel
            // would sit with no model forever waiting on an answer that is
            // never coming.
            error: () => { this.settingsResolved.set(true); this.seedModel(); },
        });
        this.captionContext.listDefinitions().subscribe(d => this.definitions.set(d ?? []));
    }

    /**
     * Choose an initial model, best available first: the configured default,
     * then anything installed, then a curated tag.
     *
     * Two independent responses call this, so it is guarded twice. It never
     * overwrites a model already set — the second response must not undo the
     * first, nor undo a pick the user made in between. And it does not fall
     * back before the settings request has answered: the model list usually
     * wins the race, and seeding `installed[0]` on arrival would silently
     * beat the configured default every time, which is the exact bug this
     * panel had before the Server-screen picker existed.
     */
    private seedModel(): void {
        if (this.model() || !this.settingsResolved()) return;
        const saved = this.defaultModel();
        if (saved) { this.model.set(saved); return; }
        const installed = this.installed();
        if (installed.length) { this.model.set(installed[0]); return; }
        const curated = this.curated();
        if (curated.length) this.model.set(curated[0]);
    }

    protected onPickDefinition(id: string): void {
        const def = this.definitions().find(d => d.id === id);
        if (!def) return;
        this.modelContext.setModelAware(true);
        this.modelContext.setDefinition(def);
        this.picking.set(false);
    }

    pull(tag: string): void {
        this.pulling.set(tag);
        this.api.pullRefineModel(tag).subscribe({
            next: ({ ok }) => {
                this.pulling.set(null);
                if (ok) {
                    this.installed.update(xs => xs.includes(tag) ? xs : [...xs, tag]);
                    this.model.set(tag);
                } else {
                    this.toast.error(`Pull failed: ${tag}`);
                }
            },
            error: () => { this.pulling.set(null); this.toast.error(`Pull failed: ${tag}`); },
        });
    }
}
