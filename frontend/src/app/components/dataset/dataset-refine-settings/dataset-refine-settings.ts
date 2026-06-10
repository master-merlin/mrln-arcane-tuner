import { ChangeDetectionStrategy, Component, OnInit, computed, effect, inject, output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DatasetService } from '../../../services/dataset';
import { CaptionContextService } from '../../../services/caption-context.service';
import { ModelContextStore, DefinitionRef } from '../../../state/model-context.store';
import { ToastService } from '../../../services/toast';

export interface RefineSettingsState {
    definitionId: string;
    preset: string;
    model: string;
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

            <!-- Preset -->
            <div>
                <label class="block mb-1 text-text-subtle uppercase tracking-wide text-[10px]">Preset</label>
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
    private captionContext = inject(CaptionContextService);
    private toast = inject(ToastService);
    protected modelContext = inject(ModelContextStore);

    protected readonly presets = PRESETS;
    protected available = signal<boolean | null>(null);
    protected installed = signal<string[]>([]);
    protected curated = signal<string[]>([]);
    protected pulling = signal<string | null>(null);
    protected picking = signal<boolean>(false);
    protected definitions = signal<DefinitionRef[]>([]);

    model = signal<string>('');
    protected preset = signal<string>('standardize');

    /** Curated models not yet installed. */
    protected pullable = computed(() => this.curated().filter(c => !this.installed().includes(c)));

    constructor() {
        // Recompute + emit whenever inputs change.
        effect(() => {
            const defId = this.modelContext.activeDefinitionId();
            const model = this.model();
            const preset = this.preset();
            const available = this.available();
            if (available === false || !defId || !model) {
                this.settingsChanged.emit(null);
                return;
            }
            this.settingsChanged.emit({ definitionId: defId, preset, model });
        });
    }

    ngOnInit(): void {
        this.api.listRefineModels().subscribe({
            next: r => {
                this.available.set(r.available);
                this.installed.set(r.installed ?? []);
                this.curated.set(r.curated ?? []);
                if (!this.model() && r.installed?.length) this.model.set(r.installed[0]);
                else if (!this.model() && r.curated?.length) this.model.set(r.curated[0]);
            },
            error: () => { this.available.set(false); },
        });
        this.captionContext.listDefinitions().subscribe(d => this.definitions.set(d ?? []));
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
