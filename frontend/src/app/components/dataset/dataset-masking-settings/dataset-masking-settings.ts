
import { Component, OnInit, inject, signal, computed, output, input, effect, untracked, ChangeDetectionStrategy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DatasetService } from '../../../services/dataset';
import { ProjectService, ProjectPreferences } from '../../../services/project.service';
import { TemplateService, Template } from '../../../services/template.service';
import { Subject } from 'rxjs';
import { debounceTime, switchMap } from 'rxjs/operators';

export interface MaskingSettingsState {
    modelId: string;
    params: Record<string, unknown>;
}

/** A single tunable parameter on a masking model (see {@link DatasetMaskingSettingsComponent.maskingModels}). */
export interface MaskingParam {
    key: string;
    label: string;
    type: 'creatable-select' | 'select' | 'checkbox' | 'number';
    default: string | number | boolean;
    /** Choices for `select` / `creatable-select`. */
    options?: string[];
    /** Inline label rendered next to a `checkbox`. */
    checkboxLabel?: string;
    min?: number;
    max?: number;
    step?: number;
}

/** A masking method and its tunable parameters. */
export interface MaskingModelConfig {
    id: string;
    name: string;
    description: string;
    params: MaskingParam[];
}

@Component({
    selector: 'app-dataset-masking-settings',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [FormsModule],
    template: `
        <div class="space-y-3 animate-fadeIn">
            <!-- Model Selector Row -->
            <div class="bg-surface-high/40 p-3 rounded-theme-lg border border-surface-high/50">
                <div class="mb-2">
                    <label class="text-[10px] uppercase tracking-wider text-text-subtle font-bold mb-1 block">Method</label>
                    <select [ngModel]="selectedMaskModel()" (ngModelChange)="onModelChange($event)"
                        data-testid="masking-model-select"
                        class="w-full bg-surface-low border border-surface-mid text-text-primary text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand transition-colors">
                        @for (model of maskingModels; track model.id) {
                            <option [value]="model.id">{{ model.name }}</option>
                        }
                    </select>
                </div>
                @if (activeModelConfig(); as config) {
                    <p class="text-[10px] text-text-subtle italic border-t border-surface-high/30 pt-1 mt-1">{{ config.description }}</p>
                }
            </div>

            <!-- Template Card -->
            <div class="bg-surface-high/40 rounded-theme-lg border border-surface-mid/50 overflow-hidden">
                <!-- Template Header & Actions -->
                @if (!hideTemplateBar()) {
                <div class="p-3 bg-surface-low/50 border-b border-surface-mid/50 flex items-end gap-2">
                    <div class="flex-1">
                        <label class="text-[10px] uppercase tracking-wider text-text-subtle font-bold mb-1 block">Settings Template</label>
                        <select [ngModel]="activeTemplateId()" (ngModelChange)="onTemplateChange($event)"
                            data-testid="masking-template-select"
                            class="w-full bg-surface-low border border-surface-high text-text-primary text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand transition-colors">
                            @for (tpl of currentTemplates(); track tpl.id) {
                                <option [value]="tpl.id">{{ tpl.name }} {{tpl.is_default ? '(Default)' : ''}}</option>
                            }
                        </select>
                    </div>

                    <button (click)="addTemplate()"
                        data-testid="add-masking-template-btn"
                        class="p-1.5 bg-surface-mid hover:bg-surface-high text-brand rounded-theme-md border border-surface-high transition-colors" title="Clone as New Template">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>
                    </button>
                    <button (click)="renameTemplate()"
                        data-testid="rename-masking-template-btn"
                        [disabled]="isDefaultTemplate()" [class.opacity-50]="isDefaultTemplate()" class="p-1.5 bg-surface-mid hover:bg-surface-high text-yellow-500 rounded-theme-md border border-surface-high transition-colors" title="Rename Template">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
                    </button>
                    <button (click)="deleteTemplate()"
                        data-testid="delete-masking-template-btn"
                        [disabled]="isDefaultTemplate()" [class.opacity-50]="isDefaultTemplate()" class="p-1.5 bg-surface-mid hover:bg-danger/20 text-danger rounded-theme-md border border-surface-high transition-colors" title="Delete Template">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                    </button>
                </div>
                }

                <!-- Params Content -->
                <div class="p-3 space-y-3">
                    @if (activeModelConfig(); as config) {
                        <div class="space-y-2">
                             @for (param of config.params; track param.key) {
                                <div>
                                    <div class="flex items-center justify-between mb-0.5">
                                        <label class="text-[10px] text-text-subtle">{{ param.label }}</label>
                                        @if (param.type === 'number') {
                                            <span class="text-[10px] text-text-disabled font-mono">{{ maskingParams()[param.key] }}</span>
                                        }
                                    </div>
                                    @if (param.type === 'creatable-select') {
                                        <div class="flex flex-col gap-1">
                                            <select 
                                                [ngModel]="isCustomValue(param, maskingParams()[param.key]) ? '__custom__' : maskingParams()[param.key]" 
                                                (ngModelChange)="onCreatableSelectChange(param, $event)"
                                                [attr.data-testid]="'masking-param-' + param.key"
                                                class="w-full bg-surface-low border border-surface-mid text-text-secondary text-xs rounded-theme-md px-2 py-1 outline-none focus:border-brand">
                                                @for (opt of getCombinedOptions(param); track opt) {
                                                    <option [value]="opt">{{ opt || '(None)' }}</option>
                                                }
                                                <option value="__custom__" class="text-brand font-bold">Custom...</option>
                                            </select>
                                            
                                            @if (isCustomValue(param, maskingParams()[param.key]) || maskingParams()[param.key] === '__custom__') {
                                                <div class="animate-fadeIn">
                                                    <input 
                                                        type="text" 
                                                        [ngModel]="maskingParams()[param.key] === '__custom__' ? '' : maskingParams()[param.key]" 
                                                        (ngModelChange)="onCustomInputChange(param, $event)"
                                                        data-testid="masking-custom-input"
                                                        placeholder="Enter custom concept..."
                                                        class="w-full bg-surface-low border border-brand/50 text-text-primary text-xs rounded-theme-md px-2 py-1 outline-none focus:border-brand transition-colors"
                                                        autofocus
                                                    >
                                                </div>
                                            }
                                        </div>
                                    } @else if (param.type === 'select') {
                                        <select [ngModel]="maskingParams()[param.key]" 
                                            (ngModelChange)="updateParam(param.key, $event)"
                                            [attr.data-testid]="'masking-param-' + param.key"
                                            class="w-full bg-surface-low border border-surface-mid text-text-secondary text-xs rounded-theme-md px-2 py-1 outline-none focus:border-brand">
                                            @for (opt of param.options ?? []; track opt) {
                                                <option [value]="opt">{{ opt }}</option>
                                            }
                                        </select>
                                    } @else if (param.type === 'number') {
                                            <input type="range" [min]="param.min" [max]="param.max" [step]="param.step"
                                            [ngModel]="maskingParams()[param.key]"
                                            (ngModelChange)="updateParam(param.key, $event)"
                                            [attr.data-testid]="'masking-param-' + param.key"
                                            class="w-full h-1 bg-surface-high rounded-theme-lg appearance-none cursor-pointer accent-brand">
                                    } @else if (param.type === 'checkbox') {
                                        <div class="flex items-center">
                                            <input type="checkbox" 
                                                [ngModel]="maskingParams()[param.key]"
                                                (ngModelChange)="updateParam(param.key, $event)"
                                                [attr.data-testid]="'masking-param-' + param.key"
                                                class="w-3.5 h-3.5 bg-surface-low border-surface-mid rounded text-brand focus:ring-brand">
                                                <span class="ml-2 text-[10px] text-text-muted">{{ param.checkboxLabel || param.label }}</span>
                                        </div>
                                    }
                                </div>
                             }
                        </div>
                    }
                </div>
            </div>
        </div>
    `,
    styles: [`
        .animate-fadeIn {
            animation: fadeIn 0.3s ease-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        /* Sliders honor an inherited --form-accent override (e.g. the green
           mass-mask modal) and fall back to brand otherwise — matching the
           global checkbox/control convention in styles.css. The prior auto
           fallback rendered native blue in the detail sidebar / mass-caption. */
        input[type="range"] {
            accent-color: var(--form-accent, var(--color-brand));
        }
    `]
})
export class DatasetMaskingSettingsComponent implements OnInit {
    private datasetService = inject(DatasetService);
    private projectService = inject(ProjectService);
    private templateService = inject(TemplateService);

    projectId = input<string | null>(null);

    // Compute effective project ID falling back to global active dataset project
    effectiveProjectId = computed(() => this.projectId() ?? this.projectService.activeDatasetProject());
    settingsChanged = output<MaskingSettingsState>();

    /** Edit-modal support (defaults preserve mass-modal behavior). See
     *  dataset-caption-settings for the rationale. */
    hideTemplateBar = input(false);
    autoSave = input(true);
    presetTemplate = input<Template | null>(null);

    maskingModels: MaskingModelConfig[] = [
        {
            id: 'sam3',
            name: 'Meta SAM 3',
            description: 'Segment Anything Model 3. Unified model for concept segmentation (text) and geometric prompts (points/boxes).',
            params: [
                { key: 'text_prompt', label: 'Concept Prompt (Text)', type: 'creatable-select', options: ['', 'subject', 'main object', 'person', 'clothing', 'background', 'hair', 'face'], default: 'subject' },
                { key: 'multimask_output', label: 'Multi-Mask Output (Ambiguity Resolution)', checkboxLabel: 'Use MMO', type: 'checkbox', default: true },
                { key: 'max_hole_area', label: 'Fill Holes (Area)', type: 'number', min: 0, max: 1000, step: 10, default: 0 },
                { key: 'max_sprinkle_area', label: 'Remove Noise (Area)', type: 'number', min: 0, max: 1000, step: 10, default: 0 }
            ]
        },
        {
            id: 'rembg',
            name: 'RemBG',
            description: 'Background Removal using U2Net and other models.',
            params: [
                {
                    key: 'model_name', label: 'Model Variant', type: 'select', options: [
                        'u2net', 'u2netp', 'u2net_human_seg', 'u2net_cloth_seg', 'silueta',
                        'isnet-general-use', 'isnet-anime',
                        'birefnet-general', 'birefnet-general-lite', 'birefnet-massive',
                        'birefnet-portrait', 'birefnet-dis', 'birefnet-hrsod', 'birefnet-cod',
                        'bria-rmbg'
                    ], default: 'birefnet-general'
                },
                { key: 'post_process_mask', label: 'Post-Process Mask', checkboxLabel: 'Smooth edges (morphological)', type: 'checkbox', default: true },
                { key: 'alpha_matting', label: 'Alpha Matting', type: 'checkbox', default: false },
                { key: 'alpha_matting_foreground_threshold', label: 'FG Threshold', type: 'number', min: 0, max: 255, step: 1, default: 240 },
                { key: 'alpha_matting_background_threshold', label: 'BG Threshold', type: 'number', min: 0, max: 255, step: 1, default: 10 },
                { key: 'alpha_matting_erode_size', label: 'Erode Size', type: 'number', min: 0, max: 50, step: 1, default: 10 }
            ]
        }
    ];

    selectedMaskModel = signal<string>('sam3');
    currentTemplates = signal<Template[]>([]);
    activeTemplateId = signal<string | null>(null);
    maskingParams = signal<Record<string, unknown>>({});

    private preferences: ProjectPreferences | null = null;
    private savedConcepts: string[] = [];
    private settingsUpdate$ = new Subject<void>();
    private pendingSaves = new Map<string, { config: Record<string, unknown> }>();

    activeModelConfig = computed(() => {
        return this.maskingModels.find(m => m.id === this.selectedMaskModel());
    });

    isDefaultTemplate() {
        const id = this.activeTemplateId();
        const templates = this.currentTemplates();
        const tpl = templates.find(t => t.id === id);
        return tpl ? tpl.is_default || tpl.readonly : false;
    }

    constructor() {
        effect(() => {
            const preset = this.presetTemplate();
            // Apply the preset OUTSIDE tracking — see dataset-caption-settings for the
            // full rationale. applyPreset → applyActiveTemplate → emitChanges() reads
            // maskingParams synchronously; without untracked() each edit would re-run
            // this effect and wipe the change before Save reads it.
            if (preset) { untracked(() => this.applyPreset(preset)); return; }
            const pid = this.effectiveProjectId();
            this.loadPreferencesAndTemplates();
        });
    }

    /** Edit-modal: load a single template into the UI, bypassing preferences. */
    private applyPreset(tpl: Template) {
        if (tpl.model_id && this.maskingModels.some(m => m.id === tpl.model_id)) {
            this.selectedMaskModel.set(tpl.model_id);
        }
        this.currentTemplates.set([tpl]);
        this.activeTemplateId.set(tpl.id);
        this.applyActiveTemplate();
    }

    ngOnInit() {
        this.settingsUpdate$.pipe(
            debounceTime(1000),
            switchMap(() => {
                if (!this.preferences) return [];

                // Inject saved masking concepts into generic training_selections JSON payload
                const sels = this.preferences.training_selections || {};
                sels['saved_masking_concepts'] = this.savedConcepts;

                return this.projectService.updatePreferences(this.effectiveProjectId(), {
                    selected_mask_model: this.selectedMaskModel(),
                    active_mask_template: this.activeTemplateId(),
                    training_selections: sels
                });
            })
        ).subscribe();
    }

    private loadPreferencesAndTemplates() {
        const pId = this.effectiveProjectId();
        this.projectService.getPreferences(pId).pipe(
            switchMap(prefs => {
                this.preferences = prefs;
                
                // Retrieve custom saved concepts from generic selections json
                if (prefs.training_selections?.['saved_masking_concepts']) {
                    this.savedConcepts = prefs.training_selections['saved_masking_concepts'] as string[];
                }

                if (prefs.selected_mask_model && this.maskingModels.some(m => m.id === prefs.selected_mask_model)) {
                    this.selectedMaskModel.set(prefs.selected_mask_model);
                }
                return this.templateService.listMaskingTemplates(this.selectedMaskModel(), pId);
            })
        ).subscribe(templates => {
            this.currentTemplates.set(templates);
            if (this.preferences?.active_mask_template && templates.some(t => t.id === this.preferences!.active_mask_template)) {
                this.activeTemplateId.set(this.preferences.active_mask_template);
            } else {
                const defaultTpl = templates.find(t => t.is_default);
                this.activeTemplateId.set(defaultTpl ? defaultTpl.id : (templates.length > 0 ? templates[0].id : null));
            }
            this.applyActiveTemplate();
        });
    }

    private loadModelTemplates(modelId: string) {
        // Use effectiveProjectId (not the raw input) so project-scoped templates
        // survive a model switch — matches dataset-caption-settings.
        this.templateService.listMaskingTemplates(modelId, this.effectiveProjectId()).subscribe(templates => {
            this.currentTemplates.set(templates);
            const defaultTpl = templates.find(t => t.is_default);
            this.activeTemplateId.set(defaultTpl ? defaultTpl.id : (templates.length > 0 ? templates[0].id : null));
            this.applyActiveTemplate();
            this.settingsUpdate$.next();
        });
    }

    private applyActiveTemplate() {
        const activeId = this.activeTemplateId();
        const tpl = this.currentTemplates().find(t => t.id === activeId);

        if (tpl) {
            const modelConfig = this.maskingModels.find(m => m.id === this.selectedMaskModel());
            const codeDefaults: Record<string, unknown> = {};
            modelConfig?.params.forEach(p => { codeDefaults[p.key] = p.default; });

            this.maskingParams.set({ ...codeDefaults, ...(tpl.config || {}) });
            this.emitChanges();
        }
    }

    onModelChange(modelId: string) {
        if (modelId === this.selectedMaskModel()) return;
        this.selectedMaskModel.set(modelId);
        if (!this.autoSave()) {
            const cfg = this.maskingModels.find(m => m.id === modelId);
            const defaults: Record<string, unknown> = {};
            cfg?.params.forEach(p => { defaults[p.key] = p.default; });
            this.maskingParams.set(defaults);
            this.emitChanges();
            return;
        }
        this.loadModelTemplates(modelId);
        this.emitChanges();
    }

    onTemplateChange(tplId: string) {
        this.activeTemplateId.set(tplId);
        this.applyActiveTemplate();
        this.settingsUpdate$.next();
    }

    updateParam(key: string, value: unknown) {
        if (!this.autoSave()) {
            this.maskingParams.update(p => ({ ...p, [key]: value }));
            this.emitChanges();
            return;
        }
        const newParams = { ...this.maskingParams(), [key]: value };
        this.updateActiveTemplate({ config: newParams });
    }

    private updateActiveTemplate(changes: { config?: Record<string, unknown> }) {
        const activeId = this.activeTemplateId();
        if (!activeId) return;

        const templates = this.currentTemplates();
        let activeTpl = templates.find(t => t.id === activeId);

        if (!activeTpl) return;

        if (activeTpl.readonly) {
            const config = { ...(activeTpl.config || {}), ...(changes.config || {}) };
            
            this.templateService.createMaskingTemplate({
                model_id: this.selectedMaskModel(),
                name: 'Custom Settings',
                project_id: this.effectiveProjectId(),
                config: config
            }).subscribe(newTpl => {
                this.currentTemplates.update(ts => [...ts, newTpl]);
                this.activeTemplateId.set(newTpl.id);
                this.applyActiveTemplate();
                this.settingsUpdate$.next();
            });
        } else {
            const config = { ...(activeTpl.config || {}), ...(changes.config || {}) };
            
            this.currentTemplates.update(ts => ts.map(t => {
                if (t.id === activeId) {
                    return { ...t, config: config };
                }
                return t;
            }));
            this.applyActiveTemplate();

            this.pendingSaves.set(activeId, { config });
            
            setTimeout(() => {
                const pending = this.pendingSaves.get(activeId);
                if (pending) {
                    this.pendingSaves.delete(activeId);
                    this.templateService.updateTemplate('masking', activeId, pending).subscribe();
                }
            }, 500);
        }
    }

    addTemplate() {
        const name = prompt('Template name:');
        if (!name) return;

        this.templateService.createMaskingTemplate({
            model_id: this.selectedMaskModel(),
            name,
            project_id: this.effectiveProjectId(),
            config: this.maskingParams()
        }).subscribe(newTpl => {
            this.currentTemplates.update(ts => [...ts, newTpl]);
            this.onTemplateChange(newTpl.id);
        });
    }

    renameTemplate() {
        const activeId = this.activeTemplateId();
        if (!activeId || this.isDefaultTemplate()) return;
        
        const tpl = this.currentTemplates().find(t => t.id === activeId);
        if (!tpl) return;

        const name = prompt('Rename template:', tpl.name);
        if (!name || name === tpl.name) return;

        this.templateService.updateTemplate('masking', activeId, { name }).subscribe(updatedTpl => {
            this.currentTemplates.update(ts => ts.map(t => t.id === activeId ? updatedTpl : t));
        });
    }

    deleteTemplate() {
        const activeId = this.activeTemplateId();
        if (!activeId || this.isDefaultTemplate()) return;

        if (!confirm('Delete this template?')) return;

        this.templateService.deleteTemplate('masking', activeId).subscribe(() => {
            this.currentTemplates.update(ts => ts.filter(t => t.id !== activeId));
            const remaining = this.currentTemplates();
            if (remaining.length > 0) {
                this.onTemplateChange(remaining[0].id);
            }
        });
    }

    saveCustomConcept(concept: string, baseOptions: string[]) {
        if (!concept || baseOptions.includes(concept) || concept === '__custom__') return;

        if (!this.savedConcepts.includes(concept)) {
            this.savedConcepts.push(concept);
            this.savedConcepts.sort();
            this.settingsUpdate$.next();
        }
    }

    getCombinedOptions(param: MaskingParam): string[] {
        return [...new Set([...(param.options ?? []), ...this.savedConcepts])];
    }

    isCustomValue(param: MaskingParam, value: unknown): boolean {
        if (!value) return false;
        if (value === '__custom__') return true;
        const options = this.getCombinedOptions(param);
        return !options.includes(value as string);
    }

    onCreatableSelectChange(param: MaskingParam, selection: string) {
        this.updateParam(param.key, selection);
    }

    onCustomInputChange(param: MaskingParam, value: string) {
        this.updateParam(param.key, value);

        if (value && value.length > 2 && value !== '__custom__') {
            this.saveCustomConcept(value, param.options ?? []);
        }
    }

    private emitChanges() {
        this.settingsChanged.emit({
            modelId: this.selectedMaskModel(),
            params: this.maskingParams()
        });
    }
}
