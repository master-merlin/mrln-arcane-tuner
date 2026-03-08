
import { Component, OnInit, inject, signal, computed, output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DatasetService } from '../../../services/dataset';
import { Subject } from 'rxjs';
import { debounceTime } from 'rxjs/operators';

export interface MaskingTemplate {
    id: string;
    name: string;
    is_default?: boolean;
    readonly?: boolean;
    params: Record<string, any>;
}

export interface MaskingSettingsState {
    modelId: string;
    params: Record<string, any>;
}

@Component({
    selector: 'app-dataset-masking-settings',
    standalone: true,
    imports: [FormsModule],
    template: `
        <div class="space-y-3 animate-fadeIn">
            <!-- Model Selector Row -->
            <div class="bg-surface-high/40 p-3 rounded-theme-lg border border-surface-high/50">
                <div class="mb-2">
                    <label class="text-[10px] uppercase tracking-wider text-text-subtle font-bold mb-1 block">Method</label>
                    <select [ngModel]="selectedMaskModel()" (ngModelChange)="onModelChange($event)"
                        data-testid="masking-model-select"
                        class="w-full bg-surface-low border border-surface-mid text-white text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand transition-colors">
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
                <div class="p-3 bg-surface-low/50 border-b border-surface-mid/50 flex items-end gap-2">
                    <div class="flex-1">
                        <label class="text-[10px] uppercase tracking-wider text-text-subtle font-bold mb-1 block">Settings Template</label>
                        <select [ngModel]="activeTemplateId()" (ngModelChange)="onTemplateChange($event)"
                            data-testid="masking-template-select"
                            class="w-full bg-surface-low border border-surface-high text-white text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand transition-colors">
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
                        [disabled]="activeTemplateId() === 'default'" [class.opacity-50]="activeTemplateId() === 'default'" class="p-1.5 bg-surface-mid hover:bg-surface-high text-yellow-500 rounded-theme-md border border-surface-high transition-colors" title="Rename Template">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
                    </button>
                    <button (click)="deleteTemplate()" 
                        data-testid="delete-masking-template-btn"
                        [disabled]="activeTemplateId() === 'default'" [class.opacity-50]="activeTemplateId() === 'default'" class="p-1.5 bg-surface-mid hover:bg-danger/20 text-danger rounded-theme-md border border-surface-high transition-colors" title="Delete Template">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                    </button>
                </div>

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
                                                        class="w-full bg-surface-low border border-brand/50 text-white text-xs rounded-theme-md px-2 py-1 outline-none focus:border-brand transition-colors"
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
                                            @for (opt of param.options; track opt) {
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
    `]
})
export class DatasetMaskingSettingsComponent implements OnInit {
    private datasetService = inject(DatasetService);

    settingsChanged = output<MaskingSettingsState>();

    maskingModels: any[] = [
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
    currentTemplates = signal<MaskingTemplate[]>([]);
    activeTemplateId = signal<string>('default');
    maskingParams = signal<Record<string, any>>({});

    globalSettings: any = { models: {} };
    private settingsUpdate$ = new Subject<void>();

    activeModelConfig = computed(() => {
        return this.maskingModels.find(m => m.id === this.selectedMaskModel());
    });

    ngOnInit() {
        this.loadSettings();

        this.settingsUpdate$.pipe(
            debounceTime(1000)
        ).subscribe(() => {
            this.pushSettings();
        });
    }

    private loadSettings() {
        this.datasetService.getSettings('masking').subscribe({
            next: (settings) => {
                this.globalSettings = settings || { models: {} };

                if (this.globalSettings.selected_model && this.maskingModels.some(m => m.id === this.globalSettings.selected_model)) {
                    this.selectedMaskModel.set(this.globalSettings.selected_model);
                }

                this.loadModelTemplates(this.selectedMaskModel());
                this.emitChanges();
            },
            error: (err) => {
                this.globalSettings = { models: {} };
                this.loadModelTemplates(this.selectedMaskModel());
                this.emitChanges();
            }
        });
    }

    private loadModelTemplates(modelId: string) {
        const modelData = this.globalSettings.models?.[modelId];
        const modelConfig = this.maskingModels.find(m => m.id === modelId);

        const defaultParams: Record<string, any> = {};
        modelConfig?.params.forEach((p: any) => {
            defaultParams[p.key] = p.default;
        });

        const defaultTemplate: MaskingTemplate = {
            id: 'default',
            name: 'Default',
            is_default: true,
            readonly: true,
            params: defaultParams
        };

        if (modelData && modelData.templates && Array.isArray(modelData.templates)) {
            this.currentTemplates.set(modelData.templates);
            this.activeTemplateId.set(modelData.active_template_id || 'default');
        } else {
            this.currentTemplates.set([defaultTemplate]);
            this.activeTemplateId.set('default');
        }

        this.applyActiveTemplate();
    }

    private applyActiveTemplate() {
        const activeId = this.activeTemplateId();
        const tpl = this.currentTemplates().find(t => t.id === activeId) || this.currentTemplates().find(t => t.id === 'default');

        if (tpl) {
            this.maskingParams.set({ ...tpl.params });
            this.emitChanges();
        }
    }

    private saveModelSettings(modelId: string) {
        if (!this.globalSettings.models) {
            this.globalSettings.models = {};
        }

        this.globalSettings.models[modelId] = {
            active_template_id: this.activeTemplateId(),
            templates: this.currentTemplates()
        };
        this.settingsUpdate$.next();
    }

    private pushSettings() {
        this.datasetService.saveSettings('masking', this.globalSettings).subscribe();
    }

    onModelChange(modelId: string) {
        if (modelId === this.selectedMaskModel()) return;
        this.saveModelSettings(this.selectedMaskModel());
        this.selectedMaskModel.set(modelId);
        this.loadModelTemplates(modelId);
        this.globalSettings.selected_model = modelId;
        this.settingsUpdate$.next();
        this.emitChanges();
    }

    onTemplateChange(tplId: string) {
        this.activeTemplateId.set(tplId);
        this.applyActiveTemplate();
        this.saveModelSettings(this.selectedMaskModel());
    }

    updateParam(key: string, value: any) {
        const currentParams = this.maskingParams();
        const newParams = { ...currentParams, [key]: value };
        this.updateActiveTemplate({ params: newParams });
    }

    private updateActiveTemplate(changes: { params?: Record<string, any> }) {
        const activeId = this.activeTemplateId();
        const templates = this.currentTemplates();
        let activeTpl = templates.find(t => t.id === activeId);

        if (!activeTpl) return;

        if (activeTpl.readonly) {
            const newTpl: MaskingTemplate = {
                id: `tpl_${Date.now()}`,
                name: 'Default by User',
                params: { ...activeTpl.params, ...(changes.params || {}) }
            };
            this.currentTemplates.update(ts => [...ts, newTpl]);
            this.activeTemplateId.set(newTpl.id);
        } else {
            this.currentTemplates.update(ts => ts.map(t => {
                if (t.id === activeId) {
                    return {
                        ...t,
                        params: { ...t.params, ...(changes.params || {}) }
                    };
                }
                return t;
            }));
        }

        this.applyActiveTemplate();
        this.saveModelSettings(this.selectedMaskModel());
    }

    addTemplate() {
        const name = prompt('Template name:');
        if (!name) return;

        const newTpl: MaskingTemplate = {
            id: `tpl_${Date.now()}`,
            name,
            params: { ...this.maskingParams() }
        };

        this.currentTemplates.update(ts => [...ts, newTpl]);
        this.onTemplateChange(newTpl.id);
    }

    renameTemplate() {
        const activeId = this.activeTemplateId();
        if (activeId === 'default') return;
        const tpl = this.currentTemplates().find(t => t.id === activeId);
        if (!tpl) return;

        const name = prompt('Rename template:', tpl.name);
        if (!name || name === tpl.name) return;

        this.currentTemplates.update(ts => ts.map(t => {
            if (t.id === activeId) return { ...t, name };
            return t;
        }));

        this.saveModelSettings(this.selectedMaskModel());
    }

    deleteTemplate() {
        const activeId = this.activeTemplateId();
        if (activeId === 'default') return;

        if (!confirm('Delete this template?')) return;

        this.currentTemplates.update(ts => ts.filter(t => t.id !== activeId));
        this.onTemplateChange('default');
    }

    saveCustomConcept(concept: string, baseOptions: string[]) {
        if (!concept || baseOptions.includes(concept) || concept === '__custom__') return;

        if (!this.globalSettings.saved_concepts) {
            this.globalSettings.saved_concepts = [];
        }

        if (!this.globalSettings.saved_concepts.includes(concept)) {
            this.globalSettings.saved_concepts.push(concept);
            this.globalSettings.saved_concepts.sort();
            this.pushSettings();
        }
    }

    getCombinedOptions(param: any): string[] {
        const saved = this.globalSettings.saved_concepts || [];
        // Unique merge
        return [...new Set([...param.options, ...saved])];
    }

    isCustomValue(param: any, value: any): boolean {
        if (!value) return false;
        if (value === '__custom__') return true;
        const options = this.getCombinedOptions(param);
        return !options.includes(value);
    }

    onCreatableSelectChange(param: any, selection: string) {
        this.updateParam(param.key, selection);
    }

    onCustomInputChange(param: any, value: string) {
        this.updateParam(param.key, value);

        if (value && value.length > 2 && value !== '__custom__') {
            this.saveCustomConcept(value, param.options);
        }
    }

    private emitChanges() {
        this.settingsChanged.emit({
            modelId: this.selectedMaskModel(),
            params: this.maskingParams()
        });
    }
}
