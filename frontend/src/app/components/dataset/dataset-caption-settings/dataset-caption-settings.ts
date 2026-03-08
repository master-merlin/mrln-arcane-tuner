
import { Component, OnInit, inject, signal, computed, input, output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DatasetService } from '../../../services/dataset';
import { Subject } from 'rxjs';
import { debounceTime } from 'rxjs/operators';

export interface CaptionTemplate {
    id: string;
    name: string;
    is_default?: boolean;
    readonly?: boolean;
    params: Record<string, any>;
    system_prompt: string;
}

export interface CaptionSettingsState {
    modelId: string;
    resolvedModelId: string;
    variant?: string;
    systemPrompt: string;
    params: Record<string, any>;
}

@Component({
    selector: 'app-dataset-caption-settings',
    standalone: true,
    imports: [FormsModule],
    template: `
        <div class="space-y-3 animate-fadeIn">
            <!-- Model Selector Row -->
            <div class="bg-surface-high/40 p-3 rounded-theme-lg border border-surface-high/50">
                <div class="flex gap-2 mb-2">
                    <div class="flex-1">
                            <label class="text-[10px] uppercase tracking-wider text-text-subtle font-bold mb-1 block">Model</label>
                            <select [ngModel]="selectedCaptionModel()" (ngModelChange)="onModelChange($event)"
                                   data-testid="caption-model-select"
                            class="w-full bg-surface-low border border-surface-mid text-white text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand transition-colors">
                            @for (model of captionModels; track model.id) {
                                <option [value]="model.id">{{ model.name }}</option>
                            }
                        </select>
                    </div>
                    @if (selectedCaptionModel() === 'qwen3-vl') {
                        <div class="w-1/3">
                            <label class="text-[10px] uppercase tracking-wider text-text-subtle font-bold mb-1 block">Variant</label>
                            <select [ngModel]="selectedQwen3Variant()" (ngModelChange)="onVariantChange($event)"
                                   data-testid="qwen3-variant-select"
                                class="w-full bg-surface-low border border-surface-mid text-white text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand transition-colors">
                                @for (variant of qwen3Variants; track variant) {
                                    <option [value]="variant">{{ variant }}</option>
                                }
                            </select>
                        </div>
                    }
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
                                   data-testid="caption-template-select"
                                class="w-full bg-surface-low border border-surface-high text-white text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand transition-colors">
                                @for (tpl of currentTemplates(); track tpl.id) {
                                    <option [value]="tpl.id">{{ tpl.name }} {{tpl.is_default ? '(Default)' : ''}}</option>
                                }
                            </select>
                        </div>
                        
                        <!-- Actions -->
                    <button (click)="addTemplate()" 
                            data-testid="add-caption-template-btn"
                            class="p-1.5 bg-surface-mid hover:bg-surface-high text-brand rounded-theme-md border border-surface-high transition-colors" title="Clone as New Template">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>
                    </button>
                    <button (click)="renameTemplate()" 
                            data-testid="rename-caption-template-btn"
                            [disabled]="activeTemplateId() === 'default'" [class.opacity-50]="activeTemplateId() === 'default'" class="p-1.5 bg-surface-mid hover:bg-surface-high text-yellow-500 rounded-theme-md border border-surface-high transition-colors" title="Rename Template">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
                    </button>
                    <button (click)="deleteTemplate()" 
                            data-testid="delete-caption-template-btn"
                            [disabled]="activeTemplateId() === 'default'" [class.opacity-50]="activeTemplateId() === 'default'" class="p-1.5 bg-surface-mid hover:bg-danger/20 text-danger rounded-theme-md border border-surface-high transition-colors" title="Delete Template">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                    </button>
                    </div>
                    
                    <!-- Params Content -->
                    <div class="p-3 space-y-3">
                    <!-- System Prompt -->
                    <div>
                        <label class="text-[10px] text-text-subtle mb-1 block">System Prompt</label>
                        <textarea [value]="captionSystemPrompt()" 
                            (input)="onSystemPromptChange($any($event.target).value)"
                            data-testid="caption-system-prompt"
                            rows="3"
                            class="w-full bg-surface-low border border-surface-mid text-text-secondary text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand transition-colors resize-none placeholder-gray-600"
                            placeholder="System Prompt..."></textarea>
                    </div>

                        <!-- Dynamic Params -->
                    @if (activeModelConfig(); as config) {
                        <div class="space-y-2">
                            @for (param of getCoreParams(config); track param.key) {
                                @if (!param.videoOnly || isVideo()) {
                                    <div>
                                        <div class="flex items-center justify-between mb-0.5">
                                            <label class="text-[10px] text-text-subtle">{{ param.label }}</label>
                                            @if (param.type === 'number') {
                                                <span class="text-[10px] text-text-disabled font-mono">{{ captionModelParams()[param.key] }}</span>
                                            }
                                        </div>
                                        @if (param.type === 'select') {
                                            <select [ngModel]="captionModelParams()[param.key]" 
                                                (ngModelChange)="updateParam(param.key, $event)"
                                                [attr.data-testid]="'caption-param-' + param.key"
                                                class="w-full bg-surface-low border border-surface-mid text-text-secondary text-xs rounded-theme-md px-2 py-1 outline-none focus:border-brand">
                                                @for (opt of param.options; track opt) {
                                                    <option [value]="opt">{{ opt }}</option>
                                                }
                                            </select>
                                        } @else if (param.type === 'text') {
                                            <input type="text" 
                                                [ngModel]="captionModelParams()[param.key]"
                                                (ngModelChange)="updateParam(param.key, $event)"
                                                [attr.data-testid]="'caption-param-' + param.key"
                                                [placeholder]="param.placeholder || ''"
                                                class="w-full bg-surface-low border border-surface-mid text-text-secondary text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand">
                                        } @else if (param.type === 'number') {
                                                <input type="range" [min]="param.min" [max]="param.max" [step]="param.step"
                                                [ngModel]="captionModelParams()[param.key]"
                                                (ngModelChange)="updateParam(param.key, $event)"
                                                [attr.data-testid]="'caption-param-' + param.key"
                                                class="w-full h-1 bg-surface-high rounded-theme-lg appearance-none cursor-pointer accent-brand">
                                        } @else if (param.type === 'checkbox') {
                                            <div class="flex items-center">
                                                <input type="checkbox" 
                                                    [ngModel]="captionModelParams()[param.key]"
                                                    (ngModelChange)="updateParam(param.key, $event)"
                                                    [attr.data-testid]="'caption-param-' + param.key"
                                                    class="w-3.5 h-3.5 bg-surface-low border-surface-mid rounded text-brand focus:ring-brand">
                                                    <span class="ml-2 text-[10px] text-text-muted">{{ param.label }}</span>
                                            </div>
                                        }
                                    </div>
                                }
                            }
                        </div>

                        <!-- Extra Options (collapsible) -->
                        @if (getExtraParams(config).length > 0) {
                            <div class="mt-3 border-t border-surface-mid/50 pt-2">
                                <h5 class="text-[10px] text-text-subtle uppercase tracking-wider font-bold mb-2 flex items-center justify-between cursor-pointer hover:text-brand transition-colors" (click)="showExtraOptions = !showExtraOptions">
                                    <span class="flex items-center gap-1.5">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.32 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
                                        Extra Options
                                    </span>
                                    <svg class="w-3 h-3 transition-transform" [class.rotate-180]="showExtraOptions" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                                </h5>
                                @if (showExtraOptions) {
                                    <div class="space-y-1.5 animate-fadeIn max-h-48 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-800 pr-1">
                                        @for (param of getExtraParams(config); track param.key) {
                                            @if (param.type === 'checkbox') {
                                                <div class="flex items-start gap-2">
                                                    <input type="checkbox" 
                                                        [ngModel]="captionModelParams()[param.key]"
                                                        (ngModelChange)="updateParam(param.key, $event)"
                                                        [attr.data-testid]="'caption-extra-' + param.key"
                                                        class="w-3.5 h-3.5 mt-0.5 bg-surface-low border-surface-mid rounded text-brand focus:ring-brand flex-shrink-0">
                                                    <span class="text-[10px] text-text-muted leading-tight">{{ param.label }}</span>
                                                </div>
                                            } @else if (param.type === 'text' && param.showWhen && captionModelParams()[param.showWhen]) {
                                                <div class="ml-5">
                                                    <input type="text" 
                                                        [ngModel]="captionModelParams()[param.key]"
                                                        (ngModelChange)="updateParam(param.key, $event)"
                                                        [attr.data-testid]="'caption-extra-' + param.key"
                                                        [placeholder]="param.placeholder || ''"
                                                        class="w-full bg-surface-low border border-surface-mid text-text-secondary text-xs rounded-theme-md px-2 py-1 outline-none focus:border-brand">
                                                </div>
                                            }
                                        }
                                    </div>
                                }
                            </div>
                        }
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
export class DatasetCaptionSettingsComponent implements OnInit {
    private datasetService = inject(DatasetService);

    isVideo = input(false);
    settingsChanged = output<CaptionSettingsState>();

    captionModels: any[] = [
        {
            id: 'youtu-vl',
            name: 'Youtu-VL',
            description: 'Tencent Youtu-VL-4B-Instruct. Visual Potential via Unified Vision-Language Supervision.',
            params: [
                { key: 'max_long_side', label: 'Max Image Size (longside px)', type: 'select', options: [512, 768, 1024], default: 768 },
                { key: 'max_num_patches', label: 'Max Vision Patches', type: 'select', options: [128, 256, 512], default: 256 },
                { key: 'temperature', label: 'Temperature', type: 'number', min: 0, max: 2, step: 0.1, default: 0.1 },
                { key: 'top_p', label: 'Top P', type: 'number', min: 0, max: 1, step: 0.001, default: 0.001 },
                { key: 'repetition_penalty', label: 'Repetition Penalty', type: 'number', min: 1, max: 2, step: 0.05, default: 1.05 },
                { key: 'max_tokens', label: 'Max Tokens', type: 'number', min: 64, max: 32768, step: 512, default: 512 }
            ]
        },
        {
            id: 'florence-2',
            name: 'Florence-2',
            description: 'Microsoft Florence-2. Fast and accurate.',
            params: [
                { key: 'task_type', label: 'Task Type', type: 'select', options: ['Caption', 'Detailed Caption', 'More Detailed Caption'], default: 'Detailed Caption' },
                { key: 'max_tokens', label: 'Max Tokens', type: 'number', min: 64, max: 2048, step: 64, default: 512 },
                { key: 'num_beams', label: 'Number of Beams', type: 'number', min: 1, max: 10, step: 1, default: 5 }
            ]
        },
        {
            id: 'qwen3-vl',
            name: 'Qwen3 VL',
            description: 'Alibaba Qwen3 VL. Excellent reasoning capabilities.',
            params: [
                { key: 'max_long_side', label: 'Max Image Size (longside px)', type: 'select', options: [768, 1024, 1280], default: 1280 },
                { key: 'temperature', label: 'Temperature', type: 'number', min: 0, max: 2, step: 0.1, default: 0.7 },
                { key: 'top_p', label: 'Top P', type: 'number', min: 0, max: 1, step: 0.05, default: 0.8 },
                { key: 'num_beams', label: 'Number of Beams', type: 'number', min: 1, max: 10, step: 1, default: 1 },
                { key: 'repetition_penalty', label: 'Repetition Penalty', type: 'number', min: 1, max: 2, step: 0.1, default: 1.2 },
                { key: 'max_tokens', label: 'Max Tokens', type: 'number', min: 64, max: 2048, step: 64, default: 512 },
                { key: 'frames', label: 'Video Frames', type: 'number', min: 1, max: 64, step: 1, default: 16, videoOnly: true }
            ]
        },
        {
            id: 'joycaption',
            name: 'JoyCaption',
            description: 'JoyCaption Beta. Uncensored, creative captioning for training diffusion models.',
            params: [
                { key: 'caption_type', label: 'Caption Type', type: 'select', options: ['Descriptive', 'Descriptive (Casual)', 'Straightforward', 'Stable Diffusion Prompt', 'MidJourney', 'Danbooru tag list', 'e621 tag list', 'Rule34 tag list', 'Booru-like tag list', 'Art Critic', 'Product Listing', 'Social Media Post'], default: 'Descriptive' },
                { key: 'caption_length', label: 'Caption Length', type: 'select', options: ['any', 'very short', 'short', 'medium-length', 'long', 'very long'], default: 'long' },
                { key: 'temperature', label: 'Temperature', type: 'number', min: 0, max: 2, step: 0.1, default: 0.6 },
                { key: 'top_p', label: 'Top P', type: 'number', min: 0, max: 1, step: 0.05, default: 0.9 },
                { key: 'max_tokens', label: 'Max Tokens', type: 'number', min: 64, max: 2048, step: 64, default: 512 },
                // Extra Options
                { key: 'refer_character_name', label: 'Refer to character by name', type: 'checkbox', default: false, group: 'extra' },
                { key: 'name_input', label: 'Character name', type: 'text', default: '', group: 'extra', showWhen: 'refer_character_name', placeholder: 'Enter character name...' },
                { key: 'exclude_people_info', label: 'Exclude unchangeable people info (ethnicity, gender)', type: 'checkbox', default: false, group: 'extra' },
                { key: 'include_lighting', label: 'Include lighting information', type: 'checkbox', default: false, group: 'extra' },
                { key: 'include_camera_angle', label: 'Include camera angle', type: 'checkbox', default: false, group: 'extra' },
                { key: 'include_watermark', label: 'Include watermark presence', type: 'checkbox', default: false, group: 'extra' },
                { key: 'include_jpeg_artifacts', label: 'Include JPEG artifacts', type: 'checkbox', default: false, group: 'extra' },
                { key: 'include_exif', label: 'Include camera/EXIF details', type: 'checkbox', default: false, group: 'extra' },
                { key: 'exclude_sexual', label: 'Exclude sexual content (keep PG)', type: 'checkbox', default: false, group: 'extra' },
                { key: 'exclude_resolution', label: 'Exclude image resolution', type: 'checkbox', default: false, group: 'extra' },
                { key: 'include_aesthetic_quality', label: 'Include aesthetic quality rating', type: 'checkbox', default: false, group: 'extra' },
                { key: 'include_composition', label: 'Include composition style', type: 'checkbox', default: false, group: 'extra' },
                { key: 'exclude_text', label: 'Exclude text in image', type: 'checkbox', default: false, group: 'extra' },
                { key: 'specify_depth_field', label: 'Specify depth of field', type: 'checkbox', default: false, group: 'extra' },
                { key: 'specify_lighting_sources', label: 'Specify lighting sources', type: 'checkbox', default: false, group: 'extra' },
                { key: 'no_ambiguous_language', label: 'No ambiguous language', type: 'checkbox', default: false, group: 'extra' },
                { key: 'include_nsfw_rating', label: 'Include SFW/NSFW rating', type: 'checkbox', default: false, group: 'extra' },
                { key: 'only_important_elements', label: 'Only most important elements', type: 'checkbox', default: false, group: 'extra' },
                { key: 'exclude_artist_name', label: 'Exclude artist name/title', type: 'checkbox', default: false, group: 'extra' },
                { key: 'identify_orientation', label: 'Identify orientation/aspect ratio', type: 'checkbox', default: false, group: 'extra' },
                { key: 'use_profanity', label: 'Use vulgar slang/profanity', type: 'checkbox', default: false, group: 'extra' },
                { key: 'no_euphemisms', label: 'No polite euphemisms', type: 'checkbox', default: false, group: 'extra' },
                { key: 'include_character_age', label: 'Include character age', type: 'checkbox', default: false, group: 'extra' },
                { key: 'include_shot_type', label: 'Include shot type (close-up, wide, etc)', type: 'checkbox', default: false, group: 'extra' },
                { key: 'exclude_mood', label: 'Exclude mood/feeling', type: 'checkbox', default: false, group: 'extra' },
                { key: 'include_vantage_height', label: 'Include vantage height', type: 'checkbox', default: false, group: 'extra' },
                { key: 'mention_watermark', label: 'Must mention watermark if present', type: 'checkbox', default: false, group: 'extra' },
                { key: 'avoid_meta_phrases', label: 'Avoid meta phrases ("This image shows...")', type: 'checkbox', default: false, group: 'extra' },
            ]
        },
    ];

    qwen3Variants = ['4B-Instruct', '4B-Thinking', '8B-Instruct', '8B-Thinking', '32B-Instruct', '32B-Thinking'];

    selectedCaptionModel = signal<string>('florence-2');
    selectedQwen3Variant = signal<string>('4B-Instruct');
    currentTemplates = signal<CaptionTemplate[]>([]);
    activeTemplateId = signal<string>('default');
    captionSystemPrompt = signal<string>('Describe this image in detail.');
    captionModelParams = signal<Record<string, any>>({});
    showExtraOptions = false;

    globalSettings: any = { models: {} };
    private settingsUpdate$ = new Subject<void>();

    activeModelConfig = computed(() => {
        return this.captionModels.find(m => m.id === this.selectedCaptionModel());
    });

    getCoreParams(config: any): any[] {
        return config.params.filter((p: any) => !p.group);
    }

    getExtraParams(config: any): any[] {
        return config.params.filter((p: any) => p.group === 'extra');
    }

    ngOnInit() {
        this.loadCaptionSettings();

        this.settingsUpdate$.pipe(
            debounceTime(1000)
        ).subscribe(() => {
            this.pushSettings();
        });
    }

    private loadCaptionSettings() {
        this.datasetService.getSettings('captioning').subscribe({
            next: (settings) => {
                this.globalSettings = settings || { models: {} };

                if (this.globalSettings.selected_model && this.captionModels.some(m => m.id === this.globalSettings.selected_model)) {
                    this.selectedCaptionModel.set(this.globalSettings.selected_model);
                }

                if (this.globalSettings.qwen3_variant && this.qwen3Variants.includes(this.globalSettings.qwen3_variant)) {
                    this.selectedQwen3Variant.set(this.globalSettings.qwen3_variant);
                }

                this.loadModelTemplates(this.selectedCaptionModel());
                this.emitChanges();
            },
            error: (err) => {
                this.globalSettings = { models: {} };
                this.loadModelTemplates(this.selectedCaptionModel());
                this.emitChanges();
            }
        });
    }

    private loadModelTemplates(modelId: string) {
        const modelData = this.globalSettings.models?.[modelId];
        const modelConfig = this.captionModels.find(m => m.id === modelId);

        const defaultParams: Record<string, any> = {};
        modelConfig?.params.forEach((p: any) => {
            defaultParams[p.key] = p.default;
        });
        const defaultTemplate: CaptionTemplate = {
            id: 'default',
            name: 'Default',
            is_default: true,
            readonly: true,
            system_prompt: 'Describe this image in detail.',
            params: defaultParams
        };

        if (modelData && modelData.templates && Array.isArray(modelData.templates)) {
            this.currentTemplates.set(modelData.templates);
            this.activeTemplateId.set(modelData.active_template_id || 'default');
        } else if (modelData && modelData.params) {
            const legacyTemplate: CaptionTemplate = {
                id: `tpl_${Date.now()}`,
                name: 'Default by User',
                is_default: false,
                readonly: false,
                system_prompt: modelData.systemPrompt || defaultTemplate.system_prompt,
                params: { ...defaultParams, ...modelData.params }
            };

            const templates = [defaultTemplate, legacyTemplate];
            this.currentTemplates.set(templates);
            this.activeTemplateId.set(legacyTemplate.id);
            this.saveModelSettings(modelId);
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
            this.captionSystemPrompt.set(tpl.system_prompt);

            // Merge code defaults with stored params so new params get default values
            const modelConfig = this.captionModels.find(m => m.id === this.selectedCaptionModel());
            const codeDefaults: Record<string, any> = {};
            modelConfig?.params.forEach((p: any) => { codeDefaults[p.key] = p.default; });

            this.captionModelParams.set({ ...codeDefaults, ...tpl.params });
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
        this.datasetService.saveSettings('captioning', this.globalSettings).subscribe();
    }

    onModelChange(modelId: string) {
        if (modelId === this.selectedCaptionModel()) return;
        this.datasetService.unloadModels().subscribe();
        this.saveModelSettings(this.selectedCaptionModel());
        this.selectedCaptionModel.set(modelId);
        this.loadModelTemplates(modelId);
        this.globalSettings.selected_model = modelId;
        this.settingsUpdate$.next();
        this.emitChanges();
    }

    onVariantChange(variant: string) {
        if (variant === this.selectedQwen3Variant()) return;
        this.datasetService.unloadModels().subscribe();
        this.selectedQwen3Variant.set(variant);
        this.globalSettings.qwen3_variant = variant;
        this.settingsUpdate$.next();
        this.emitChanges();
    }

    onTemplateChange(tplId: string) {
        this.activeTemplateId.set(tplId);
        this.applyActiveTemplate();
        this.saveModelSettings(this.selectedCaptionModel());
    }

    updateParam(key: string, value: any) {
        const currentParams = this.captionModelParams();
        const newParams = { ...currentParams, [key]: value };
        this.updateActiveTemplate({ params: newParams });
    }

    onSystemPromptChange(prompt: string) {
        this.updateActiveTemplate({ system_prompt: prompt });
    }

    private updateActiveTemplate(changes: { params?: Record<string, any>; system_prompt?: string }) {
        const activeId = this.activeTemplateId();
        const templates = this.currentTemplates();
        let activeTpl = templates.find(t => t.id === activeId);

        if (!activeTpl) return;

        if (activeTpl.readonly) {
            const newTpl: CaptionTemplate = {
                id: `tpl_${Date.now()}`,
                name: 'Default by User',
                system_prompt: changes.system_prompt ?? activeTpl.system_prompt,
                params: { ...activeTpl.params, ...(changes.params || {}) }
            };
            this.currentTemplates.update(ts => [...ts, newTpl]);
            this.activeTemplateId.set(newTpl.id);
        } else {
            this.currentTemplates.update(ts => ts.map(t => {
                if (t.id === activeId) {
                    return {
                        ...t,
                        system_prompt: changes.system_prompt ?? t.system_prompt,
                        params: { ...t.params, ...(changes.params || {}) }
                    };
                }
                return t;
            }));
        }

        this.applyActiveTemplate();
        this.saveModelSettings(this.selectedCaptionModel());
    }

    addTemplate() {
        const name = prompt('Template name:');
        if (!name) return;

        const newTpl: CaptionTemplate = {
            id: `tpl_${Date.now()}`,
            name,
            system_prompt: this.captionSystemPrompt(),
            params: { ...this.captionModelParams() }
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

        this.saveModelSettings(this.selectedCaptionModel());
    }

    deleteTemplate() {
        const activeId = this.activeTemplateId();
        if (activeId === 'default') return;

        if (!confirm('Delete this template?')) return;

        this.currentTemplates.update(ts => ts.filter(t => t.id !== activeId));
        this.onTemplateChange('default');
    }

    private emitChanges() {
        const modelId = this.selectedCaptionModel();
        const variant = this.selectedQwen3Variant();
        const resolvedModelId = modelId === 'qwen3-vl' ? `qwen3-vl-${variant}` : modelId;

        this.settingsChanged.emit({
            modelId: modelId,
            resolvedModelId: resolvedModelId,
            variant: modelId === 'qwen3-vl' ? variant : undefined,
            systemPrompt: this.captionSystemPrompt(),
            params: this.captionModelParams()
        });
    }
}
