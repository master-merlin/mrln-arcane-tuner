
import { Component, OnInit, inject, signal, computed, input, output, effect, untracked, ChangeDetectionStrategy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DatasetService } from '../../../services/dataset';
import { ProjectService, ProjectPreferences } from '../../../services/project.service';
import { TemplateService, Template } from '../../../services/template.service';
import { ApiCaptionService, ApiProviderStatus } from '../../../services/api-caption.service';
import { ModelContextStore } from '../../../state/model-context.store';
import { Subject } from 'rxjs';
import { debounceTime, switchMap } from 'rxjs/operators';

export interface CaptionTemplate extends Template {}

/** A single tunable parameter on a caption model (see {@link DatasetCaptionSettingsComponent.captionModels}). */
export interface CaptionParam {
    key: string;
    label: string;
    type: 'select' | 'number' | 'checkbox' | 'text';
    default: string | number | boolean;
    /** Choices for `select` (may be numeric, e.g. image sizes). */
    options?: (string | number)[];
    min?: number;
    max?: number;
    step?: number;
    /** `'extra'` params live in the collapsible "Extra Options" panel. */
    group?: 'extra';
    /** Only render for video media. */
    videoOnly?: boolean;
    /** Param key whose truthiness gates this one (extra `text` rows). */
    showWhen?: string;
    placeholder?: string;
}

/** A caption method and its tunable parameters. */
export interface CaptionModelConfig {
    id: string;
    name: string;
    description: string;
    params: CaptionParam[];
    /** Can caption from control + target together (edit-instruction captions).
     *  The UI greys out "include control images" for models without it. */
    supportsMultiImage?: boolean;
}

/** Shared tunables for every API provider (model name is set per template). */
const API_PROVIDER_PARAMS: CaptionParam[] = [
    { key: 'model', label: 'Model', type: 'text', default: '', placeholder: 'e.g. gpt-4o' },
    { key: 'temperature', label: 'Temperature', type: 'number', min: 0, max: 2, step: 0.1, default: 0.7 },
    { key: 'top_p', label: 'Top P', type: 'number', min: 0, max: 1, step: 0.05, default: 1.0 },
    { key: 'max_tokens', label: 'Max Tokens', type: 'number', min: 64, max: 8192, step: 64, default: 512 },
    { key: 'max_long_side', label: 'Max Upload Size (longside px)', type: 'select', options: [512, 768, 1024, 1280], default: 1024 },
];

export interface CaptionSettingsState {
    modelId: string;
    resolvedModelId: string;
    variant?: string;
    /** Raw prompt as authored — may contain {wildcard} tokens. Persisted to
     *  the template verbatim so the stored prompt stays clean. */
    systemPrompt: string;
    /** Prompt with every {wildcard} token replaced by `wildcard`. This is what
     *  callers send to the captioning model. */
    resolvedSystemPrompt: string;
    /** The wildcard substitution value (per-template). */
    wildcard: string;
    params: Record<string, unknown>;
    /** API mode only: whether the selected provider has usable credentials.
     *  Undefined in local mode. Hosts should disable generation when false. */
    apiConfigured?: boolean;
    /** Whether the selected model can caption from control + target together
     *  (edit-instruction captions). Hosts gate the "include control" toggle. */
    supportsMultiImage?: boolean;
    /** Additional instructions for structured-caption (model-aware) generation.
     *  Empty string when plain format or no instructions entered. */
    captionInstructions: string;
}

@Component({
    selector: 'app-dataset-caption-settings',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [FormsModule],
    template: `
        <div class="space-y-3 animate-fadeIn">
            <!-- Local / API mode tabs -->
            <div class="flex gap-1 bg-surface-low/60 p-1 rounded-theme-lg border border-surface-mid/50">
                <button type="button" data-testid="caption-mode-local"
                    (click)="switchMode('local')"
                    [class.bg-surface-high]="captionMode() === 'local'"
                    [class.text-brand]="captionMode() === 'local'"
                    class="flex-1 text-[11px] font-bold uppercase tracking-wider py-1.5 rounded-theme-md text-text-subtle transition-colors">
                    Local
                </button>
                <button type="button" data-testid="caption-mode-api"
                    (click)="switchMode('api')"
                    [class.bg-surface-high]="captionMode() === 'api'"
                    [class.text-brand]="captionMode() === 'api'"
                    class="flex-1 text-[11px] font-bold uppercase tracking-wider py-1.5 rounded-theme-md text-text-subtle transition-colors">
                    API
                </button>
            </div>

            <!-- Model Selector Row -->
            <div class="bg-surface-high/40 p-3 rounded-theme-lg border border-surface-high/50">
                <div class="flex gap-2 mb-2">
                    <div class="flex-1">
                            <label class="text-[10px] uppercase tracking-wider text-text-subtle font-bold mb-1 block">{{ captionMode() === 'api' ? 'Provider' : 'Model' }}</label>
                            <select [ngModel]="selectedCaptionModel()" (ngModelChange)="onModelChange($event)"
                                   data-testid="caption-model-select"
                            class="w-full bg-surface-low border border-surface-mid text-text-primary text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand transition-colors">
                            @for (model of (captionMode() === 'api' ? apiProviders : captionModels); track model.id) {
                                <option [value]="model.id">{{ model.name }}</option>
                            }
                        </select>
                    </div>
                    @if (selectedCaptionModel() === 'qwen3-vl') {
                        <div class="w-1/3">
                            <label class="text-[10px] uppercase tracking-wider text-text-subtle font-bold mb-1 block">Variant</label>
                            <select [ngModel]="selectedQwen3Variant()" (ngModelChange)="onVariantChange($event)"
                                   data-testid="qwen3-variant-select"
                                class="w-full bg-surface-low border border-surface-mid text-text-primary text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand transition-colors">
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

            <!-- API connection card (API mode only) -->
            @if (captionMode() === 'api') {
                <div class="bg-surface-high/40 p-3 rounded-theme-lg border border-surface-high/50 space-y-2">
                    <div class="flex items-center justify-between">
                        <label class="text-[10px] uppercase tracking-wider text-text-subtle font-bold">Connection</label>
                        @if (activeProviderStatus(); as st) {
                            <span class="text-[10px] font-mono"
                                  [class.text-success]="st.configured" [class.text-danger]="!st.configured"
                                  data-testid="api-key-status">
                                {{ st.configured ? (st.key_masked || 'configured') + ' ✓' : 'not configured' }}
                            </span>
                        } @else if (providerStatusError()) {
                            <span class="text-[10px] text-danger" data-testid="api-status-error">{{ providerStatusError() }}</span>
                        }
                    </div>
                    @if (selectedCaptionModel() === 'api-custom') {
                        <input type="text" [ngModel]="baseUrlInput()" (ngModelChange)="baseUrlInput.set($event)"
                            data-testid="api-base-url"
                            placeholder="Base URL, e.g. http://localhost:11434/v1"
                            class="w-full bg-surface-low border border-surface-mid text-text-secondary text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand">
                    }
                    <div class="flex gap-2">
                        <input type="password" [ngModel]="keyInput()" (ngModelChange)="keyInput.set($event)"
                            data-testid="api-key-input"
                            [placeholder]="selectedCaptionModel() === 'api-custom' ? 'API key (optional)' : 'Paste API key…'"
                            class="flex-1 bg-surface-low border border-surface-mid text-text-secondary text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand">
                        <button type="button" (click)="saveProviderCredentials()"
                            data-testid="api-key-save"
                            class="px-3 text-[10px] font-bold uppercase tracking-wider bg-surface-mid hover:bg-surface-high text-brand rounded-theme-md border border-surface-high transition-colors">
                            Save
                        </button>
                    </div>

                    <!-- Provider model: free text until a fetch succeeds, then a select -->
                    <div>
                        <div class="flex items-center justify-between mb-0.5">
                            <label class="text-[10px] text-text-subtle">Model</label>
                            <button type="button" (click)="fetchProviderModels()"
                                data-testid="api-fetch-models"
                                class="text-[10px] text-brand hover:underline">Fetch models</button>
                        </div>
                        @if (fetchedModels().length > 0) {
                            <select [ngModel]="captionModelParams()['model']" (ngModelChange)="updateParam('model', $event)"
                                data-testid="api-model-select"
                                class="w-full bg-surface-low border border-surface-mid text-text-secondary text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand">
                                @for (m of fetchedModels(); track m) { <option [value]="m">{{ m }}</option> }
                            </select>
                        } @else {
                            <input type="text" [ngModel]="captionModelParams()['model']" (ngModelChange)="updateParam('model', $event)"
                                data-testid="api-model-input" placeholder="e.g. gpt-4o"
                                class="w-full bg-surface-low border border-surface-mid text-text-secondary text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand">
                        }
                        @if (fetchModelsError()) {
                            <p class="text-[10px] text-danger mt-1">{{ fetchModelsError() }}</p>
                        }
                    </div>
                </div>
            }

            <!-- Template Card -->
            <div class="bg-surface-high/40 rounded-theme-lg border border-surface-mid/50 overflow-hidden">
                    <!-- Template Header & Actions -->
                    @if (!hideTemplateBar()) {
                    <div class="p-3 bg-surface-low/50 border-b border-surface-mid/50 flex items-end gap-2">
                        <div class="flex-1">
                            <label class="text-[10px] uppercase tracking-wider text-text-subtle font-bold mb-1 block">Settings Template</label>
                            <select [ngModel]="activeTemplateId()" (ngModelChange)="onTemplateChange($event)"
                                   data-testid="caption-template-select"
                                class="w-full bg-surface-low border border-surface-high text-text-primary text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand transition-colors">
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
                            [disabled]="isDefaultTemplate()" [class.opacity-50]="isDefaultTemplate()" class="p-1.5 bg-surface-mid hover:bg-surface-high text-yellow-500 rounded-theme-md border border-surface-high transition-colors" title="Rename Template">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
                    </button>
                    <button (click)="deleteTemplate()"
                            data-testid="delete-caption-template-btn"
                            [disabled]="isDefaultTemplate()" [class.opacity-50]="isDefaultTemplate()" class="p-1.5 bg-surface-mid hover:bg-danger/20 text-danger rounded-theme-md border border-surface-high transition-colors" title="Delete Template">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                    </button>
                    </div>
                    }

                    <!-- Params Content -->
                    <div class="p-3 space-y-3">
                    <!-- Wildcard -->
                    <div>
                        <label class="text-[10px] text-text-subtle mb-1 block flex items-center gap-1.5"
                            title="Replaces every {wildcard} token in the System Prompt with this value before captioning. Lets you reuse a name (or any term) in several places while keeping the stored prompt itself clean.">
                            Wildcard
                            <span class="text-text-disabled font-normal normal-case">— fills <span class="font-mono text-brand-light">{{ '{wildcard}' }}</span></span>
                        </label>
                        <input type="text" [value]="captionWildcard()"
                            (input)="onWildcardChange($any($event.target).value)"
                            data-testid="caption-wildcard"
                            title="Replaces every {wildcard} token in the System Prompt before captioning."
                            class="w-full bg-surface-low border border-surface-mid text-text-secondary text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand transition-colors placeholder-gray-600"
                            placeholder="e.g. a name to inject">
                    </div>

                    <!-- System Prompt — grows when Detailed Settings is collapsed -->
                    <div>
                        <label class="text-[10px] text-text-subtle mb-1 block">System Prompt</label>
                        <textarea [value]="captionSystemPrompt()"
                            (input)="onSystemPromptChange($any($event.target).value)"
                            data-testid="caption-system-prompt"
                            [attr.rows]="showDetailedSettings() ? 3 : 8"
                            class="w-full bg-surface-low border border-surface-mid text-text-secondary text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand transition-colors resize-none placeholder-gray-600"
                            placeholder="System Prompt..."></textarea>
                    </div>

                    <!-- Additional instructions — only shown for structured (model-aware) formats -->
                    @if (modelContext.activeCaptionFormat() !== 'plain') {
                        <div>
                            <label class="text-[10px] text-text-subtle mb-1 block"
                                title="Optional extra guidance appended to the structured caption prompt. Use to focus on specific aspects, style, or details.">
                                Additional instructions
                                <span class="text-text-disabled font-normal normal-case ml-1">— optional guidance for this re-caption</span>
                            </label>
                            <textarea [value]="captionInstructions()"
                                (input)="captionInstructions.set($any($event.target).value)"
                                data-testid="caption-additional-instructions"
                                rows="2"
                                class="w-full bg-surface-low border border-surface-mid text-text-secondary text-xs rounded-theme-md px-2 py-1.5 outline-none focus:border-brand transition-colors resize-none placeholder-gray-600"
                                placeholder="e.g. focus on composition and lighting…"></textarea>
                        </div>
                    }

                        <!-- Detailed Settings (collapsible) — all model params live here -->
                    @if (activeModelConfig(); as config) {
                        <div class="border-t border-surface-mid/50 pt-2">
                            <h5 class="text-[10px] text-text-subtle uppercase tracking-wider font-bold mb-2 flex items-center justify-between cursor-pointer hover:text-brand transition-colors"
                                data-testid="caption-detailed-toggle"
                                (click)="showDetailedSettings.set(!showDetailedSettings())">
                                <span class="flex items-center gap-1.5">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>
                                    Detailed Settings
                                </span>
                                <svg class="w-3 h-3 transition-transform" [class.rotate-180]="showDetailedSettings()" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                            </h5>
                        @if (showDetailedSettings()) {
                        <div class="space-y-3 animate-fadeIn">
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
                                                @for (opt of param.options ?? []; track opt) {
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
        /* Honor an inherited --form-accent override (e.g. the green mass-mask
           modal) and fall back to brand otherwise — matching the global
           checkbox/control convention in styles.css. The prior auto fallback
           rendered native blue in the detail sidebar / mass-caption modal. */
        input[type="range"] {
            accent-color: var(--form-accent, var(--color-brand));
        }
    `]
})
export class DatasetCaptionSettingsComponent implements OnInit {
    private datasetService = inject(DatasetService);
    private projectService = inject(ProjectService);
    private templateService = inject(TemplateService);
    /** Injected now; the API connection UI (Task 9) consumes it. */
    private apiCaptionService = inject(ApiCaptionService);
    /** Exposes activeCaptionFormat() and activeDefinitionId() for the template. */
    protected modelContext = inject(ModelContextStore);

    projectId = input<string | null>(null);
    effectiveProjectId = computed(() => this.projectId() ?? this.projectService.activeDatasetProject());
    isVideo = input(false);
    settingsChanged = output<CaptionSettingsState>();

    /** Edit-modal support. Defaults preserve the mass-modal behavior:
     *  - hideTemplateBar: hide the SETTINGS TEMPLATE switcher + +/edit/trash row.
     *  - autoSave: when false, edits stay local (just emit settingsChanged) — the
     *    host owns persistence via an explicit Save.
     *  - presetTemplate: load this one template (model + params + prompt) on open,
     *    bypassing preference-driven selection. */
    hideTemplateBar = input(false);
    autoSave = input(true);
    presetTemplate = input<Template | null>(null);

    captionModels: CaptionModelConfig[] = [
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
            supportsMultiImage: true,
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

    /** External API providers — rendered under the API tab. Each is a pseudo
     *  caption model (id `api-<provider>`) so templates/preferences/batch
     *  reuse the local-model machinery unchanged. */
    apiProviders: CaptionModelConfig[] = [
        { id: 'api-openai', name: 'OpenAI', description: 'OpenAI vision models (gpt-4o, gpt-4o-mini, …) via api.openai.com.', params: API_PROVIDER_PARAMS, supportsMultiImage: true },
        { id: 'api-anthropic', name: 'Anthropic', description: 'Claude vision models via the OpenAI-compatible endpoint.', params: API_PROVIDER_PARAMS, supportsMultiImage: true },
        { id: 'api-gemini', name: 'Gemini', description: 'Google Gemini vision models via the OpenAI-compatible endpoint.', params: API_PROVIDER_PARAMS, supportsMultiImage: true },
        { id: 'api-openrouter', name: 'OpenRouter', description: 'Any OpenRouter-hosted vision model behind one key.', params: API_PROVIDER_PARAMS, supportsMultiImage: true },
        { id: 'api-custom', name: 'Local / Custom', description: 'Any OpenAI-compatible server: Ollama, LM Studio, vLLM. Set the Base URL below.', params: API_PROVIDER_PARAMS, supportsMultiImage: true },
    ];

    /** Local models + API providers — single lookup space for model ids. */
    get allModelConfigs(): CaptionModelConfig[] {
        return [...this.captionModels, ...this.apiProviders];
    }

    qwen3Variants = ['4B-Instruct', '4B-Thinking', '8B-Instruct', '8B-Thinking', '32B-Instruct', '32B-Thinking'];

    selectedCaptionModel = signal<string>('florence-2');

    /** Explicit Local/API tab. Derived from the active model id so persisted
     *  api-* preferences reopen on the API tab. */
    captionMode = computed<'local' | 'api'>(() =>
        this.selectedCaptionModel().startsWith('api-') ? 'api' : 'local');

    /** Last-used model per tab, restored on switch (session-scoped). */
    private lastLocalModelId = 'florence-2';
    private lastApiModelId = 'api-openai';

    switchMode(mode: 'local' | 'api') {
        if (mode === this.captionMode()) return;
        this.onModelChange(mode === 'api' ? this.lastApiModelId : this.lastLocalModelId);
    }

    selectedQwen3Variant = signal<string>('4B-Instruct');
    currentTemplates = signal<Template[]>([]);
    activeTemplateId = signal<string | null>(null);
    captionSystemPrompt = signal<string>('Describe this image in detail.');
    /** Wildcard value substituted into {wildcard} tokens of the system prompt. */
    captionWildcard = signal<string>('');
    captionModelParams = signal<Record<string, unknown>>({});
    showExtraOptions = false;
    /** Collapsible holding all model-detail params. Collapsed by default so the
     *  top items (System Prompt) get the room; expanding shrinks the prompt. */
    showDetailedSettings = signal<boolean>(false);
    /** Additional instructions sent as `caption_instructions` param for
     *  structured-caption (model-aware) generation. Not persisted in templates;
     *  user sets it per session. Empty string means no extra instructions. */
    captionInstructions = signal<string>('');

    // ── API provider connection state ──
    providerStatuses = signal<ApiProviderStatus[]>([]);
    keyInput = signal('');
    baseUrlInput = signal('');
    fetchedModels = signal<string[]>([]);
    fetchModelsError = signal('');
    /** Non-empty when GET /captions/api-providers failed — rendered in place of
     *  the status badge so hosts' disabled CTAs aren't left unexplained. */
    providerStatusError = signal('');

    /** Status entry for the currently selected api-* provider. */
    activeProviderStatus = computed(() => {
        const id = this.selectedCaptionModel();
        if (!id.startsWith('api-')) return undefined;
        return this.providerStatuses().find(s => `api-${s.provider}` === id);
    });

    private get activeProviderName(): string {
        return this.selectedCaptionModel().replace(/^api-/, '');
    }

    private loadProviderStatuses() {
        this.providerStatusError.set('');
        this.apiCaptionService.listProviders().subscribe({
            next: statuses => {
                this.providerStatuses.set(statuses);
                // Opening with a persisted api-custom selection never goes through
                // onModelChange, so seed the Base URL field from the status here.
                if (!this.baseUrlInput()) {
                    this.baseUrlInput.set(this.activeProviderStatus()?.base_url ?? '');
                }
                this.emitChanges();
            },
            error: () => {
                this.providerStatusError.set('Could not load provider status.');
                // Still re-emit so hosts get a definitive apiConfigured=false
                // instead of waiting on a settingsChanged that never comes.
                this.emitChanges();
            },
        });
    }

    saveProviderCredentials() {
        const updates: { api_key?: string; base_url?: string } = {};
        if (this.keyInput().trim()) updates.api_key = this.keyInput().trim();
        if (this.activeProviderName === 'custom' && this.baseUrlInput().trim()) {
            updates.base_url = this.baseUrlInput().trim();
        }
        if (Object.keys(updates).length === 0) return;
        this.apiCaptionService.updateProvider(this.activeProviderName, updates)
            .subscribe({
                next: status => {
                    this.providerStatusError.set('');
                    this.providerStatuses.update(list => {
                        const rest = list.filter(s => s.provider !== status.provider);
                        return [...rest, status];
                    });
                    this.keyInput.set('');
                    this.emitChanges();
                },
                // Keep the typed key so the user can retry without re-pasting.
                error: () => this.providerStatusError.set('Could not save credentials.'),
            });
    }

    fetchProviderModels() {
        this.fetchModelsError.set('');
        this.apiCaptionService.listModels(this.activeProviderName).subscribe({
            next: models => this.fetchedModels.set(models),
            error: () => this.fetchModelsError.set(
                'Could not fetch models — check the key/base URL.'),
        });
    }

    private preferences: ProjectPreferences | null = null;
    private settingsUpdate$ = new Subject<void>();
    private pendingSaves = new Map<string, { config: Record<string, unknown>; system_prompt?: string; wildcard?: string }>();

    activeModelConfig = computed(() => {
        return this.allModelConfigs.find(m => m.id === this.selectedCaptionModel());
    });

    isDefaultTemplate() {
        const id = this.activeTemplateId();
        const templates = this.currentTemplates();
        const tpl = templates.find(t => t.id === id);
        return tpl ? tpl.is_default || tpl.readonly : false;
    }

    getCoreParams(config: CaptionModelConfig): CaptionParam[] {
        return config.params.filter(p => !p.group && p.key !== 'model');
    }

    getExtraParams(config: CaptionModelConfig): CaptionParam[] {
        return config.params.filter(p => p.group === 'extra');
    }

    constructor() {
        effect(() => {
            const preset = this.presetTemplate();
            // Apply the preset OUTSIDE tracking: applyPreset → applyActiveTemplate →
            // emitChanges() reads captionModelParams/systemPrompt synchronously, which
            // would otherwise make those signals dependencies of this effect. Then every
            // edit (updateParam/onSystemPromptChange writes them) would re-run the effect,
            // re-apply the preset, and wipe the user's edit. untracked() breaks that loop
            // so the preset is applied once. The effect still reacts to presetTemplate itself.
            if (preset) { untracked(() => this.applyPreset(preset)); return; }
            const pid = this.effectiveProjectId();
            // untracked for the same reason as the preset path: with synchronous
            // observables (tests), the subscribe callbacks run inside this effect's
            // tracked execution, making activeTemplateId/currentTemplates accidental
            // dependencies — every edit would then re-run the effect and clobber the
            // user's selection with a reload. React to preset + project id only.
            untracked(() => this.loadPreferencesAndTemplates());
        });
    }

    /** Edit-modal: load a single template into the UI, bypassing preference-driven
     *  selection and the template list fetch. */
    private applyPreset(tpl: Template) {
        if (tpl.model_id && this.allModelConfigs.some(m => m.id === tpl.model_id)) {
            this.selectedCaptionModel.set(tpl.model_id);
        }
        this.currentTemplates.set([tpl]);
        this.activeTemplateId.set(tpl.id);
        this.applyActiveTemplate();
    }

    ngOnInit() {
        this.loadProviderStatuses();
        this.settingsUpdate$.pipe(
            debounceTime(1000),
            switchMap(() => {
                if (!this.preferences) return [];
                return this.projectService.updatePreferences(this.effectiveProjectId(), {
                    selected_caption_model: this.selectedCaptionModel(),
                    qwen3_variant: this.selectedQwen3Variant(),
                    active_caption_template: this.activeTemplateId()
                });
            })
        ).subscribe();
    }

    private loadPreferencesAndTemplates() {
        const pId = this.effectiveProjectId();
        this.projectService.getPreferences(pId).pipe(
            switchMap(prefs => {
                this.preferences = prefs;
                if (prefs.selected_caption_model && this.allModelConfigs.some(m => m.id === prefs.selected_caption_model)) {
                    this.selectedCaptionModel.set(prefs.selected_caption_model);
                    if (prefs.selected_caption_model.startsWith('api-')) this.lastApiModelId = prefs.selected_caption_model;
                    else this.lastLocalModelId = prefs.selected_caption_model;
                }
                if (prefs.qwen3_variant && this.qwen3Variants.includes(prefs.qwen3_variant)) {
                    this.selectedQwen3Variant.set(prefs.qwen3_variant);
                }
                return this.templateService.listCaptioningTemplates(this.selectedCaptionModel(), pId);
            })
        ).subscribe({
            next: templates => {
                this.currentTemplates.set(templates);
                if (this.preferences?.active_caption_template && templates.some(t => t.id === this.preferences!.active_caption_template)) {
                    this.activeTemplateId.set(this.preferences.active_caption_template);
                } else {
                    const defaultTpl = templates.find(t => t.is_default);
                    this.activeTemplateId.set(defaultTpl ? defaultTpl.id : (templates.length > 0 ? templates[0].id : null));
                }
                this.applyActiveTemplate();
            },
            // Best-effort init load: on failure keep the code defaults instead of
            // surfacing an unhandled rejection (zoneless has no zone to absorb it).
            error: err => console.warn('caption preferences/templates load failed', err),
        });
    }

    private loadModelTemplates(modelId: string) {
        this.templateService.listCaptioningTemplates(modelId, this.effectiveProjectId()).subscribe(templates => {
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
            this.captionSystemPrompt.set(tpl.system_prompt || '');
            this.captionWildcard.set(tpl.wildcard || '');
        }

        const modelConfig = this.allModelConfigs.find(m => m.id === this.selectedCaptionModel());
        const codeDefaults: Record<string, unknown> = {};
        modelConfig?.params.forEach(p => { codeDefaults[p.key] = p.default; });

        // No template for this model (no seeded default yet): fall back to the
        // model's code defaults instead of keeping the previous model's params.
        this.captionModelParams.set({ ...codeDefaults, ...(tpl?.config || {}) });
        this.emitChanges();
    }

    onModelChange(modelId: string) {
        if (modelId === this.selectedCaptionModel()) return;
        this.datasetService.unloadModels().subscribe();
        this.selectedCaptionModel.set(modelId);
        if (modelId.startsWith('api-')) this.lastApiModelId = modelId;
        else this.lastLocalModelId = modelId;
        this.fetchedModels.set([]);
        this.fetchModelsError.set('');
        this.baseUrlInput.set(this.activeProviderStatus()?.base_url ?? '');
        if (!this.autoSave()) {
            // Edit mode: switch model locally, reset params to its defaults; the
            // host's Save persists. No backend template list reload.
            const cfg = this.allModelConfigs.find(m => m.id === modelId);
            const defaults: Record<string, unknown> = {};
            cfg?.params.forEach(p => { defaults[p.key] = p.default; });
            this.captionModelParams.set(defaults);
            this.emitChanges();
            return;
        }
        this.loadModelTemplates(modelId);
        this.emitChanges();
    }

    onVariantChange(variant: string) {
        if (variant === this.selectedQwen3Variant()) return;
        this.datasetService.unloadModels().subscribe();
        this.selectedQwen3Variant.set(variant);
        this.settingsUpdate$.next();
        this.emitChanges();
    }

    onTemplateChange(tplId: string) {
        this.activeTemplateId.set(tplId);
        this.applyActiveTemplate();
        this.settingsUpdate$.next();
    }

    updateParam(key: string, value: unknown) {
        if (!this.autoSave()) {
            this.captionModelParams.update(p => ({ ...p, [key]: value }));
            this.emitChanges();
            return;
        }
        const newParams = { ...this.captionModelParams(), [key]: value };
        this.updateActiveTemplate({ config: newParams });
    }

    onSystemPromptChange(prompt: string) {
        if (!this.autoSave()) {
            this.captionSystemPrompt.set(prompt);
            this.emitChanges();
            return;
        }
        this.updateActiveTemplate({ system_prompt: prompt });
    }

    /** Wildcard edits persist with the template (per the chosen design); in
     *  edit-modal mode (autoSave off) they stay local until the host's Save. */
    onWildcardChange(wildcard: string) {
        if (!this.autoSave()) {
            this.captionWildcard.set(wildcard);
            this.emitChanges();
            return;
        }
        this.updateActiveTemplate({ wildcard });
    }

    /** Replace every {wildcard} token in `prompt` with the current wildcard
     *  value. Empty wildcard removes the token rather than sending literal
     *  braces to the model. */
    private resolveWildcard(prompt: string): string {
        return prompt.replace(/\{wildcard\}/g, this.captionWildcard());
    }

    /** True while the copy-on-edit POST is in flight; buffers the newest change
     *  so rapid edits can't spawn duplicate 'Custom Settings' rows. */
    private creatingCopy = false;
    private pendingCopyChanges: { config?: Record<string, unknown>; system_prompt?: string; wildcard?: string } | null = null;

    private updateActiveTemplate(changes: { config?: Record<string, unknown>; system_prompt?: string; wildcard?: string }) {
        if (this.creatingCopy) {
            this.pendingCopyChanges = {
                ...(this.pendingCopyChanges || {}),
                ...changes,
                config: { ...(this.pendingCopyChanges?.config || {}), ...(changes.config || {}) },
            };
            return;
        }

        const activeId = this.activeTemplateId();
        const templates = this.currentTemplates();
        const activeTpl = templates.find(t => t.id === activeId);

        // System defaults (readonly or is_default) are never written through,
        // and a model without any template must not silently drop the edit:
        // both save into the user's 'Custom Settings' copy for this model —
        // reused when it already exists, created exactly once otherwise.
        if (!activeTpl || activeTpl.readonly || activeTpl.is_default) {
            const existing = templates.find(t =>
                t.name === 'Custom Settings' && !t.readonly && !t.is_default &&
                t.model_id === this.selectedCaptionModel());
            if (existing) {
                const merged = {
                    system_prompt: changes.system_prompt ?? existing.system_prompt ?? '',
                    wildcard: changes.wildcard ?? existing.wildcard ?? '',
                    config: { ...(existing.config || {}), ...(changes.config || {}) },
                };
                this.activeTemplateId.set(existing.id);
                this.currentTemplates.update(ts => ts.map(t => t.id === existing.id ? { ...t, ...merged } : t));
                this.applyActiveTemplate();
                this.settingsUpdate$.next();
                this.saveEditableTemplate(existing.id, merged);
                return;
            }

            const systemPrompt = changes.system_prompt ?? activeTpl?.system_prompt ?? this.captionSystemPrompt() ?? '';
            const wildcard = changes.wildcard ?? activeTpl?.wildcard ?? this.captionWildcard() ?? '';
            const config = { ...(activeTpl?.config || {}), ...(changes.config || {}) };

            this.creatingCopy = true;
            this.templateService.createCaptioningTemplate({
                model_id: this.selectedCaptionModel(),
                name: 'Custom Settings',
                project_id: this.effectiveProjectId(),
                system_prompt: systemPrompt,
                wildcard: wildcard,
                config: config
            }).subscribe({
                next: newTpl => {
                    this.creatingCopy = false;
                    this.currentTemplates.update(ts => [...ts, newTpl]);
                    this.activeTemplateId.set(newTpl.id);
                    this.applyActiveTemplate();
                    this.settingsUpdate$.next();
                    const pending = this.pendingCopyChanges;
                    this.pendingCopyChanges = null;
                    if (pending) this.updateActiveTemplate(pending);
                },
                error: () => {
                    this.creatingCopy = false;
                    this.pendingCopyChanges = null;
                },
            });
            return;
        }

        const systemPrompt = changes.system_prompt ?? activeTpl.system_prompt ?? '';
        const wildcard = changes.wildcard ?? activeTpl.wildcard ?? '';
        const config = { ...(activeTpl.config || {}), ...(changes.config || {}) };

        this.currentTemplates.update(ts => ts.map(t => {
            if (t.id === activeId) {
                return { ...t, system_prompt: systemPrompt, wildcard: wildcard, config: config };
            }
            return t;
        }));
        this.applyActiveTemplate();
        this.saveEditableTemplate(activeTpl.id, { system_prompt: systemPrompt, wildcard, config });
    }

    /** Debounced write-back for an editable template (coalesces rapid edits). */
    private saveEditableTemplate(id: string, data: { config: Record<string, unknown>; system_prompt?: string; wildcard?: string }) {
        this.pendingSaves.set(id, data);

        setTimeout(() => {
            const pending = this.pendingSaves.get(id);
            if (pending) {
                this.pendingSaves.delete(id);
                this.templateService.updateTemplate('captioning', id, pending).subscribe();
            }
        }, 500);
    }

    addTemplate() {
        const name = prompt('Template name:');
        if (!name) return;

        this.templateService.createCaptioningTemplate({
            model_id: this.selectedCaptionModel(),
            name,
            project_id: this.effectiveProjectId(),
            system_prompt: this.captionSystemPrompt(),
            wildcard: this.captionWildcard(),
            config: this.captionModelParams()
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

        this.templateService.updateTemplate('captioning', activeId, { name }).subscribe(updatedTpl => {
            this.currentTemplates.update(ts => ts.map(t => t.id === activeId ? updatedTpl : t));
        });
    }

    deleteTemplate() {
        const activeId = this.activeTemplateId();
        if (!activeId || this.isDefaultTemplate()) return;

        if (!confirm('Delete this template?')) return;

        this.templateService.deleteTemplate('captioning', activeId).subscribe(() => {
            this.currentTemplates.update(ts => ts.filter(t => t.id !== activeId));
            const remaining = this.currentTemplates();
            if (remaining.length > 0) {
                this.onTemplateChange(remaining[0].id);
            }
        });
    }

    private emitChanges() {
        const modelId = this.selectedCaptionModel();
        const variant = this.selectedQwen3Variant();
        const resolvedModelId = modelId === 'qwen3-vl' ? `qwen3-vl-${variant}` : modelId;

        const rawPrompt = this.captionSystemPrompt();
        const modelConfig = this.allModelConfigs.find(m => m.id === modelId);
        this.settingsChanged.emit({
            modelId: modelId,
            resolvedModelId: resolvedModelId,
            variant: modelId === 'qwen3-vl' ? variant : undefined,
            systemPrompt: rawPrompt,
            resolvedSystemPrompt: this.resolveWildcard(rawPrompt),
            wildcard: this.captionWildcard(),
            params: this.captionModelParams(),
            apiConfigured: modelId.startsWith('api-')
                ? (this.activeProviderStatus()?.configured ?? false)
                : undefined,
            supportsMultiImage: modelConfig?.supportsMultiImage ?? false,
            captionInstructions: this.captionInstructions(),
        });
    }
}
