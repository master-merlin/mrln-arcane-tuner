import { ChangeDetectionStrategy, Component, computed, inject, input, signal } from '@angular/core';
import { IcoComponent } from '../../../../icons/ico.component';
import { CropPanelComponent } from '../panels/crop-panel.component';
import { PipelineEditorState } from '../pipeline-editor.state';
import { TAB_DEFS, TabDef, TabKind } from '../operation-defs';

@Component({
    selector: 'app-edit-left-panel',
    standalone: true,
    imports: [IcoComponent, CropPanelComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="tabs">
            @for (group of groups; track group.label) {
                <div class="tab-group">
                    <div class="tab-group-head" [class.ai]="group.kind === 'ai'">
                        @if (group.kind === 'ai') {
                            <app-ico name="Sparkles" [size]="9"/>
                        }
                        <span>{{ group.label }}</span>
                        <span class="rule"></span>
                        <span class="hint">{{ group.hint }}</span>
                    </div>
                    <div class="tab-row">
                        @for (t of group.tabs; track t.kind) {
                            <button type="button" class="tab"
                                    [class.active]="active() === t.kind"
                                    [class.ai]="group.kind === 'ai'"
                                    (click)="active.set(t.kind)">
                                <app-ico [name]="t.icon" [size]="11"/>
                                <span>{{ t.label }}</span>
                            </button>
                        }
                    </div>
                </div>
            }
        </div>

        <div class="panel-host">
            <div class="panel-head">
                <span class="title">{{ activeLabel() }}</span>
                @if (isAi()) {
                    <span class="ai-badge"><app-ico name="Sparkles" [size]="8"/> AI</span>
                }
                <button type="button" class="reset" (click)="resetActive()" title="Reset to default">
                    <app-ico name="RefreshCw" [size]="11"/> Reset
                </button>
            </div>

            <!-- Panel switch — cases added in Phase 6 (Tasks 14-24). -->
            @switch (active()) {
                @case ('crop') {
                    <app-crop-panel
                        [datasetName]="datasetName()"
                        [mediaFile]="mediaFile()"
                        [width]="width()"
                        [height]="height()"/>
                }
                @default {
                    <div class="panel-todo">
                        Panel "<b>{{ activeLabel() }}</b>" implementation pending.
                    </div>
                }
            }
        </div>
    `,
    styles: [`
        :host { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
        .tabs {
            padding: 8px 10px;
            border-bottom: 1px solid var(--color-border-subtle);
            background: var(--color-base);
            display: flex; flex-direction: column; gap: 6px;
            flex-shrink: 0;
        }
        .tab-group-head {
            display: flex; align-items: center; gap: 6px;
            padding: 2px 4px 4px;
            font-size: 9px; font-weight: 700;
            letter-spacing: 0.10em; text-transform: uppercase;
            color: var(--color-text-subtle);
        }
        .tab-group-head.ai { color: var(--color-violet); }
        .tab-group-head .rule { flex: 1; height: 1px; background: var(--color-border-subtle); margin-left: 4px; }
        .tab-group-head .hint {
            font-size: 8.5px; font-weight: 500; letter-spacing: 0.04em;
            text-transform: none; color: var(--color-text-subtle);
        }
        .tab-row { display: flex; flex-wrap: wrap; gap: 2px; }
        .tab {
            display: inline-flex; align-items: center; gap: 5px;
            padding: 5px 8px;
            font-size: 11px; font-weight: 600;
            border-radius: var(--radius-theme-sm);
            background: transparent;
            color: var(--color-text-muted);
            border: 1px solid transparent;
            cursor: pointer;
        }
        .tab:hover { color: var(--color-text-primary); }
        .tab.active {
            background: color-mix(in oklab, var(--color-brand) 18%, transparent);
            color: var(--color-brand);
            border-color: color-mix(in oklab, var(--color-brand) 40%, transparent);
        }
        .tab.ai.active {
            background: color-mix(in oklab, var(--color-violet) 18%, transparent);
            color: var(--color-violet);
            border-color: color-mix(in oklab, var(--color-violet) 40%, transparent);
        }
        .panel-host { padding: 14px 16px; display: flex; flex-direction: column; gap: 12px; flex: 1; overflow-y: auto; }
        .panel-head { display: flex; align-items: center; gap: 8px; }
        .panel-head .title {
            font-size: 11px; font-weight: 700;
            letter-spacing: 0.10em; text-transform: uppercase;
            color: var(--color-text-primary);
        }
        .ai-badge {
            display: inline-flex; align-items: center; gap: 3px;
            font-size: 9px; font-weight: 600;
            padding: 1px 6px;
            border-radius: var(--radius-theme-sm);
            background: color-mix(in oklab, var(--color-violet) 18%, transparent);
            color: var(--color-violet);
            border: 1px solid color-mix(in oklab, var(--color-violet) 35%, transparent);
            text-transform: uppercase; letter-spacing: 0.05em;
        }
        .reset {
            margin-left: auto;
            display: inline-flex; align-items: center; gap: 4px;
            padding: 3px 8px;
            font-size: 10.5px;
            background: transparent; border: 1px solid transparent;
            border-radius: var(--radius-theme-sm);
            color: var(--color-text-muted);
            cursor: pointer;
        }
        .reset:hover { color: var(--color-text-primary); background: var(--color-surface-mid); }
        .panel-todo {
            padding: 16px;
            border: 1px dashed var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
            text-align: center;
            color: var(--color-text-muted);
            font-size: 12px;
        }
    `],
})
export class EditLeftPanelComponent {
    datasetName = input.required<string>();
    mediaFile = input.required<string>();
    width = input<number | undefined>(undefined);
    height = input<number | undefined>(undefined);

    protected state = inject(PipelineEditorState);

    protected active = signal<TabKind>('crop');

    protected groups = (() => {
        const adjust: TabDef[] = TAB_DEFS.filter(t => t.group === 'adjust');
        const ai: TabDef[] = TAB_DEFS.filter(t => t.group === 'ai');
        return [
            { kind: 'adjust' as const, label: 'Adjust', hint: 'real-time · reversible', tabs: adjust },
            { kind: 'ai' as const,     label: 'AI Models', hint: 'async · may change output', tabs: ai },
        ];
    })();

    protected activeLabel = computed(() => TAB_DEFS.find(t => t.kind === this.active())?.label ?? '');
    protected isAi = computed(() => TAB_DEFS.find(t => t.kind === this.active())?.group === 'ai');

    protected resetActive(): void {
        const k = this.active();
        if (k === 'crop') return;  // Crop has no state to reset
        this.state.resetPanel(k);
    }
}
