// frontend/src/app/shell/model-selector/model-selector.component.ts
import {
    ChangeDetectionStrategy,
    Component,
    computed,
    effect,
    inject,
    signal,
    untracked,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CaptionContextService } from '../../services/caption-context.service';
import { ModelContextStore, type DefinitionRef } from '../../state/model-context.store';

/**
 * Top-bar control: a "Model-aware" checkbox that, when enabled, reveals
 * cascading Family → definition dropdowns. Selecting a definition sets the
 * workspace-wide active model context (drives Phase 1 token counting,
 * Phase 2 analytics, Phase 3 caption variants).
 */
@Component({
    selector: 'app-model-selector',
    standalone: true,
    imports: [FormsModule],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './model-selector.component.html',
})
export class ModelSelectorComponent {
    protected ctx = inject(ModelContextStore);
    private api = inject(CaptionContextService);

    private definitions = signal<DefinitionRef[]>([]);
    private loaded = signal(false);

    protected selectedFamily = signal<string>('');

    protected families = computed<string[]>(() => {
        const seen: string[] = [];
        for (const d of this.definitions()) {
            if (!seen.includes(d.family)) seen.push(d.family);
        }
        return seen;
    });

    protected familyDefinitions = computed<DefinitionRef[]>(() =>
        this.definitions().filter(d => d.family === this.selectedFamily()),
    );

    constructor() {
        // Lazy-load the definition list the first time model-aware turns on.
        effect(() => {
            if (this.ctx.modelAware() && !this.loaded()) {
                this.loaded.set(true);
                this.api.listDefinitions().subscribe(defs => this.definitions.set(defs));
            }
        });

        // Reflect the retained/persisted definition's family in the dropdown
        // when model-aware is (re)enabled — so flipping the toggle on/off (or a
        // page reload) keeps the family + definition visible without re-picking.
        // Reads `selectedFamily` untracked so this only reacts to the active
        // definition changing, never fighting a manual family pick.
        effect(() => {
            const fam = this.ctx.activeDefinition()?.family;
            if (fam && fam !== untracked(this.selectedFamily)) {
                this.selectedFamily.set(fam);
            }
        });
    }

    protected toggleModelAware(on: boolean): void {
        this.ctx.setModelAware(on);
    }

    protected onFamilyChange(family: string): void {
        this.selectedFamily.set(family);
        this.ctx.setDefinition(null); // force a fresh definition pick
    }

    protected onDefinitionChange(id: string): void {
        const def = this.definitions().find(d => d.id === id) ?? null;
        this.ctx.setDefinition(def);
    }
}
