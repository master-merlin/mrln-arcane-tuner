// caption-suggestion-review.ts
import { ChangeDetectionStrategy, Component, effect, inject, input, output, signal } from '@angular/core';
import { DatasetService } from '../../../../services/dataset';

/**
 * Shows the pending LLM-refined caption suggestion (if any) for one image under the
 * active model definition, with Accept / Reject. Accept promotes the suggestion to the
 * live per-definition variant (server snapshots the prior variant); Reject discards it.
 */
@Component({
    selector: 'app-caption-suggestion-review',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @if (suggestion(); as s) {
            <div class="p-2 bg-warning/10 rounded-theme-md border border-warning/30 animate-fadeIn">
                <h5 class="text-[10px] text-warning font-bold mb-1 uppercase tracking-wide">{{ masked() ? 'Refined masked variant — review' : 'Refined variant — review' }}</h5>
                <p class="text-[10px] text-text-secondary font-mono mb-2 max-h-32 overflow-y-auto">{{ s }}</p>
                <div class="flex gap-2">
                    <button (click)="accept()" data-testid="suggestion-accept"
                            class="flex-1 bg-success hover:bg-success/90 text-white text-[10px] py-1 rounded-theme-md transition-colors">Accept</button>
                    <button (click)="reject()" data-testid="suggestion-reject"
                            class="flex-1 bg-surface-high hover:bg-surface-high/80 text-text-secondary text-[10px] py-1 rounded-theme-md transition-colors">Reject</button>
                </div>
            </div>
        }
    `,
})
export class CaptionSuggestionReviewComponent {
    datasetName = input.required<string>();
    stem = input.required<string>();
    definitionId = input<string | null>(null);
    masked = input<boolean>(false);
    accepted = output<void>();

    private api = inject(DatasetService);
    protected suggestion = signal<string | null>(null);

    constructor() {
        effect(() => {
            const name = this.datasetName();
            const stem = this.stem();
            const def = this.definitionId();
            this.suggestion.set(null);
            if (!def || !stem || !name) return;
            this.api.listCaptionSuggestions(name, def, this.masked()).subscribe(r => {
                const item = r.items.find(i => i.stem === stem);
                this.suggestion.set(item ? item.suggestion : null);
            });
        });
    }

    accept(): void {
        const def = this.definitionId();
        if (!def) return;
        this.api.acceptCaptionSuggestion(this.datasetName(), def, this.stem(), this.masked()).subscribe(() => {
            this.suggestion.set(null);
            this.accepted.emit();
        });
    }

    reject(): void {
        const def = this.definitionId();
        if (!def) return;
        this.api.rejectCaptionSuggestion(this.datasetName(), def, this.stem(), this.masked()).subscribe(() => this.suggestion.set(null));
    }
}
