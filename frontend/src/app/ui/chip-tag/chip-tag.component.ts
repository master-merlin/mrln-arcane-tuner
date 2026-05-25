import { ChangeDetectionStrategy, Component, input } from '@angular/core';

export type ChipTagTone = '' | 'success' | 'warning' | 'danger' | 'brand' | 'violet' | 'teal';

/** Uppercase mono tag — wraps the design's `.tag` class. */
@Component({
    selector: 'app-chip-tag',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `<span class="tag" [class]="tone()"><ng-content/></span>`,
})
export class ChipTagComponent {
    tone = input<ChipTagTone>('');
}
