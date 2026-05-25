import { ChangeDetectionStrategy, Component, input } from '@angular/core';

export type ChipTone = 'default' | 'brand' | 'success' | 'warning' | 'danger' | 'violet' | 'teal' | 'solid';

/** Small pill — wraps the design's `.chip` class. */
@Component({
    selector: 'app-chip',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <span class="chip" [class]="toneClass()">
            @if (dot()) { <span class="dot"></span> }
            <ng-content/>
        </span>
    `,
})
export class ChipComponent {
    tone = input<ChipTone>('default');
    dot = input<boolean>(false);

    protected toneClass(): string {
        const t = this.tone();
        return t === 'default' ? '' : t;
    }
}
