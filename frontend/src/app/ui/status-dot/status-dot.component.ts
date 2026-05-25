import { ChangeDetectionStrategy, Component, input } from '@angular/core';

export type StatusTone = 'success' | 'warning' | 'danger' | 'brand';

/** Colored status dot — wraps the design's `.sdot` class. */
@Component({
    selector: 'app-status-dot',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `<span class="sdot" [class]="tone()"></span>`,
})
export class StatusDotComponent {
    tone = input<StatusTone>('success');
}
