import { ChangeDetectionStrategy, Component, input } from '@angular/core';

/** Horizontal progress bar — wraps the design's `.bar` class. */
@Component({
    selector: 'app-progress-bar',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="bar" [class.lg]="lg()">
            <i [style.width.%]="pct()" [style.background]="color()"></i>
        </div>
    `,
})
export class ProgressBarComponent {
    pct = input.required<number>();
    lg = input<boolean>(false);
    /** Optional CSS color or `var(--color-…)` override. */
    color = input<string | undefined>(undefined);
}
