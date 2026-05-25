import { ChangeDetectionStrategy, Component, input } from '@angular/core';

/**
 * Styled `.popover` shell. Open/close is the caller's responsibility —
 * this component only renders the surface (and slots content) when
 * `open` is true.
 */
@Component({
    selector: 'app-popover',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @if (open()) {
            <div class="popover" [style.min-width.px]="minWidth()">
                <ng-content/>
            </div>
        }
    `,
})
export class PopoverComponent {
    open = input.required<boolean>();
    minWidth = input<number>(240);
}
