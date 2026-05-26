import { ChangeDetectionStrategy, Component, input } from '@angular/core';

export interface StatePillsState {
    harmonized?: boolean;
    captioned?: boolean;
    masked?: boolean;
    /** Optional native tooltips per pill — used to surface coverage details. */
    titles?: {
        harmonized?: string;
        captioned?: string;
        masked?: string;
    };
}

/** H / C / M readiness trio — wraps `.state-pills` + `.state-pill`. */
@Component({
    selector: 'app-state-pills',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    host: { style: 'display: inline-flex;' },
    template: `
        <span class="state-pills">
            <span class="state-pill H" [class.on]="!!state().harmonized"
                  [attr.title]="state().titles?.harmonized ?? null">H</span>
            <span class="state-pill C" [class.on]="!!state().captioned"
                  [attr.title]="state().titles?.captioned ?? null">C</span>
            <span class="state-pill M" [class.on]="!!state().masked"
                  [attr.title]="state().titles?.masked ?? null">M</span>
        </span>
    `,
})
export class StatePillsComponent {
    state = input.required<StatePillsState>();
}
