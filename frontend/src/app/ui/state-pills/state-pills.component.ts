import { ChangeDetectionStrategy, Component, input } from '@angular/core';

export interface StatePillsState {
    harmonized?: boolean;
    captioned?: boolean;
    masked?: boolean;
}

/** H / C / M readiness trio — wraps `.state-pills` + `.state-pill`. */
@Component({
    selector: 'app-state-pills',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <span class="state-pills">
            <span class="state-pill H" [class.on]="!!state().harmonized">H</span>
            <span class="state-pill C" [class.on]="!!state().captioned">C</span>
            <span class="state-pill M" [class.on]="!!state().masked">M</span>
        </span>
    `,
})
export class StatePillsComponent {
    state = input.required<StatePillsState>();
}
