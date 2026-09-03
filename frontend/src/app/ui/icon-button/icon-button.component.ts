import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { IcoComponent, type IconKey } from '../../icons/ico.component';

/** Square icon-only button — wraps `.icon-btn` with an `<app-ico/>` inside. */
@Component({
    selector: 'app-icon-button',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <button type="button"
                class="icon-btn"
                [class.brand]="brand()"
                [attr.title]="title()"
                [attr.aria-label]="title()">
            <app-ico [name]="icon()" [size]="size()"/>
        </button>
    `,
})
export class IconButtonComponent {
    icon = input.required<IconKey>();
    size = input<number>(15);
    brand = input<boolean>(false);
    title = input<string | undefined>(undefined);
}
