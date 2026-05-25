import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { icons } from '@lucide/angular';
import { IcoComponent } from '../../icons/ico.component';

type IconKey = keyof typeof icons extends `Lucide${infer R}` ? R : never;

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
