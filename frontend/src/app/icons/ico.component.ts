import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';
import { LucideAngularModule, LucideIconData, icons } from 'lucide-angular';

@Component({
    selector: 'app-ico',
    standalone: true,
    imports: [LucideAngularModule],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `<lucide-angular [img]="iconData()" [size]="size()" [strokeWidth]="strokeWidth()" />`,
})
export class IcoComponent {
    name = input.required<keyof typeof icons>();
    size = input<number>(16);
    strokeWidth = input<number>(2);

    iconData = computed<LucideIconData>(() => icons[this.name()]);
}
