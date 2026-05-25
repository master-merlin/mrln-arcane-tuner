import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';
import { LucideDynamicIcon, type LucideIcon, type LucideIconData, icons } from '@lucide/angular';

/**
 * Thin wrapper around `@lucide/angular`'s `LucideDynamicIcon`. Lets us
 * swap the icon provider in a single file later.
 *
 * Usage: `<app-ico name="Database" [size]="16"/>`
 *
 * The `name` is the PascalCase icon stem (e.g. `"Database"` /
 * `"ChevronDown"`); the wrapper prepends `Lucide` to look the component
 * up in the `icons` barrel from `@lucide/angular`, then extracts the
 * static `icon` data so `LucideDynamicIcon` can render it without
 * needing the icon to be registered via `provideLucideIcons`.
 */

// All exported names from `icons` start with `Lucide`. Strip that prefix
// for the public API of this component so the call sites stay short.
type IconKey = keyof typeof icons extends `Lucide${infer R}` ? R : never;

@Component({
    selector: 'app-ico',
    standalone: true,
    imports: [LucideDynamicIcon],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `<svg
        [lucideIcon]="iconData()"
        [size]="size()"
        [strokeWidth]="strokeWidth()"
    ></svg>`,
})
export class IcoComponent {
    name = input.required<IconKey>();
    size = input<number>(16);
    strokeWidth = input<number>(2);

    iconData = computed<LucideIconData>(() => {
        const key = `Lucide${this.name()}` as keyof typeof icons;
        const cmp = icons[key] as unknown as LucideIcon;
        return cmp.icon;
    });
}
