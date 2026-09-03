import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';
import { LucideDynamicIcon, type LucideIcon, type LucideIconData } from '@lucide/angular';

import { type IconKey, iconSet } from './icon-set';

/**
 * Thin wrapper around `@lucide/angular`'s `LucideDynamicIcon`. Lets us
 * swap the icon provider in a single file later.
 *
 * Usage: `<app-ico name="Database" [size]="16"/>`
 *
 * The `name` is the PascalCase icon stem (e.g. `"Database"` /
 * `"ChevronDown"`); the wrapper looks it up in `iconSet` and extracts the
 * static `icon` data so `LucideDynamicIcon` can render it without needing
 * the icon to be registered via `provideLucideIcons`.
 *
 * The lookup used to run against `icons`, the barrel of every icon Lucide
 * ships, with a key built at runtime — which no bundler can narrow, so all of
 * them shipped. `icon-set.ts` says why that stopped being acceptable. The
 * shape here is unchanged; only the map it reads is now finite.
 */

export { type IconKey } from './icon-set';

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
        const cmp = iconSet[this.name()] as unknown as LucideIcon;
        return cmp.icon;
    });
}
