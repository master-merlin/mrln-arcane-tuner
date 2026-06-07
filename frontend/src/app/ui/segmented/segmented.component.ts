import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';
import { icons } from '@lucide/angular';
import { IcoComponent } from '../../icons/ico.component';

type IconKey = keyof typeof icons extends `Lucide${infer R}` ? R : never;

export interface SegOption<T> {
    value: T;
    label: string;
    /** Optional leading icon — Lucide name (e.g. `'Grid'`, `'Image'`). */
    icon?: IconKey;
}

/** Segmented switch — wraps the design's `.seg` block. */
@Component({
    selector: 'app-segmented',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="seg">
            @for (o of options(); track o.value) {
                <button type="button"
                        [class.active]="o.value === value()"
                        [attr.data-testid]="testidPrefix() ? testidPrefix() + '-' + o.value : null"
                        (click)="changed.emit(o.value)">
                    @if (o.icon) {
                        <app-ico [name]="o.icon" [size]="12"/>
                    }
                    <span>{{ o.label }}</span>
                </button>
            }
        </div>
    `,
    styles: [`
        .seg button { display: inline-flex; align-items: center; gap: 4px; }
    `],
})
export class SegmentedComponent<T> {
    options = input.required<ReadonlyArray<SegOption<T>>>();
    value = input.required<T>();
    /** Optional kebab prefix; when set, each button gets
     *  `data-testid="{prefix}-{value}"` for stable e2e selection. */
    testidPrefix = input<string | null>(null);
    changed = output<T>();
}
