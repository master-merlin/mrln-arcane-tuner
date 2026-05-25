import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';

export interface SegOption<T> {
    value: T;
    label: string;
}

/** Segmented switch — wraps the design's `.seg` block. */
@Component({
    selector: 'app-segmented',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="seg">
            @for (o of options(); track o.value) {
                <button type="button"
                        [class.active]="o.value === value()"
                        (click)="changed.emit(o.value)">{{ o.label }}</button>
            }
        </div>
    `,
})
export class SegmentedComponent<T> {
    options = input.required<ReadonlyArray<SegOption<T>>>();
    value = input.required<T>();
    changed = output<T>();
}
