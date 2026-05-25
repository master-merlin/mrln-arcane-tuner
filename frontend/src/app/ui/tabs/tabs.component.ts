import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';

export interface TabItem<T> {
    value: T;
    label: string;
    disabled?: boolean;
}

/** Horizontal tab strip — wraps the design's `.tab` class. */
@Component({
    selector: 'app-tabs',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="tabs">
            @for (t of tabs(); track t.value) {
                <button type="button"
                        class="tab"
                        [class.active]="t.value === active()"
                        [disabled]="!!t.disabled"
                        (click)="onClick(t)">{{ t.label }}</button>
            }
        </div>
    `,
})
export class TabsComponent<T> {
    tabs = input.required<ReadonlyArray<TabItem<T>>>();
    active = input.required<T>();
    changed = output<T>();

    protected onClick(t: TabItem<T>) {
        if (!t.disabled) this.changed.emit(t.value);
    }
}
