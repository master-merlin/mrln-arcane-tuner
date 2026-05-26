import { ChangeDetectionStrategy, Component, input } from '@angular/core';

@Component({
    selector: 'app-edit-right-panel',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `<div style="padding: 16px;">Right panel — coming in Task 7.</div>`,
})
export class EditRightPanelComponent {
    datasetName = input.required<string>();
    mediaFile = input.required<string>();
}
