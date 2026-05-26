import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
    selector: 'app-edit-left-panel',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `<div style="padding: 16px;">Left panel — coming in Task 5.</div>`,
})
export class EditLeftPanelComponent {}
