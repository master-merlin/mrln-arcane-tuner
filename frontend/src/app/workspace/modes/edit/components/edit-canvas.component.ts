import { ChangeDetectionStrategy, Component, input } from '@angular/core';

@Component({
    selector: 'app-edit-canvas',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `<div style="flex:1; display:flex; align-items:center; justify-content:center;">Canvas — coming in Task 6.<br>{{ datasetName() }} / {{ mediaFile() }}</div>`,
})
export class EditCanvasComponent {
    datasetName = input.required<string>();
    mediaFile = input.required<string>();
    hasOverlay = input<boolean>(false);
}
