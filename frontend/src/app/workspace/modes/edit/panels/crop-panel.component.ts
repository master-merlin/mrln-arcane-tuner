import { ChangeDetectionStrategy, Component, inject, input } from '@angular/core';
import { IcoComponent } from '../../../../icons/ico.component';
import { OverlayStore } from '../../../../state/overlay.store';

@Component({
    selector: 'app-crop-panel',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="info">
            <app-ico name="Info" [size]="13"/>
            <p>Crop is destructive (changes the source image's aspect-ratio bucket). It's not part of the non-destructive pipeline, so it has its own editor.</p>
        </div>
        <button type="button" class="btn primary" (click)="openCrop()" [disabled]="!datasetName() || !mediaFile()">
            <app-ico name="Crop" [size]="13"/> Open crop editor
        </button>
    `,
    styles: [`
        :host { display: flex; flex-direction: column; gap: 12px; }
        .info {
            display: flex; gap: 8px;
            padding: 10px 12px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
            color: var(--color-text-muted);
            font-size: 11.5px; line-height: 1.45;
        }
        .info p { margin: 0; }
    `],
})
export class CropPanelComponent {
    datasetName = input.required<string>();
    mediaFile = input.required<string>();
    width = input<number | undefined>(undefined);
    height = input<number | undefined>(undefined);

    private overlay = inject(OverlayStore);

    openCrop(): void {
        this.overlay.openModal('crop-preview', {
            datasetName: this.datasetName(),
            path: this.mediaFile(),
            width: this.width(),
            height: this.height(),
            target_width: this.width(),
            target_height: this.height(),
        });
    }
}
