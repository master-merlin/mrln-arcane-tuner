import { ChangeDetectionStrategy, Component, computed, inject, input } from '@angular/core';
import { HistogramDisplayComponent } from '../../../../components/dataset/dataset-viewer/components/histogram-display';
import { PreviewPipeline } from '../preview/preview-pipeline';

@Component({
    selector: 'app-histogram-panel',
    standalone: true,
    imports: [HistogramDisplayComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `<app-histogram-display [data]="data()"/>`,
})
export class HistogramPanelComponent {
    // Inputs preserved for template-call-site compatibility; not read.
    datasetName = input.required<string>();
    mediaFile = input.required<string>();

    private preview = inject(PreviewPipeline);
    protected data = computed(() => this.preview.histogram());
}
