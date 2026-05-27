import { ChangeDetectionStrategy, Component, effect, inject, input, signal } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { DatasetService, HistogramData } from '../../../../services/dataset';
import { HistogramDisplayComponent } from '../../../../components/dataset/dataset-viewer/components/histogram-display';
import { PipelineEditorState } from '../pipeline-editor.state';

@Component({
    selector: 'app-histogram-panel',
    standalone: true,
    imports: [HistogramDisplayComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `<app-histogram-display [data]="data()"/>`,
})
export class HistogramPanelComponent {
    datasetName = input.required<string>();
    mediaFile = input.required<string>();

    private datasets = inject(DatasetService);
    private state = inject(PipelineEditorState);

    protected data = signal<HistogramData | null>(null);

    constructor() {
        // Refetch on image change OR when a preview overlay arrives.
        let lastKey = '';
        effect(() => {
            const key = `${this.datasetName()}/${this.mediaFile()}|${this.state.previewOverlay()?.hash ?? ''}`;
            if (key === lastKey) return;
            lastKey = key;
            void this.fetch();
        });
    }

    private async fetch(): Promise<void> {
        const name = this.datasetName();
        const file = this.mediaFile();
        if (!name || !file) return;
        try {
            const resp = await firstValueFrom(this.datasets.getHistogram(name, file));
            this.data.set(resp as HistogramData);
        } catch {
            this.data.set(null);
        }
    }
}
