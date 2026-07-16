import {
    ChangeDetectionStrategy, Component, ElementRef, OnDestroy,
    ViewChild, effect, input,
} from '@angular/core';
import uPlot from 'uplot';

/**
 * Minimal uPlot host: rebuilds the chart when `data`/`opts` change, sizes to
 * the container width. Skips instantiation when no 2D canvas is available
 * (jsdom/Vitest) so component specs can render the parent without crashing.
 */
@Component({
    selector: 'app-stats-uplot',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `<div #host class="stats-uplot"></div>`,
    styles: [`:host { display: block; width: 100%; } .stats-uplot { width: 100%; }`],
})
export class StatsUplotComponent implements OnDestroy {
    data = input.required<uPlot.AlignedData>();
    opts = input.required<Omit<uPlot.Options, 'width' | 'height'>>();
    height = input<number>(160);

    @ViewChild('host', { static: true }) private host!: ElementRef<HTMLDivElement>;
    private plot: uPlot | null = null;

    constructor() {
        effect(() => {
            const data = this.data();
            const opts = this.opts();
            this.plot?.destroy();
            this.plot = null;
            if (!document.createElement('canvas').getContext?.('2d')) return; // jsdom guard
            const el = this.host.nativeElement;
            this.plot = new uPlot(
                { ...opts, width: el.clientWidth || 560, height: this.height() },
                data,
                el,
            );
        });
    }

    ngOnDestroy(): void { this.plot?.destroy(); }
}
