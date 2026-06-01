import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';
import { TrainingSegment } from '../training-dynamic-config/training-dynamic-config';

/**
 * Presentational table-of-contents for the training screen.
 *
 * Renders the dynamic-config segment list as clickable two-line items with a
 * status dot, highlights the active section, and emits `jump` on click. All
 * scroll-spy / active-tracking logic lives in the parent shell; this component
 * is pure input → output.
 */
@Component({
    selector: 'app-training-toc',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="ttoc-head">
            <div class="eyebrow">CONFIGURATION</div>
            <div class="page-title">Sections</div>
        </div>
        @for (s of segments(); track s.id) {
            <button
                type="button"
                class="ttoc-item"
                [class.active]="s.id === activeId()"
                (click)="jump.emit(s.id)">
                <span
                    class="ttoc-dot"
                    [class.success]="s.status === 'success'"
                    [class.warning]="s.status === 'warning'"></span>
                <span class="ttoc-text">
                    <span class="ttoc-label">{{ s.label }}</span>
                    @if (s.sub) {
                        <span class="ttoc-sub">{{ s.sub }}</span>
                    }
                </span>
            </button>
        }
    `,
    styleUrl: './training-toc.css',
})
export class TrainingToc {
    segments = input<TrainingSegment[]>([]);
    activeId = input<string | null>(null);
    jump = output<string>();
}
