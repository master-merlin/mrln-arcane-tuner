import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
    selector: 'app-datasets-screen',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="page-head">
            <div>
                <div class="eyebrow">PLACEHOLDER</div>
                <h1 class="page-title">Datasets</h1>
                <p class="page-sub">Implemented in Phase 3.</p>
            </div>
        </div>
    `,
})
export class DatasetsScreen {}
