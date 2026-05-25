import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
    selector: 'app-training-screen',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="page-head">
            <div>
                <div class="eyebrow">PLACEHOLDER</div>
                <h1 class="page-title">Training</h1>
                <p class="page-sub">Implemented in Phase 6.</p>
            </div>
        </div>
    `,
})
export class TrainingScreen {}
