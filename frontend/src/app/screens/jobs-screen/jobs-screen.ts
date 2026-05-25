import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
    selector: 'app-jobs-screen',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="page-head">
            <div>
                <div class="eyebrow">PLACEHOLDER</div>
                <h1 class="page-title">Jobs</h1>
                <p class="page-sub">Implemented in Phase 7.</p>
            </div>
        </div>
    `,
})
export class JobsScreen {}
