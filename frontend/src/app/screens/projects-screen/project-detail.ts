import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
    selector: 'app-project-detail',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="page-head">
            <div>
                <div class="eyebrow">PLACEHOLDER</div>
                <h1 class="page-title">Project Detail</h1>
                <p class="page-sub">Implemented in Phase 5.</p>
            </div>
        </div>
    `,
})
export class ProjectDetail {}
