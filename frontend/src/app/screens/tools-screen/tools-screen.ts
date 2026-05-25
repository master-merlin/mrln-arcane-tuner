import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
    selector: 'app-tools-screen',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="page-head">
            <div>
                <div class="eyebrow">PLACEHOLDER</div>
                <h1 class="page-title">Tools</h1>
                <p class="page-sub">Implemented in Phase 8.</p>
            </div>
        </div>
    `,
})
export class ToolsScreen {}
