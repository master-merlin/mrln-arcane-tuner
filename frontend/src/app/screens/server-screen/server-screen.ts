import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
    selector: 'app-server-screen',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="page-head">
            <div>
                <div class="eyebrow">PLACEHOLDER</div>
                <h1 class="page-title">Server</h1>
                <p class="page-sub">Implemented in Phase 8.</p>
            </div>
        </div>
    `,
})
export class ServerScreen {}
