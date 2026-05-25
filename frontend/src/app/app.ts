import { ChangeDetectionStrategy, Component } from '@angular/core';
import { ShellComponent } from './shell/shell.component';

/**
 * Thin host component. The pre-overhaul AppComponent owned model
 * fetching, restart polling, training schema loading, etc. — those
 * responsibilities are moving to their respective screens in Phases
 * 6-8. For Phase 2 we just mount the shell.
 */
@Component({
    selector: 'app-root',
    standalone: true,
    imports: [ShellComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './app.html',
})
export class AppComponent {}
