import { ChangeDetectionStrategy, Component, ElementRef, HostListener, inject } from '@angular/core';
import { IcoComponent } from '../../icons/ico.component';
import { TaskStore } from '../../state/task.store';
import { TopbarPanelStore } from '../../state/topbar-panel.store';

@Component({
    selector: 'app-task-center',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @if (count() > 0 || recent().length > 0) {
            <div class="tc">
                <button class="tc-pill" type="button" (click)="toggle()"
                        [class.busy]="count() > 0" title="Background tasks">
                    <app-ico [name]="count() > 0 ? 'Loader2' : 'Activity'" [size]="14"/>
                    @if (count() > 0) { <span class="tc-count mono">{{ count() }}</span> }
                </button>
                @if (open()) {
                    <div class="tc-pop">
                        <div class="tc-head">Activity</div>
                        @for (t of active(); track t.id) {
                            <div class="tc-row">
                                <div class="tc-title">{{ t.title }}
                                    @if (t.status === 'pending') { <span class="tc-q mono">Queued</span> }
                                </div>
                                <div class="bar"><i [style.width.%]="pct(t)"></i></div>
                                <div class="tc-meta mono">
                                    <span>{{ t.current }} / {{ t.total }}</span>
                                    <button class="tc-cancel" type="button" (click)="cancel(t.id)">Cancel</button>
                                </div>
                                @if (t.current_item) { <div class="tc-item mono">› {{ t.current_item }}</div> }
                            </div>
                        }
                        @if (recent().length > 0) { <div class="tc-sep"></div> }
                        @for (t of recent(); track t.id) {
                            <div class="tc-row done">
                                <div class="tc-title">
                                    <app-ico [name]="t.status === 'completed' ? 'Check' : 'X'" [size]="12"/>
                                    {{ t.title }}
                                </div>
                                <div class="tc-meta mono">{{ t.ok }} done · {{ t.failed }} failed</div>
                            </div>
                        }
                    </div>
                }
            </div>
        }
    `,
    styles: [`
        .tc { position: relative; }
        .tc-pill { display: inline-flex; align-items: center; gap: 5px; padding: 5px 8px;
            border: 1px solid var(--color-border-subtle); border-radius: var(--radius-theme-md);
            background: var(--color-surface-mid); color: var(--color-text-secondary); cursor: pointer; }
        .tc-pill.busy { color: var(--color-brand-light); }
        .tc-count { font-size: 11px; font-weight: 700; }
        .tc-pop { position: absolute; right: 0; top: calc(100% + 6px); width: 320px; z-index: 50;
            background: var(--color-surface-low); border: 1px solid var(--color-border-default);
            border-radius: var(--radius-theme-lg); box-shadow: var(--shadow-lg); padding: 10px; }
        .tc-head { font-size: 10px; font-weight: 700; letter-spacing: 0.10em; text-transform: uppercase;
            color: var(--color-text-subtle); margin-bottom: 8px; }
        .tc-row { padding: 6px 0; }
        .tc-row.done { color: var(--color-text-muted); display: flex; justify-content: space-between; }
        .tc-title { font-size: 12px; font-weight: 600; display: flex; align-items: center; gap: 6px; }
        .tc-q { color: var(--color-warning); font-size: 10px; }
        .bar { height: 6px; background: var(--color-surface-mid); border-radius: 3px; overflow: hidden; margin: 5px 0; }
        .bar i { display: block; height: 100%; background: var(--color-brand); }
        .tc-meta { display: flex; justify-content: space-between; font-size: 10.5px; color: var(--color-text-muted); }
        .tc-cancel { background: none; border: none; color: var(--color-danger); cursor: pointer; font: inherit; }
        .tc-item { font-size: 10px; color: var(--color-text-disabled); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .tc-sep { height: 1px; background: var(--color-border-subtle); margin: 8px 0; }
    `],
})
export class TaskCenterComponent {
    private store = inject(TaskStore);
    private host = inject(ElementRef<HTMLElement>);
    private panels = inject(TopbarPanelStore);
    protected active = this.store.active;
    protected recent = this.store.recent;
    protected count = this.store.activeCount;
    protected open = this.panels.isOpen('tasks');

    protected toggle(): void { this.panels.toggle('tasks'); }

    protected pct(t: { current: number; total: number }): number {
        return t.total > 0 ? Math.round((t.current / t.total) * 100) : 0;
    }
    protected cancel(id: string): void { this.store.cancel(id); }

    @HostListener('document:mousedown', ['$event'])
    protected onOutsidePointer(event: MouseEvent): void {
        if (!this.open()) return;
        if (!this.host.nativeElement.contains(event.target as Node)) {
            this.panels.close('tasks');
        }
    }

    @HostListener('document:keydown.escape')
    protected onEsc(): void {
        if (this.open()) this.panels.close('tasks');
    }
}
