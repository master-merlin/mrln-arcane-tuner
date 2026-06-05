import {
    ChangeDetectionStrategy, Component, ElementRef, HostListener,
    inject,
} from '@angular/core';
import { IcoComponent } from '../../icons/ico.component';
import { ToastService, ToastHistoryEntry } from '../../services/toast';
import { TopbarPanelStore } from '../../state/topbar-panel.store';

/**
 * Topbar bell button + dropdown listing the most recent toast messages
 * (newest first) with their wall-clock time. Reads ToastService.history().
 * Click-toggle; closes on outside pointer or Escape — mirrors the
 * download-indicator dropdown pattern.
 */
@Component({
    selector: 'app-notification-panel',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="anchor">
            <button class="icon-btn"
                    type="button"
                    [class.brand]="open()"
                    (click)="toggle()"
                    [title]="open() ? 'Hide notifications' : 'Show notifications'">
                <app-ico name="Bell" [size]="15" />
            </button>
            @if (open()) {
                <div class="panel">
                    <div class="section-label">Notifications</div>
                    @if (toast.history().length === 0) {
                        <div class="empty">No recent notifications</div>
                    } @else {
                        @for (n of toast.history(); track n.id) {
                            <div class="row">
                                <span class="glyph" [class]="n.type">{{ glyph(n) }}</span>
                                <span class="msg">{{ n.message }}</span>
                                <span class="time">{{ time(n) }}</span>
                            </div>
                        }
                    }
                </div>
            }
        </div>
    `,
    styles: [`
        :host { display: inline-flex; }
        .anchor { position: relative; }
        /* Persistent open-state affordance: the global .icon-btn.brand only
           styles :hover, so without this the bell looks identical open vs. closed. */
        .icon-btn.brand { color: var(--color-brand); background: var(--color-surface-mid); }
        .panel {
            position: absolute; top: calc(100% + 6px); right: 0;
            width: 320px;
            background: var(--color-surface-low);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
            box-shadow: 0 8px 24px rgba(0,0,0,0.35);
            padding: 8px;
            display: flex; flex-direction: column; gap: 2px;
            z-index: 100;
        }
        .section-label {
            font-size: 9px; color: var(--color-text-muted);
            text-transform: uppercase; letter-spacing: 0.1em;
            padding: 4px 4px 6px;
        }
        .empty {
            font-size: 11px; color: var(--color-text-muted);
            padding: 8px; text-align: center;
        }
        .row {
            display: flex; align-items: center; gap: 8px;
            padding: 6px 8px; border-radius: var(--radius-theme-sm);
            font-size: 11px;
        }
        .row:hover { background: var(--color-surface-mid); }
        .glyph { flex-shrink: 0; font-weight: 700; width: 12px; text-align: center; }
        .glyph.success { color: var(--color-success); }
        .glyph.error   { color: var(--color-danger); }
        .glyph.warning { color: var(--color-warning); }
        .glyph.info    { color: var(--color-brand); }
        .msg { flex: 1; min-width: 0; color: var(--color-text-primary); word-break: break-word; }
        .time { flex-shrink: 0; font-size: 9px; color: var(--color-text-muted); font-variant-numeric: tabular-nums; }
    `],
})
export class NotificationPanelComponent {
    protected toast = inject(ToastService);
    private host = inject(ElementRef<HTMLElement>);
    private panels = inject(TopbarPanelStore);
    protected open = this.panels.isOpen('notifications');

    protected toggle(): void { this.panels.toggle('notifications'); }

    protected glyph(n: ToastHistoryEntry): string {
        switch (n.type) {
            case 'success': return '✓';
            case 'error': return '✗';
            case 'warning': return '⚠';
            case 'info': return 'ℹ';
        }
    }

    protected time(n: ToastHistoryEntry): string {
        const d = new Date(n.timestamp);
        const hh = d.getHours().toString().padStart(2, '0');
        const mm = d.getMinutes().toString().padStart(2, '0');
        return `${hh}:${mm}`;
    }

    @HostListener('document:mousedown', ['$event'])
    protected onOutsidePointer(event: MouseEvent): void {
        if (!this.open()) return;
        if (!this.host.nativeElement.contains(event.target as Node)) {
            this.panels.close('notifications');
        }
    }

    @HostListener('document:keydown.escape')
    protected onEsc(): void {
        if (this.open()) this.panels.close('notifications');
    }
}
