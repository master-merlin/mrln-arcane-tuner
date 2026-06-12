import {
    ChangeDetectionStrategy, Component, ElementRef, HostListener, inject,
} from '@angular/core';
import { IcoComponent } from '../../icons/ico.component';
import { SystemUpdateService } from '../../services/system-update.service';
import { TopbarPanelStore } from '../../state/topbar-panel.store';

@Component({
    selector: 'app-update-indicator',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @if (update.available()) {
            <div class="ui-wrap">
                <button class="icon-btn" type="button" (click)="toggle()"
                        [class.has-update]="update.updateReady()"
                        title="App updates" aria-label="App updates">
                    <app-ico name="RefreshCw" [size]="15" />
                    @if (update.updateReady()) { <span class="ui-dot"></span> }
                </button>
                @if (open()) {
                    <div class="ui-pop">
                        <div class="ui-head">App updates</div>
                        @if (update.updateReady()) {
                            <p class="ui-line">{{ update.status()?.behind }} new commit(s) on origin.</p>
                            <button class="btn btn-primary" type="button"
                                    [disabled]="update.isBusy()" (click)="onApply()">
                                <app-ico name="Download" [size]="14" /> Update &amp; restart
                            </button>
                        } @else {
                            <p class="ui-line">Up to date.</p>
                            <button class="btn" type="button"
                                    [disabled]="update.isBusy()" (click)="onCheck()">
                                <app-ico name="RefreshCw" [size]="14" /> Check now
                            </button>
                        }
                    </div>
                }
            </div>
        }
    `,
    styles: [`
        .ui-wrap { position: relative; }
        .ui-dot {
            position: absolute; top: 4px; right: 4px; width: 8px; height: 8px;
            border-radius: 50%; background: var(--brand, #6e8bff);
        }
        .icon-btn.has-update { color: var(--brand, #6e8bff); }
        .ui-pop {
            position: absolute; right: 0; top: calc(100% + 8px); z-index: 1000;
            min-width: 240px; padding: 12px; border-radius: 10px;
            background: var(--surface-1, #161a23); border: 1px solid var(--border, #2a2f3a);
            box-shadow: 0 8px 28px rgba(0,0,0,.4); display: flex; flex-direction: column; gap: 10px;
        }
        .ui-head { font-weight: 600; font-size: 13px; }
        .ui-line { margin: 0; font-size: 12px; opacity: .8; }
    `],
})
export class UpdateIndicatorComponent {
    protected update = inject(SystemUpdateService);
    private panels = inject(TopbarPanelStore);
    private host = inject(ElementRef<HTMLElement>);

    protected open = this.panels.isOpen('updates');
    protected toggle(): void { this.panels.toggle('updates'); }

    onApply(): void {
        this.update.apply().subscribe();
        this.panels.close('updates');
    }

    onCheck(): void {
        this.update.check().subscribe();
    }

    @HostListener('document:mousedown', ['$event'])
    protected onOutside(e: MouseEvent): void {
        if (this.open() && !this.host.nativeElement.contains(e.target as Node)) {
            this.panels.close('updates');
        }
    }

    @HostListener('document:keydown.escape')
    protected onEsc(): void { if (this.open()) this.panels.close('updates'); }
}
