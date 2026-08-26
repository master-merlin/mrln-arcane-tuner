import {
    ChangeDetectionStrategy,
    Component,
    OnDestroy,
    computed,
    effect,
    inject,
    signal,
} from '@angular/core';
import { WebSocketService } from '../../services/websocket.service';
import { SessionService } from '../../services/session.service';
import { IcoComponent } from '../../icons/ico.component';

/**
 * Global backend-down banner — the redesign's replacement for the legacy
 * "Failed to connect to backend… Is it running?" error + Retry strip
 * (legacy `app.ts:90-96`, audit gap #7). Rather than re-probing a one-shot
 * HTTP fetch, it reflects the always-on {@link WebSocketService} connection
 * (audit gap #16: those signals previously had no shell consumer).
 *
 * Debounced ~1.5s so the initial pre-connect window and quick server
 * restarts don't flash the banner. The Retry button forces an immediate
 * reconnect attempt rather than waiting for the 1s auto-retry.
 */
const SHOW_DELAY_MS = 1500;

@Component({
    selector: 'app-connection-banner',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @if (expired()) {
            <div class="conn-banner" role="alert" aria-live="assertive" data-testid="conn-banner-expired">
                <app-ico name="WifiOff" [size]="16" />
                <span class="conn-banner-text">Your session expired — sign in to continue.</span>
                <a class="conn-banner-retry" [href]="loginPath">Sign in</a>
            </div>
        } @else if (visible()) {
            <div class="conn-banner" role="alert" aria-live="assertive" data-testid="conn-banner">
                <app-ico name="WifiOff" [size]="16" />
                <span class="conn-banner-text">{{ message() }}</span>
                <button class="conn-banner-retry" type="button" (click)="retry()">Retry</button>
            </div>
        }
    `,
    styles: [`
        .conn-banner {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 9px 16px;
            background: oklch(0.70 0.17 25 / 0.12);
            border-bottom: 1px solid oklch(0.70 0.17 25 / 0.30);
            color: var(--color-danger);
            font-size: 12.5px;
            font-weight: 600;
        }
        .conn-banner-text { flex: 1; }
        .conn-banner-retry {
            border: 1px solid oklch(0.40 0.12 25);
            color: var(--color-danger);
            background: transparent;
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 11.5px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s ease;
        }
        .conn-banner-retry:hover { background: oklch(0.20 0.08 25); }
    `],
})
export class ConnectionBannerComponent implements OnDestroy {
    private ws = inject(WebSocketService);
    private session = inject(SessionService);
    protected visible = signal(false);
    protected expired = this.session.expired;
    protected readonly loginPath = SessionService.LOGIN_PATH;

    /**
     * Once the ladder is in its degraded band the retry interval is measured
     * in tens of seconds, so "reconnecting…" stops being true in any useful
     * sense. Say what is actually happening and let the Retry button be the
     * user's lever.
     */
    protected message = computed(() =>
        this.ws.degraded()
            ? 'Cannot reach the backend. Still trying — check that the server is running.'
            : 'Lost connection to the backend — reconnecting…'
    );

    private showTimer?: ReturnType<typeof setTimeout>;

    constructor() {
        effect(() => {
            const connected = this.ws.isConnected();
            if (this.showTimer) {
                clearTimeout(this.showTimer);
                this.showTimer = undefined;
            }
            if (connected) {
                this.visible.set(false);
            } else {
                // Only surface after a grace window so transient drops
                // (initial connect, fast restarts) don't flicker the banner.
                this.showTimer = setTimeout(() => this.visible.set(true), SHOW_DELAY_MS);
            }
        });
    }

    protected retry(): void {
        this.ws.forceReconnect();
    }

    ngOnDestroy(): void {
        if (this.showTimer) clearTimeout(this.showTimer);
    }
}
