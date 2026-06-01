import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { SystemControlService } from '../../services/system-control.service';

/**
 * App-global graceful-restart overlay (audit gap #8). Previously lived inside
 * `server-screen`, so it only locked the UI while on `/server`. Mounted in the
 * shell and driven by {@link SystemControlService.isRestarting} it now locks
 * the whole app whenever a restart is in flight, regardless of route.
 *
 * Full-screen blurred backdrop + title + description + three bouncing dots
 * (no spinning circle — per the legacy-parity verdict 2026-05-28).
 */
@Component({
    selector: 'app-restart-overlay',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @if (system.isRestarting()) {
            <div class="restart-overlay" role="alertdialog" aria-live="polite" aria-busy="true">
                <h2 class="restart-overlay-title">Server Restarting</h2>
                <p class="restart-overlay-desc">
                    The backend is performing a graceful restart. The connection will be
                    restored automatically in a few seconds.
                </p>
                <div class="restart-overlay-dots" aria-hidden="true">
                    <span class="restart-overlay-dot"></span>
                    <span class="restart-overlay-dot"></span>
                    <span class="restart-overlay-dot"></span>
                </div>
            </div>
        }
    `,
    styles: [`
        .restart-overlay {
            position: fixed;
            inset: 0;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 32px;
            text-align: center;
            background: color-mix(in oklab, var(--color-base) 70%, transparent);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            animation: restart-overlay-fade-in 0.5s ease;
        }
        .restart-overlay-title {
            font-size: 24px;
            font-weight: 700;
            color: var(--color-text-primary);
            margin: 0 0 8px;
            letter-spacing: 0.01em;
        }
        .restart-overlay-desc {
            color: var(--color-text-muted);
            max-width: 28rem;
            margin: 0 0 32px;
            line-height: 1.5;
            font-size: 14px;
        }
        .restart-overlay-dots {
            display: flex;
            gap: 8px;
        }
        .restart-overlay-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--color-brand);
            animation: restart-overlay-bounce 1s infinite ease-in-out;
        }
        /* Staggered delays so the three dots travel a phase apart. */
        .restart-overlay-dot:nth-child(1) { animation-delay: -0.3s; }
        .restart-overlay-dot:nth-child(2) { animation-delay: -0.15s; }
        .restart-overlay-dot:nth-child(3) { animation-delay: 0s; }

        @keyframes restart-overlay-fade-in {
            from { opacity: 0; }
            to   { opacity: 1; }
        }
        @keyframes restart-overlay-bounce {
            0%, 100% { transform: translateY(0);    opacity: 0.55; }
            50%      { transform: translateY(-8px); opacity: 1;    }
        }

        /* Reduced-motion: hold the dots statically rather than bouncing. */
        @media (prefers-reduced-motion: reduce) {
            .restart-overlay     { animation: none; }
            .restart-overlay-dot { animation: none; opacity: 1; }
        }
    `],
})
export class RestartOverlayComponent {
    protected system = inject(SystemControlService);
}
