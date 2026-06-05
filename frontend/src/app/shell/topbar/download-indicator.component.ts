import {
    ChangeDetectionStrategy, Component, ElementRef, HostListener,
    computed, inject,
} from '@angular/core';
import { ModelDownloadStore, DownloadProgress, RecentDownload } from '../../state/model-download.store';
import { TopbarPanelStore } from '../../state/topbar-panel.store';
import { IcoComponent } from '../../icons/ico.component';

/**
 * Global topbar pill + dropdown for model download progress.
 *
 * Visible only when there is at least one active OR recent download.
 * Pill click toggles a 320px panel showing active (with progress bars)
 * and recent (with ✓/✗ glyphs) entries.
 */
@Component({
    selector: 'app-download-indicator',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @if (visible()) {
            <div class="anchor">
                <button class="pill"
                        type="button"
                        [class.busy]="store.activeCount() > 0"
                        [class.open]="open()"
                        (click)="toggle()"
                        [title]="pillTitle()">
                    <app-ico name="Download" [size]="14"/>
                    @if (store.activeCount() > 0) { <span class="count mono">{{ store.activeCount() }}</span> }
                </button>
                @if (open()) {
                    <div class="panel">
                        @if (store.active().length > 0) {
                            <div class="section-label">Active</div>
                            @for (d of store.active(); track keyOf(d)) {
                                <div class="row active">
                                    <div class="row-head">
                                        <span class="name mono">{{ d.model_id }}</span>
                                        <span class="meta">
                                            @if (d.percent != null && d.total_bytes != null) {
                                                {{ mb(d.current_bytes) }} / {{ mb(d.total_bytes) }} MB · {{ d.percent }}%
                                            } @else {
                                                downloading…
                                            }
                                        </span>
                                    </div>
                                    <div class="bar">
                                        <div class="fill"
                                             [style.width.%]="d.percent ?? 0"
                                             [class.indeterminate]="d.percent == null"></div>
                                    </div>
                                    <div class="row-foot">{{ d.source }} · {{ d.category }}</div>
                                </div>
                            }
                        }
                        @if (store.recent().length > 0) {
                            <div class="section-label">Recent</div>
                            @for (r of store.recent(); track keyOf(r)) {
                                <div class="row recent">
                                    @if (r.status === 'complete') {
                                        <span class="glyph ok">✓</span>
                                    } @else {
                                        <span class="glyph err">✗</span>
                                    }
                                    <span class="name mono">{{ r.model_id }}</span>
                                    @if (humanSize(r); as sz) { <span class="size mono">{{ sz }}</span> }
                                    <span class="age">{{ ageOf(r) }}</span>
                                </div>
                                @if (r.error) { <div class="row-foot err">{{ r.error }}</div> }
                            }
                        }
                    </div>
                }
            </div>
        }
    `,
    styles: [`
        :host { display: inline-flex; }
        .anchor { position: relative; }
        /* Rectangle + rounded corners, centered icon + count — mirrors the
           task-center / notification buttons. Keeps the violet download
           identity via the .busy / .open accents. */
        .pill {
            display: inline-flex; align-items: center; gap: 5px;
            padding: 5px 8px;
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
            background: var(--color-surface-mid);
            color: var(--color-text-secondary);
            cursor: pointer;
        }
        .pill.busy { color: var(--color-violet); }
        .pill.open {
            color: var(--color-violet);
            border-color: color-mix(in oklab, var(--color-violet) 35%, transparent);
        }
        .count { font-size: 11px; font-weight: 700; }
        .panel {
            position: absolute; top: calc(100% + 6px); right: 0;
            width: 320px;
            background: var(--color-surface-low);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
            box-shadow: 0 8px 24px rgba(0,0,0,0.35);
            padding: 8px;
            display: flex; flex-direction: column; gap: 6px;
            z-index: 100;
        }
        .section-label {
            font-size: 9px; color: var(--color-text-muted);
            text-transform: uppercase; letter-spacing: 0.1em;
            padding: 4px 4px 0;
        }
        .row { padding: 6px 8px; border-radius: var(--radius-theme-sm); background: color-mix(in oklab, var(--color-violet) 8%, transparent); }
        .row.active { color: color-mix(in oklab, var(--color-violet) 90%, var(--color-text-primary)); }
        .row-head { display: flex; justify-content: space-between; gap: 6px; font-size: 11px; }
        .row .name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .row .meta { font-size: 10px; color: var(--color-text-muted); flex-shrink: 0; }
        .bar { height: 2px; background: rgba(0,0,0,0.3); margin-top: 3px; border-radius: 1px; overflow: hidden; }
        .bar .fill { height: 100%; background: var(--color-violet); transition: width 200ms; }
        .bar .fill.indeterminate {
            width: 30% !important;
            animation: slide 1.4s ease-in-out infinite;
        }
        @keyframes slide {
            0%   { transform: translateX(-100%); }
            100% { transform: translateX(330%); }
        }
        .row-foot { font-size: 9px; color: var(--color-text-muted); margin-top: 2px; }
        .row-foot.err { color: var(--color-danger); }
        .row.recent { display: flex; align-items: center; gap: 6px; background: transparent; padding: 4px 8px; font-size: 11px; }
        .row.recent .name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .row.recent .size { font-size: 9px; color: var(--color-text-secondary); flex-shrink: 0; font-variant-numeric: tabular-nums; }
        .row.recent .age { font-size: 9px; color: var(--color-text-muted); flex-shrink: 0; }
        .glyph.ok  { color: var(--color-success); }
        .glyph.err { color: var(--color-danger); }
        .mono { font-family: var(--font-mono); }
    `],
})
export class DownloadIndicatorComponent {
    protected store = inject(ModelDownloadStore);
    private host = inject(ElementRef<HTMLElement>);
    private panels = inject(TopbarPanelStore);
    protected open = this.panels.isOpen('downloads');

    protected visible = computed(() =>
        this.store.activeCount() > 0 || this.store.recent().length > 0,
    );

    protected pillTitle = computed(() =>
        this.open() ? 'Hide downloads' : 'Show downloads',
    );

    protected toggle(): void { this.panels.toggle('downloads'); }

    protected mb(bytes: number): string {
        return (bytes / (1024 * 1024)).toFixed(1);
    }

    /** Human-readable total size of a finished download (GB ≥ 1024 MB, else MB). */
    protected humanSize(r: RecentDownload): string {
        const bytes = r.total_bytes ?? r.current_bytes ?? 0;
        if (!bytes) return '';
        const mb = bytes / (1024 * 1024);
        return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
    }

    protected ageOf(r: RecentDownload): string {
        const sec = Math.max(0, Math.floor((Date.now() - r.finishedAt) / 1000));
        if (sec < 60) return `${sec}s ago`;
        const min = Math.floor(sec / 60);
        return `${min}m ago`;
    }

    protected keyOf(d: DownloadProgress): string {
        return `${d.source}::${d.model_id}`;
    }

    @HostListener('document:mousedown', ['$event'])
    protected onOutsidePointer(event: MouseEvent): void {
        if (!this.open()) return;
        if (!this.host.nativeElement.contains(event.target as Node)) {
            this.panels.close('downloads');
        }
    }

    @HostListener('document:keydown.escape')
    protected onEsc(): void {
        if (this.open()) this.panels.close('downloads');
    }
}
