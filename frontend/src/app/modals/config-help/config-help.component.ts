import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';

/**
 * Config-help modal — renders the per-field help content that used to live as
 * an inline `@if (activeHelpKey())` blob inside training-dynamic-config.
 *
 * The opener (training-dynamic-config) resolves the help data from
 * `public/config_help.json` (+ the schema title) and passes the already-built
 * payload through `overlay.openModal('config-help', payload)`. This modal is
 * purely presentational: it displays the title / tip / rendered detail HTML and
 * closes. Keeping the data resolution in the host preserves the original data
 * flow (config_help.json → component) without giving the modal HTTP/schema deps.
 */
export interface ConfigHelpData {
    /** Human-readable field title. */
    title: string;
    /** Short tooltip string shown under the title. */
    tip: string;
    /** Pre-rendered (safe) detail HTML — bold/code/list markup only. */
    detailHtml: string;
}

@Component({
    selector: 'app-modal-config-help',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">HELP</div>
                <div class="ch-title" data-testid="config-help-title">{{ data().title }}</div>
                @if (data().tip) {
                    <div class="ch-tip">{{ data().tip }}</div>
                }
            </div>
            <button class="icon-btn" type="button" (click)="close()"
                    data-testid="config-help-close" aria-label="Close">×</button>
        </div>

        <div class="modal-body">
            <div class="ch-detail" data-testid="config-help-detail"
                 [innerHTML]="data().detailHtml"></div>
        </div>

        <div class="modal-foot">
            <button class="btn primary" type="button" (click)="close()"
                    data-testid="config-help-got-it">Got it</button>
        </div>
    `,
    styles: [`
        .ch-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .ch-tip { font-size: 11px; color: var(--color-text-subtle); margin-top: 2px; }
        .ch-detail { font-size: 13px; color: var(--color-text-secondary); line-height: 1.6; }
        .ch-detail :is(ul) { margin: 6px 0; padding-left: 18px; }
        .ch-detail :is(code) {
            font-family: var(--font-mono, monospace);
            background: var(--color-surface-mid);
            padding: 1px 5px; border-radius: 4px; font-size: 12px;
        }
    `],
})
export class ConfigHelpModalComponent {
    protected overlay = inject(OverlayStore);

    protected data = computed<ConfigHelpData>(
        () => (this.overlay.topModal()?.data ?? { title: '', tip: '', detailHtml: '' }) as ConfigHelpData,
    );

    protected close(): void {
        this.overlay.closeModal();
    }
}
