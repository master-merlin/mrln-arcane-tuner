import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';

/**
 * Model source modal — stub.
 *
 * The full design (modals-more.jsx → ModelSourceModal) lets the user pick
 * a model from HuggingFace, a local Diffusers folder, or a single
 * safetensors file. None of those flows have a backend behind them yet,
 * so this PR ships a backend-aware placeholder that explains the gap and
 * lets the user dismiss the modal.
 *
 * TODO(backend): HF + CivitAI search + download endpoints; once those
 * land, replace this body with the full source-picker UI (source type
 * radios, path validation, "Skip HF updates" toggle, etc.).
 */
@Component({
    selector: 'app-modal-model-source',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">DOWNLOAD</div>
                <div class="modal-title">Model Source</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>
        <div class="modal-body">
            <p class="muted">
                Hugging Face / CivitAI integration is not yet wired to a backend.
                Once the search + download endpoints land, this dialog will let you
                pick a model by ID, browse local Diffusers folders, or load a single
                <code class="mono">.safetensors</code> file.
            </p>
            <!-- TODO(backend): HF + CivitAI search + download endpoints -->
        </div>
        <div class="modal-foot">
            <button class="btn ghost" type="button" (click)="overlay.closeModal()">Close</button>
        </div>
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .muted { color: var(--color-text-muted); font-size: 13px; line-height: 1.6; }
        .mono { font-family: var(--f-mono); padding: 1px 6px; background: var(--color-surface-mid); border-radius: 3px; }
    `],
})
export class ModelSourceModalComponent {
    protected overlay = inject(OverlayStore);
}
