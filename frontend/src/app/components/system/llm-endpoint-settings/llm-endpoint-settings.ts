import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { IcoComponent } from '../../../icons/ico.component';
import { LlmSettingsService } from '../../../services/llm-settings.service';
import { DatasetService } from '../../../services/dataset';
import { LlmAvailabilityStore } from '../../../state/llm-availability.store';
import { ToastService } from '../../../services/toast';

type LlmProvider = 'ollama' | 'lmstudio';

/** Default base-URL hint per provider. */
const PROVIDER_DEFAULTS: Record<LlmProvider, string> = {
    ollama: 'http://localhost:11434',
    lmstudio: 'http://localhost:1234/v1',
};

/**
 * LLM endpoint config card for the Server screen. Reads/writes the
 * `llm_refine` settings module (base URL + provider) and offers a
 * "Save & Test" action that persists the settings then probes
 * `GET /api/llm-refine/models`, showing reachability + model count and
 * refreshing the shared {@link LlmAvailabilityStore}.
 */
@Component({
    selector: 'app-llm-endpoint-settings',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [FormsModule, IcoComponent],
    template: `
    <div class="card">
        <div class="card-head"><div class="card-title"><app-ico name="Bot" [size]="11" /> LLM Refine Endpoint</div></div>
        <div class="card-body sc-body">
            <div class="sc-grid-2">
                <div class="sc-field">
                    <label class="field-label">Provider</label>
                    <select class="select mono"
                            [ngModel]="provider()"
                            (ngModelChange)="onProviderChange($event)"
                            data-testid="llm-provider">
                        <option value="ollama">Ollama</option>
                        <option value="lmstudio">LM Studio</option>
                    </select>
                    <p class="sc-hint">local LLM server type</p>
                </div>
                <div class="sc-field">
                    <label class="field-label">Base URL</label>
                    <input type="text" class="input mono"
                           [ngModel]="baseUrl()"
                           (ngModelChange)="baseUrl.set($event)"
                           [placeholder]="providerDefault()"
                           data-testid="llm-base-url" />
                    <p class="sc-hint">endpoint of the local LLM server</p>
                </div>
            </div>
            <div class="sc-save-row">
                @if (reachable() === true) {
                    <span class="chip success" data-testid="llm-status"><span class="dot"></span> Reachable · {{ modelCount() }} models</span>
                } @else if (reachable() === false) {
                    <span class="chip danger" data-testid="llm-status"><span class="dot"></span> Unreachable</span>
                }
                <button type="button" class="btn primary sm" (click)="saveAndTest()" [disabled]="testing()"
                        data-testid="llm-save-test">
                    {{ testing() ? 'Testing…' : 'Save & Test' }}
                </button>
            </div>
        </div>
    </div>
  `,
    styles: [`
        .sc-body { display: flex; flex-direction: column; gap: 14px; }
        .sc-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .sc-field { display: flex; flex-direction: column; min-width: 0; }
        .sc-hint { font-size: 10.5px; color: var(--color-text-disabled); margin: 5px 0 0; }
        .sc-save-row { display: flex; align-items: center; justify-content: flex-end; gap: 10px; }
    `],
})
export class LlmEndpointSettingsComponent implements OnInit {
    private settings = inject(LlmSettingsService);
    private datasets = inject(DatasetService);
    private availability = inject(LlmAvailabilityStore);
    private toast = inject(ToastService);

    readonly baseUrl = signal<string>('');
    readonly provider = signal<LlmProvider>('ollama');
    readonly testing = signal<boolean>(false);
    /** null = idle / untested; true = reachable; false = unreachable. */
    readonly reachable = signal<boolean | null>(null);
    readonly modelCount = signal<number>(0);

    readonly providerDefault = computed(() => PROVIDER_DEFAULTS[this.provider()]);

    ngOnInit(): void {
        this.settings.get().subscribe({
            next: s => {
                if (s.base_url) this.baseUrl.set(s.base_url);
                if (s.provider === 'ollama' || s.provider === 'lmstudio') this.provider.set(s.provider);
            },
            error: () => { /* leave defaults; the card is still usable */ },
        });
    }

    onProviderChange(p: LlmProvider): void {
        this.provider.set(p);
        // Offer the provider's default URL when the field is empty.
        if (!this.baseUrl().trim()) this.baseUrl.set(PROVIDER_DEFAULTS[p]);
    }

    /** Persist the endpoint settings, then probe the model list to confirm reachability. */
    saveAndTest(): void {
        this.testing.set(true);
        this.reachable.set(null);
        this.settings.save({ base_url: this.baseUrl(), provider: this.provider() }).subscribe({
            next: () => this.probe(),
            error: () => {
                this.testing.set(false);
                this.reachable.set(false);
                this.toast.error('Failed to save LLM endpoint settings.');
            },
        });
    }

    private probe(): void {
        this.datasets.listRefineModels().subscribe({
            next: r => {
                this.testing.set(false);
                this.reachable.set(!!r.available);
                this.modelCount.set((r.installed?.length ?? 0));
                // Sync the shared availability store off this same probe result
                // (no extra HTTP round-trip) so the top-bar icon + LLM-gated
                // controls reflect the freshly-saved endpoint immediately.
                this.availability.available.set(!!r.available);
                this.availability.installed.set(r.installed ?? []);
                this.availability.checked.set(true);
                if (r.available) this.toast.success(`LLM endpoint reachable — ${r.installed?.length ?? 0} models.`);
                else this.toast.warning('LLM endpoint saved but not reachable.');
            },
            error: () => {
                this.testing.set(false);
                this.reachable.set(false);
                this.availability.available.set(false);
                this.availability.installed.set([]);
                this.availability.checked.set(true);
                this.toast.error('LLM endpoint unreachable.');
            },
        });
    }
}
