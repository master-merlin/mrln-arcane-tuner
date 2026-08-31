import {
    ChangeDetectionStrategy, Component, ElementRef, OnInit,
    afterRenderEffect, computed, inject, signal, viewChild,
} from '@angular/core';
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
 * `llm_refine` settings module (base URL + provider + default model) and
 * offers a "Save & Test" action that persists the settings then probes
 * `GET /api/llm-refine/models`, showing reachability + model count and
 * refreshing the shared {@link LlmAvailabilityStore}.
 *
 * The default-model picker exists because the card configured an endpoint you
 * could not choose a model on. The backend half was already there and already
 * consumed — `llm_refine.model` is what `_default_model()` reads
 * (`api/llm_refine_routes.py:50`) — but nothing ever wrote it, so every refine
 * fell back to `CURATED_MODELS[0]` or to whatever the per-dataset panel
 * happened to pick first. This is the writer.
 *
 * Models are probed on init, not only on "Save & Test": a picker that is empty
 * until you press a button is not a picker.
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
            <div class="sc-field">
                <label class="field-label">Default model</label>
                <!-- Deliberately NOT ngModel. This select's options arrive from
                     an HTTP response, so the value is written before they exist,
                     and ngModel's writeValue does not retry: measured
                     selectedIndex -1 for [ngValue], [value] and an optgroup
                     variant, and 0 for this plain binding. A blank select here
                     is not cosmetic — Save would persist the blank.

                     [value] alone is not enough either, and that is LANE-49:
                     when the bound value matches no option YET, the browser's
                     selectedness reset picks the first ENABLED option, so a
                     stored empty value painted the first installed model. The
                     afterRenderEffect in the constructor re-applies the value
                     after every render. -->
                <select #modelSelect class="select mono"
                        [value]="model()"
                        (change)="model.set($any($event.target).value)"
                        data-testid="llm-model">
                    <!-- Always rendered and never disabled: this is the option
                         that must be SELECTABLE when nothing is saved, both so
                         the control can display "nothing" and so the user can
                         go back to no default. -->
                    <option value="">{{ noDefaultLabel() }}</option>
                    @if (orphanModel(); as orphan) {
                        <optgroup label="Configured (not reported by this endpoint)">
                            <option [value]="orphan">{{ orphan }}</option>
                        </optgroup>
                    }
                    @if (installed().length) {
                        <optgroup label="Installed">
                            @for (m of installed(); track m) { <option [value]="m">{{ m }}</option> }
                        </optgroup>
                    }
                    @if (suggested().length) {
                        <optgroup label="Suggested — pull before first use">
                            @for (m of suggested(); track m) { <option [value]="m">{{ m }}</option> }
                        </optgroup>
                    }
                </select>
                <div class="sc-model-row">
                    <p class="sc-hint" data-testid="llm-model-hint">
                        @if (!model()) { no default saved — caption refine will use {{ fallbackModel() || 'the backend default' }} }
                        @else if (selectedIsInstalled()) { installed on this endpoint }
                        @else { not installed — refine will fail until it is pulled }
                    </p>
                    @if (model() && !selectedIsInstalled()) {
                        <button type="button" class="btn sm" (click)="pullSelected()"
                                [disabled]="pulling() !== null || reachable() === false"
                                data-testid="llm-model-pull">
                            {{ pulling() === model() ? 'Pulling…' : 'Pull' }}
                        </button>
                    }
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
        .sc-model-row { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; }
        .sc-model-row .sc-hint { flex: 1 1 auto; min-width: 0; }
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

    /** Persisted default refine model (`llm_refine.model`). */
    readonly model = signal<string>('');
    /** Models the endpoint reports as present. */
    readonly installed = signal<string[]>([]);
    /** Backend's curated list, whether or not they are installed. */
    readonly curated = signal<string[]>([]);
    /** Tag currently being pulled, or null. */
    readonly pulling = signal<string | null>(null);

    private readonly modelSelect = viewChild<ElementRef<HTMLSelectElement>>('modelSelect');

    readonly providerDefault = computed(() => PROVIDER_DEFAULTS[this.provider()]);

    /**
     * What the backend falls back to when `llm_refine.model` is unset.
     *
     * `curated[0]` IS that fallback — `refine_settings.DEFAULT_MODEL` is
     * `CURATED_MODELS[0]` and the equality is pinned server-side
     * (`test_refine_settings_empty_as_absent.py::test_the_curated_default_is_the_first_curated_model`),
     * so naming it here does not invent a second source of truth.
     */
    readonly fallbackModel = computed(() => this.curated()[0] ?? '');

    /** Label of the "nothing is saved" option. Says what will happen, not just
     *  that a choice is missing. */
    readonly noDefaultLabel = computed(() => {
        const fallback = this.fallbackModel();
        return fallback ? `No default — uses ${fallback}` : 'No default';
    });

    /** Curated models not present on the endpoint — the "pull me" set. */
    readonly suggested = computed(() =>
        this.curated().filter(c => !this.installed().includes(c)),
    );

    readonly selectedIsInstalled = computed(() => this.installed().includes(this.model()));

    /**
     * A saved model the endpoint does not currently list — Ollama down, a model
     * deleted, a URL pointed somewhere else. It gets its own option so the
     * `<select>` has something to bind to: an ngModel value absent from the
     * option list renders blank, and the next Save would then persist that
     * blank over a setting the user never touched.
     */
    readonly orphanModel = computed(() => {
        const saved = this.model().trim();
        if (!saved) return null;
        return this.installed().includes(saved) || this.suggested().includes(saved) ? null : saved;
    });

    constructor() {
        // LANE-49: the control must display what is SAVED.
        //
        // `model()` and storage were both "" and the picker still painted
        // `gemma3:12b`, because [value] is applied when the <select> is updated
        // and its @if/@for options are created AFTER that — so the value
        // matches nothing, and the HTML "ask for a reset" algorithm then selects
        // the first enabled option. Every later option change re-runs that
        // algorithm, so this has to re-run too, not just once on load.
        //
        // Reading the option sources makes this fire on exactly those changes.
        afterRenderEffect(() => {
            const want = this.model();
            this.installed(); this.suggested(); this.orphanModel();  // option sources
            const el = this.modelSelect()?.nativeElement;
            if (el && el.value !== want) el.value = want;
        });
    }

    ngOnInit(): void {
        this.settings.get().subscribe({
            next: s => {
                if (s.base_url) this.baseUrl.set(s.base_url);
                if (s.provider === 'ollama' || s.provider === 'lmstudio') this.provider.set(s.provider);
                if (s.model) this.model.set(s.model);
            },
            error: () => { /* leave defaults; the card is still usable */ },
        });
        // Populate the picker without requiring a Save & Test. Failure is not
        // reported here: on first load an unconfigured endpoint is the normal
        // case, and a red toast for it would be noise, not information.
        this.datasets.listRefineModels().subscribe({
            next: r => {
                this.installed.set(r.installed ?? []);
                this.curated.set(r.curated ?? []);
            },
            error: () => { /* leave the picker to whatever Save & Test finds */ },
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
        this.settings.save({
            base_url: this.baseUrl(),
            provider: this.provider(),
            model: this.model(),
        }).subscribe({
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
                this.installed.set(r.installed ?? []);
                this.curated.set(r.curated ?? []);
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

    /**
     * Pull the selected model onto the endpoint.
     *
     * Ollama does NOT fetch on demand: inference goes to
     * `/v1/chat/completions` (`core/llm/ollama_client.py:71`), which answers
     * "model not found" rather than downloading, so an explicit pull is the
     * only way a suggested model becomes usable.
     */
    pullSelected(): void {
        const tag = this.model().trim();
        if (!tag || this.pulling()) return;
        this.pulling.set(tag);
        this.datasets.pullRefineModel(tag).subscribe({
            next: ({ ok }) => {
                this.pulling.set(null);
                if (!ok) { this.toast.error(`Pull failed: ${tag}`); return; }
                this.installed.update(xs => xs.includes(tag) ? xs : [...xs, tag]);
                this.availability.installed.set(this.installed());
                this.modelCount.set(this.installed().length);
                this.toast.success(`Pulled ${tag}.`);
            },
            error: () => { this.pulling.set(null); this.toast.error(`Pull failed: ${tag}`); },
        });
    }
}
