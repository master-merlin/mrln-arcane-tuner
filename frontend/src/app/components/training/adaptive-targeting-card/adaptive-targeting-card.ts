import {
  ChangeDetectionStrategy, Component, DestroyRef, OnInit, computed, inject, input, output, signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormControl } from '@angular/forms';
import { EMPTY, Subject } from 'rxjs';
import { catchError, debounceTime, switchMap } from 'rxjs/operators';

import { TemplateService, type Template } from '../../../services/template.service';
import { ToastService } from '../../../services/toast';
import { isTruthyFlag } from '../../../shared/truthy-flag';

/**
 * The materialized knob dict stored in `adaptive_targeting_config`.
 *
 * Declared as a TYPE (not an interface) so it stays assignable to the
 * `Record<string, unknown>` the form control and the template `config` field
 * are typed with.
 */
export type AdaptiveKnobs = {
  /** Provenance ONLY — `factory:<name>` or a user preset id. The trainer never
   *  resolves it; every value below is authoritative (spec §4 / D1). */
  preset: string;
  warmup_pct: number;
  interval_steps: number;
  energy_threshold: number;
  min_active_pct: number;
  heat_ema: number;
  reactivation: boolean;
  probe_every: number;
  probe_steps: number;
  action: 'freeze' | 'rebuild';
  rebuild_min_shrink_pct: number;
};

/**
 * Mirrors `AdaptiveTargetingConfig`'s field defaults (backend
 * `app/engine/models/adaptive.py`), which are the "balanced" factory numbers
 * plus the probe/action knobs. Duplicated deliberately: the control's value must
 * be a complete, self-contained dict even before (or without) the preset list
 * request — the backend re-validates every submitted config regardless.
 */
export const ADAPTIVE_TARGETING_DEFAULTS: AdaptiveKnobs = {
  preset: 'factory:balanced',
  warmup_pct: 0.25,
  interval_steps: 200,
  energy_threshold: 0.93,
  min_active_pct: 0.25,
  heat_ema: 0.5,
  reactivation: false,
  probe_every: 5,
  probe_steps: 30,
  action: 'freeze',
  rebuild_min_shrink_pct: 25,
};

/** Namespace for this card's `config_help.json` keys. */
const HELP_NS = 'adaptive_targeting';

type SliderKey = 'warmup_pct' | 'energy_threshold' | 'min_active_pct' | 'heat_ema';

interface SliderKnob {
  key: SliderKey;
  label: string;
  /** Widget bounds. They are ALSO the clamp bounds, and every one of them sits
   *  strictly inside the backend's validated interval, so no slider position
   *  can compose a config `AdaptiveTargetingConfig` would reject. */
  min: number;
  max: number;
  step: number;
  /** true → render as a percentage. */
  pct: boolean;
  hint: string;
}

const SLIDERS: readonly SliderKnob[] = [
  { key: 'warmup_pct', label: 'Warm-up', min: 0, max: 0.95, step: 0.05, pct: true,
    hint: 'Share of the run trained with every layer active before the first measurement.' },
  { key: 'energy_threshold', label: 'Energy kept', min: 0.5, max: 1, step: 0.01, pct: true,
    hint: 'Keep the hottest layers until this share of total movement is covered.' },
  { key: 'min_active_pct', label: 'Floor', min: 0.05, max: 1, step: 0.05, pct: true,
    hint: 'Never leave fewer than this share of LoRA modules active.' },
  { key: 'heat_ema', label: 'Heat smoothing', min: 0, max: 0.95, step: 0.05, pct: false,
    hint: 'EMA factor applied to each layer\'s measured heat (higher = slower to react).' },
];

const MIN_INTERVAL_STEPS = 10;
const MIN_PROBE_EVERY = 2;

/** Same window the training form's template autosave uses
 *  (`template-autosave.service.ts`) — one save per burst of edits. */
export const PRESET_SAVE_DEBOUNCE_MS = 1200;

function num(v: unknown, fallback: number): number {
  const n = typeof v === 'number' ? v : Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function clampInt(v: unknown, fallback: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Math.round(num(v, fallback))));
}

function clampFloat(v: unknown, fallback: number, min: number, max: number): number {
  const n = Math.min(max, Math.max(min, num(v, fallback)));
  // Guard against slider/keyboard float noise leaking into the persisted config.
  return Math.round(n * 10000) / 10000;
}

/**
 * Advanced-Engine card for adaptive LoRA layer targeting.
 *
 * The bound control ALWAYS holds the fully materialized knob dict (D1) — a job
 * config must be interpretable without a template lookup, because the trainer
 * never reads the `adaptive` template domain. Presets are a convenience layer:
 * selecting one copies its knobs in, and editing a knob writes back to the
 * selected preset (auto-branching first if it is a readonly factory row, D2).
 *
 * Both backend cross-field rules are mirrored here (D3) so the form cannot
 * compose a payload the API would 400 on; a rejection that slips through
 * anyway is surfaced with the backend's own message, never swallowed.
 */
@Component({
  selector: 'app-adaptive-targeting-card',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="bg-surface-mid/30 border border-surface-mid rounded-theme-xl overflow-hidden mt-4"
         data-testid="adaptive-targeting-card">

      <div class="flex items-center justify-between gap-3 p-4 border-b border-surface-mid bg-surface-low/50">
        <div class="flex items-center gap-3 min-w-0">
          <div class="p-2 bg-brand/10 text-brand rounded-theme-md shrink-0">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"
                 fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 12h4l3 8 4-16 3 8h4"></path>
            </svg>
          </div>
          <div class="min-w-0">
            <h3 class="text-sm font-bold text-white tracking-widest uppercase flex items-center gap-1.5">
              Adaptive Layer Targeting
              @if (hasHelp('')) {
                <span class="config-help-icon" [title]="helpTip('')"
                      data-testid="adaptive-help-adaptive_targeting"
                      (click)="requestHelp(''); $event.preventDefault()">?</span>
              }
            </h3>
            <p class="text-xs text-text-subtle mt-0.5">
              Periodically measures which LoRA modules still move and freezes the cold ones,
              so late training concentrates on the layers that matter.
            </p>
          </div>
        </div>
        @if (enabled()) {
          <div class="text-[10px] font-black uppercase tracking-[0.1em] px-3 py-1 rounded-full shrink-0"
               [class]="knobs().action === 'rebuild'
                 ? 'text-warning bg-warning/10 border border-warning/20'
                 : 'text-brand bg-brand/10 border border-brand/20'"
               data-testid="adaptive-action-chip">
            {{ knobs().action === 'rebuild' ? 'Rebuild' : 'Freeze' }}
          </div>
        }
      </div>

      @if (!enabled()) {
        <div class="p-8 text-center text-text-subtle text-sm" data-testid="adaptive-disabled-hint">
          Enable <span class="text-text-secondary font-semibold">Adaptive targeting</span> above to configure
          presets and measurement knobs.
        </div>
      } @else {
        <div class="p-4 flex flex-col gap-4">

          <!-- Preset ------------------------------------------------------ -->
          <div class="flex flex-col gap-1.5">
            <label class="field-label flex items-center gap-1.5" for="adaptive-preset">
              Preset
              @if (hasHelp('preset')) {
                <span class="config-help-icon" [title]="helpTip('preset')"
                      data-testid="adaptive-help-preset"
                      (click)="requestHelp('preset'); $event.preventDefault()">?</span>
              }
            </label>
            <select id="adaptive-preset" class="select" data-testid="adaptive-preset-select"
                    (change)="selectPreset($any($event.target).value)">
              @for (o of presetOptions(); track o.ref) {
                <option [value]="o.ref" [selected]="o.ref === knobs().preset"
                        data-testid="adaptive-preset-option">{{ o.label }}</option>
              }
            </select>
            <p class="text-[10.5px] text-text-muted">
              Presets only seed the values below — the run stores the knobs themselves, so a
              preset edited later never changes an already-queued job.
            </p>
          </div>

          <!-- Measurement knobs ------------------------------------------- -->
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3">
            @for (s of sliders; track s.key) {
              <div class="flex flex-col gap-1">
                <label class="field-label flex items-center justify-between gap-1.5">
                  <span class="flex items-center gap-1.5">
                    {{ s.label }}
                    @if (hasHelp(s.key)) {
                      <span class="config-help-icon" [title]="helpTip(s.key)"
                            [attr.data-testid]="'adaptive-help-' + s.key"
                            (click)="requestHelp(s.key); $event.preventDefault()">?</span>
                    }
                  </span>
                  <span class="font-mono text-brand-light">{{ display(s) }}</span>
                </label>
                <input type="range" class="w-full accent-brand"
                       [attr.data-testid]="'adaptive-knob-' + s.key"
                       [min]="s.min" [max]="s.max" [step]="s.step"
                       [value]="knobs()[s.key]"
                       (change)="onNumberKnob(s.key, $event)">
                <p class="text-[10.5px] text-text-muted">{{ s.hint }}</p>
              </div>
            }

            <div class="flex flex-col gap-1.5">
              <label class="field-label flex items-center gap-1.5" for="adaptive-interval">
                Measure every (steps)
                @if (hasHelp('interval_steps')) {
                  <span class="config-help-icon" [title]="helpTip('interval_steps')"
                        data-testid="adaptive-help-interval_steps"
                        (click)="requestHelp('interval_steps'); $event.preventDefault()">?</span>
                }
              </label>
              <input id="adaptive-interval" type="number" class="input"
                     data-testid="adaptive-knob-interval_steps"
                     [min]="minIntervalSteps" step="10"
                     [value]="knobs().interval_steps"
                     (change)="onNumberKnob('interval_steps', $event)">
            </div>

            <div class="flex flex-col gap-1.5">
              <label class="field-label flex items-center gap-1.5" for="adaptive-action">
                Action
                @if (hasHelp('action')) {
                  <span class="config-help-icon" [title]="helpTip('action')"
                        data-testid="adaptive-help-action"
                        (click)="requestHelp('action'); $event.preventDefault()">?</span>
                }
              </label>
              <select id="adaptive-action" class="select" data-testid="adaptive-knob-action"
                      (change)="selectAction($any($event.target).value)">
                <option value="freeze" [selected]="knobs().action === 'freeze'">Freeze cold layers</option>
                <option value="rebuild" [selected]="knobs().action === 'rebuild'">Rebuild adapter (restart)</option>
              </select>
            </div>

            @if (knobs().action === 'rebuild') {
              <div class="flex flex-col gap-1">
                <label class="field-label flex items-center justify-between gap-1.5">
                  <span class="flex items-center gap-1.5">
                    Min shrink to rebuild
                    @if (hasHelp('rebuild_min_shrink_pct')) {
                      <span class="config-help-icon" [title]="helpTip('rebuild_min_shrink_pct')"
                            data-testid="adaptive-help-rebuild_min_shrink_pct"
                            (click)="requestHelp('rebuild_min_shrink_pct'); $event.preventDefault()">?</span>
                    }
                  </span>
                  <span class="font-mono text-brand-light">{{ knobs().rebuild_min_shrink_pct }}%</span>
                </label>
                <input type="range" class="w-full accent-brand"
                       data-testid="adaptive-knob-rebuild_min_shrink_pct"
                       min="1" max="100" step="1"
                       [value]="knobs().rebuild_min_shrink_pct"
                       (change)="onNumberKnob('rebuild_min_shrink_pct', $event)">
                <p class="text-[10.5px] text-text-muted">
                  Only restart with a smaller adapter when the selection drops the module count
                  by at least this much.
                </p>
              </div>
            }
          </div>

          <!-- Reactivation ------------------------------------------------ -->
          <div class="border-t border-surface-mid pt-3 flex flex-col gap-3">
            <div class="flex items-start gap-2.5">
              <label class="relative inline-flex items-center cursor-pointer group shrink-0 mt-0.5"
                     [class.opacity-50]="reactivationLocked()"
                     [class.cursor-not-allowed]="reactivationLocked()">
                <input type="checkbox" class="sr-only peer"
                       data-testid="adaptive-knob-reactivation"
                       [checked]="knobs().reactivation"
                       [disabled]="reactivationLocked()"
                       (change)="setReactivation($any($event.target).checked)">
                <div class="w-7 h-4 bg-surface-high/50 border border-surface-mid rounded-full peer peer-focus:ring-2 peer-focus:ring-brand/50 peer-checked:after:translate-x-3 after:content-[''] after:absolute after:top-[1px] after:left-[2px] after:bg-white after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-brand group-hover:bg-surface-mid transition-all relative"></div>
              </label>
              <div class="flex flex-col gap-0.5">
                <span class="text-[11px] font-medium text-text-secondary flex items-center gap-1.5">
                  Re-check frozen layers
                  @if (hasHelp('reactivation')) {
                    <span class="config-help-icon" [title]="helpTip('reactivation')"
                          data-testid="adaptive-help-reactivation"
                          (click)="requestHelp('reactivation'); $event.preventDefault()">?</span>
                  }
                </span>
                <p class="text-[10px] text-text-subtle italic leading-snug">
                  @if (reactivationLocked()) {
                    Unavailable in rebuild mode — a rebuilt adapter holds no optimizer state for
                    frozen layers, so a probe would measure zero heat.
                  } @else {
                    Periodically unfreeze everything for a short probe window so a layer that
                    starts learning again can come back.
                  }
                </p>
              </div>
            </div>

            <div class="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3"
                 [class.opacity-50]="!knobs().reactivation">
              <div class="flex flex-col gap-1.5">
                <label class="field-label flex items-center gap-1.5" for="adaptive-probe-every">
                  Probe every N windows
                  @if (hasHelp('probe_every')) {
                    <span class="config-help-icon" [title]="helpTip('probe_every')"
                          data-testid="adaptive-help-probe_every"
                          (click)="requestHelp('probe_every'); $event.preventDefault()">?</span>
                  }
                </label>
                <input id="adaptive-probe-every" type="number" class="input"
                       data-testid="adaptive-knob-probe_every"
                       [min]="minProbeEvery" step="1"
                       [disabled]="!knobs().reactivation"
                       [value]="knobs().probe_every"
                       (change)="onNumberKnob('probe_every', $event)">
              </div>
              <div class="flex flex-col gap-1.5">
                <label class="field-label flex items-center gap-1.5" for="adaptive-probe-steps">
                  Probe length (steps)
                  @if (hasHelp('probe_steps')) {
                    <span class="config-help-icon" [title]="helpTip('probe_steps')"
                          data-testid="adaptive-help-probe_steps"
                          (click)="requestHelp('probe_steps'); $event.preventDefault()">?</span>
                  }
                </label>
                <input id="adaptive-probe-steps" type="number" class="input"
                       data-testid="adaptive-knob-probe_steps"
                       min="1" [max]="maxProbeSteps()" step="1"
                       [disabled]="!knobs().reactivation"
                       [value]="knobs().probe_steps"
                       (change)="onNumberKnob('probe_steps', $event)">
                <p class="text-[10.5px] text-text-muted">Must stay below the measurement interval.</p>
              </div>
            </div>
          </div>
        </div>
      }
    </div>
  `,
})
export class AdaptiveTargetingCardComponent implements OnInit {
  /** The `adaptive_targeting_config` form control (a plain FormControl holding a dict). */
  control = input.required<FormControl<Record<string, unknown>>>();
  /** Mirrors the `adaptive_targeting` toggle — knobs are only editable when on. */
  enabled = input(true);
  /** Active project scope, so a branched preset lands in the right scope. */
  projectId = input<string | null>(null);
  /** The parent's already-loaded `config_help.json` map (same idiom as dynamic-form-group). */
  configHelp = input<Record<string, { tip: string; detail: string }>>({});

  /** Namespaced help key (`adaptive_targeting.<knob>`) for the parent's modal. */
  helpRequested = output<string>();

  private templates = inject(TemplateService);
  private toast = inject(ToastService);
  private destroyRef = inject(DestroyRef);

  readonly sliders = SLIDERS;
  readonly minIntervalSteps = MIN_INTERVAL_STEPS;
  readonly minProbeEvery = MIN_PROBE_EVERY;

  /** Local mirror of the control value — the single source the template reads. */
  readonly knobs = signal<AdaptiveKnobs>(ADAPTIVE_TARGETING_DEFAULTS);
  readonly presets = signal<Template[]>([]);

  /** A branch POST is in flight; further edits must not start a second one. */
  private branchInFlight = false;

  /** Debounced write path — one save per burst of edits. */
  private edits$ = new Subject<AdaptiveKnobs>();
  /** Autosave into an EXISTING user preset (the PUT path only). */
  private putSaves$ = new Subject<{ id: string; config: AdaptiveKnobs }>();

  constructor() {
    this.edits$.pipe(
      debounceTime(PRESET_SAVE_DEBOUNCE_MS),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe(value => this.persist(value));

    this.putSaves$.pipe(
      // A later save supersedes an in-flight one. Every PUT carries the FULL
      // dict, so two overlapping saves arriving out of order would leave the
      // preset row holding the older knob set. The BRANCH path deliberately
      // stays outside this pipe: cancelling a create client-side does not
      // un-create the row on the server, and the next emission would branch a
      // second time.
      switchMap(({ id, config }) => this.templates.updateTemplate('adaptive', id, { config }).pipe(
        catchError(err => {
          this.toast.error('Could not save adaptive preset: ' + this.msg(err));
          // Swallow into EMPTY, never into the outer stream: an error reaching
          // the outer subscription would tear the pipe down and silently kill
          // every later autosave.
          return EMPTY;
        }),
      )),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe();
  }

  /** Reference under which a preset row is selected: factory rows key on their
   *  `config.preset` (`factory:<name>`), user rows on their id. */
  private refOf(t: Template): string {
    const carried = (t.config as Record<string, unknown> | undefined)?.['preset'];
    return t.readonly && typeof carried === 'string' && carried ? carried : t.id;
  }

  readonly selectedPreset = computed<Template | null>(() => {
    const ref = this.knobs().preset;
    return this.presets().find(t => this.refOf(t) === ref || t.id === ref) ?? null;
  });

  /** Factory rows first, each followed by the user presets branched from it;
   *  anything unparented trails at the end. */
  readonly presetOptions = computed<{ ref: string; label: string }[]>(() => {
    const all = this.presets();
    const parents = all.filter(t => t.readonly);
    const parentIds = new Set(parents.map(p => p.id));
    const out: { ref: string; label: string }[] = [];
    for (const p of parents) {
      out.push({ ref: this.refOf(p), label: p.name });
      for (const c of all) {
        if (!c.readonly && c.branched_from === p.id) {
          out.push({ ref: this.refOf(c), label: `↳ ${c.name}` });
        }
      }
    }
    for (const c of all) {
      if (!c.readonly && (!c.branched_from || !parentIds.has(c.branched_from))) {
        out.push({ ref: this.refOf(c), label: c.name });
      }
    }
    return out;
  });

  /** Rebuild is contractually monotonic — reactivation cannot be combined with it. */
  readonly reactivationLocked = computed(() => this.knobs().action === 'rebuild');
  readonly maxProbeSteps = computed(() => Math.max(1, this.knobs().interval_steps - 1));

  ngOnInit(): void {
    this.refreshFromControl();
    this.loadPresets();
  }

  /** Re-read (and re-normalize) the control — called by the parent after a
   *  template apply / defaults reset patched the control from outside. */
  refreshFromControl(): void {
    const next = this.normalize(this.control().value);
    this.knobs.set(next);
    if (!this.sameValue(this.control().value, next)) {
      // A normalization, not a user edit: never let it trigger template autosave.
      this.control().setValue(next, { emitEvent: false });
    }
  }

  private loadPresets(): void {
    this.templates.listAdaptivePresets(this.projectId()).subscribe({
      next: list => this.presets.set(list ?? []),
      error: err => this.toast.error('Could not load adaptive presets: ' + this.msg(err)),
    });
  }

  // ── Help ────────────────────────────────────────────────────────────────
  /** `''` addresses the feature toggle itself; any other key is namespaced. */
  private helpKey(knob: string): string {
    return knob ? `${HELP_NS}.${knob}` : HELP_NS;
  }
  hasHelp(knob: string): boolean { return !!this.configHelp()[this.helpKey(knob)]; }
  helpTip(knob: string): string { return this.configHelp()[this.helpKey(knob)]?.tip ?? ''; }
  requestHelp(knob: string): void { this.helpRequested.emit(this.helpKey(knob)); }

  display(s: SliderKnob): string {
    const v = this.knobs()[s.key];
    return s.pct ? `${Math.round(v * 100)}%` : v.toFixed(2);
  }

  // ── Edits ───────────────────────────────────────────────────────────────

  selectPreset(ref: string): void {
    const previous = this.knobs().preset;
    const t = this.presets().find(p => this.refOf(p) === ref);
    // Selecting a preset COPIES its knobs in; it never leaves the control
    // holding a bare reference (D1). Unknown ref → keep the current values.
    // Knobs a preset does not carry (the factory rows define only the five
    // measurement values) keep whatever is currently set — they are orthogonal
    // to the preset's quality/speed tradeoff.
    const carried = (t?.config ?? {}) as Partial<AdaptiveKnobs>;
    const next = this.normalize({ ...this.knobs(), ...carried, preset: ref });
    this.knobs.set(next);
    this.control().setValue(next);
    this.control().markAsDirty();
    // Selecting is not editing — nothing is written back to the preset.
    // The ROW id, never the `factory:*` provenance ref. Not reached from the
    // hydration path, which materializes knobs without going through here.
    if (t && ref !== previous) this.templates.recordUse('adaptive', t.id);
  }

  selectAction(raw: string): void {
    const action: 'freeze' | 'rebuild' = raw === 'rebuild' ? 'rebuild' : 'freeze';
    // reactivation is forced off in the SAME patch, so no intermediate value
    // ever violates the backend rule.
    this.commit(action === 'rebuild' ? { action, reactivation: false } : { action });
  }

  setReactivation(on: boolean): void {
    if (this.reactivationLocked()) return;
    this.commit({ reactivation: on });
  }

  onNumberKnob(key: SliderKey | 'interval_steps' | 'probe_every' | 'probe_steps'
    | 'rebuild_min_shrink_pct', ev: Event): void {
    const node = ev.target as HTMLInputElement;
    this.commit({ [key]: Number(node.value) } as Partial<AdaptiveKnobs>);
    // Reflect a clamped value back into the widget. The `[value]` binding only
    // writes when the BOUND value changed, so a rejected out-of-range entry
    // that normalizes to the current value would otherwise stay on screen.
    const applied = String(this.knobs()[key]);
    if (node.value !== applied) node.value = applied;
  }

  private commit(patch: Partial<AdaptiveKnobs>): void {
    const next = this.normalize({ ...this.knobs(), ...patch });
    // A no-op edit must not reach the write path: typing a value that clamps
    // back to the current one (5 into interval_steps when it is already 10)
    // would otherwise branch a "<Parent> (custom)" preset for a change that
    // changed nothing.
    if (this.sameValue(this.knobs(), next)) return;
    this.knobs.set(next);
    this.control().setValue(next);
    this.control().markAsDirty();
    this.edits$.next(next);
  }

  /**
   * Write the edited knobs back to the selected preset: a readonly factory row
   * is branched into a user preset first (D2), everything else autosaves in
   * place. An edit that arrives while a branch is in flight is NOT dropped —
   * it is already in `knobs()`, and the post-branch write below persists that
   * latest value.
   */
  private persist(value: AdaptiveKnobs): void {
    const current = this.selectedPreset();
    if (!current) return;

    if (this.branchInFlight) return;

    if (current.readonly) {
      this.branchInFlight = true;
      this.templates.createAdaptivePreset({
        name: `${current.name} (custom)`,
        project_id: this.projectId(),
        branched_from: current.id,
        config: { ...value },
      }).subscribe({
        next: created => {
          this.branchInFlight = false;
          // Insert optimistically so `selectedPreset()` resolves the new row
          // synchronously (a re-list would race the next edit).
          this.presets.update(list => list.some(t => t.id === created.id) ? list : [...list, created]);
          const latest: AdaptiveKnobs = { ...this.knobs(), preset: created.id };
          this.knobs.set(latest);
          this.control().setValue(latest);
          // The POST body necessarily carried the PARENT's `preset` ref — the
          // new id did not exist yet. Write the corrected dict straight back so
          // the stored row's provenance is self-consistent (D1). `latest` also
          // carries any edit made while the POST was in flight, so this is the
          // flush for those too.
          this.persist(latest);
        },
        error: err => {
          // Always release the guard — a wedged flag would silently disable
          // every later save for the life of the card.
          this.branchInFlight = false;
          this.toast.error('Could not save adaptive preset: ' + this.msg(err));
        },
      });
      return;
    }

    this.putSaves$.next({ id: current.id, config: { ...value } });
  }

  // ── Normalization ───────────────────────────────────────────────────────

  /**
   * Coerce an arbitrary (possibly partial, possibly stringified) dict into a
   * complete, correctly-typed, backend-valid knob set. The schema declares a
   * bare `dict`, so the submit-time numeric coercion cannot reach inside this
   * field — typing has to be right here.
   */
  private normalize(raw: unknown): AdaptiveKnobs {
    const d = ADAPTIVE_TARGETING_DEFAULTS;
    const src = (raw && typeof raw === 'object' && !Array.isArray(raw))
      ? raw as Record<string, unknown> : {};

    const preset = typeof src['preset'] === 'string' && src['preset'] ? src['preset'] as string : d.preset;
    const action: 'freeze' | 'rebuild' = src['action'] === 'rebuild' ? 'rebuild' : 'freeze';
    const interval = clampInt(src['interval_steps'], d.interval_steps, MIN_INTERVAL_STEPS, 1_000_000);

    const slider = (k: SliderKey) => {
      const s = SLIDERS.find(x => x.key === k)!;
      return clampFloat(src[k], d[k], s.min, s.max);
    };

    return {
      preset,
      warmup_pct: slider('warmup_pct'),
      interval_steps: interval,
      energy_threshold: slider('energy_threshold'),
      min_active_pct: slider('min_active_pct'),
      heat_ema: slider('heat_ema'),
      // `rebuild` cannot carry reactivation (backend rule) — collapse rather
      // than surface an invalid pair. Read through the shared flag predicate:
      // a value round-tripped through a template/JSON editor arrives as the
      // STRING "true", which a strict `=== true` would silently drop to false.
      reactivation: action === 'rebuild' ? false : isTruthyFlag(src['reactivation']),
      probe_every: clampInt(src['probe_every'], d.probe_every, MIN_PROBE_EVERY, 1_000),
      // probe_steps < interval_steps, enforced against the interval decided above.
      probe_steps: clampInt(src['probe_steps'], d.probe_steps, 1, Math.max(1, interval - 1)),
      action,
      rebuild_min_shrink_pct: clampFloat(src['rebuild_min_shrink_pct'], d.rebuild_min_shrink_pct, 1, 100),
    };
  }

  private sameValue(a: unknown, b: AdaptiveKnobs): boolean {
    if (!a || typeof a !== 'object') return false;
    const src = a as Record<string, unknown>;
    const keys = Object.keys(b) as (keyof AdaptiveKnobs)[];
    if (Object.keys(src).length !== keys.length) return false;
    return keys.every(k => src[k] === b[k]);
  }

  private msg(err: unknown): string {
    const e = err as { error?: { detail?: string }; message?: string } | undefined;
    return e?.error?.detail ?? e?.message ?? 'unknown error';
  }
}
