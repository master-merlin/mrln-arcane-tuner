import { TestBed } from '@angular/core/testing';
import { FormControl } from '@angular/forms';
import { Observable, Subject, of, throwError } from 'rxjs';

import { AdaptiveTargetingCardComponent, PRESET_SAVE_DEBOUNCE_MS } from './adaptive-targeting-card';
import { TemplateService, type Template } from '../../../services/template.service';
import { ToastService } from '../../../services/toast';
import { byTestId, allByTestId } from '../../../../testing/by-test-id';

/**
 * Adaptive LoRA layer targeting — the Advanced Engine card that configures the
 * `adaptive_targeting_config` knob dict.
 *
 * The contracts pinned here (spec §4 / controller decisions D1-D3):
 *   D1 the control's value is ALWAYS the fully materialized knob dict; `preset`
 *      is provenance only, so a job config never needs a template lookup;
 *   D2 editing a knob while a READONLY factory preset is selected auto-branches
 *      a user preset ONCE (`branched_from` + "<Parent> (custom)"), then keeps
 *      autosaving into it via PUT;
 *   D3 the two backend cross-field rules are mirrored client-side so an invalid
 *      payload cannot be composed: `probe_steps < interval_steps`, and
 *      `action: 'rebuild'` is incompatible with `reactivation: true`.
 *
 * Assertions are on the RENDERED DOM and the emitted control value, never on
 * the card's internals.
 */

function tpl(over: Partial<Template>): Template {
  return {
    id: 'x', name: 'N', project_id: null, config: {}, created_at: 0, updated_at: 0,
    used_count: 0, is_default: false, readonly: false, ...over,
  } as Template;
}

/** The three seeded factory rows (Task 9): readonly, `is_default = 0`. */
const FACTORY: Template[] = [
  tpl({
    id: 'factory-conservative', name: 'Conservative', readonly: true,
    config: {
      preset: 'factory:conservative', warmup_pct: 0.4, interval_steps: 300,
      energy_threshold: 0.97, min_active_pct: 0.35, heat_ema: 0.6,
    },
  }),
  tpl({
    id: 'factory-balanced', name: 'Balanced', readonly: true,
    config: {
      preset: 'factory:balanced', warmup_pct: 0.25, interval_steps: 200,
      energy_threshold: 0.93, min_active_pct: 0.25, heat_ema: 0.5,
    },
  }),
  tpl({
    id: 'factory-aggressive', name: 'Aggressive', readonly: true,
    config: {
      preset: 'factory:aggressive', warmup_pct: 0.15, interval_steps: 150,
      energy_threshold: 0.85, min_active_pct: 0.15, heat_ema: 0.35,
    },
  }),
];

const BRANCHED = tpl({
  id: 'user-1', name: 'Balanced (custom)', readonly: false,
  branched_from: 'factory-balanced',
  config: { preset: 'factory:balanced' },
});

type Svc = {
  listAdaptivePresets: ReturnType<typeof vi.fn>;
  createAdaptivePreset: ReturnType<typeof vi.fn>;
  updateTemplate: ReturnType<typeof vi.fn>;
  useTemplate: ReturnType<typeof vi.fn>;
};

let svc: Svc;
let toast: { error: ReturnType<typeof vi.fn>; success: ReturnType<typeof vi.fn>; warning: ReturnType<typeof vi.fn> };

function build(opts: {
  presets?: Template[];
  initial?: Record<string, unknown>;
  created?: Template;
  createObservable?: Observable<Template>;
  createError?: unknown;
  updateObservable?: Observable<never>;
  updateError?: unknown;
  useError?: unknown;
  enabled?: boolean;
} = {}) {
  TestBed.resetTestingModule();
  svc = {
    listAdaptivePresets: vi.fn().mockReturnValue(of(opts.presets ?? FACTORY)),
    createAdaptivePreset: vi.fn().mockReturnValue(
      opts.createObservable
        ?? (opts.createError ? throwError(() => opts.createError) : of(opts.created ?? BRANCHED)),
    ),
    updateTemplate: vi.fn().mockImplementation(() =>
      opts.updateObservable
        ?? (opts.updateError ? throwError(() => opts.updateError) : of({})),
    ),
    useTemplate: vi.fn().mockReturnValue(
      opts.useError ? throwError(() => opts.useError) : of({ status: 'recorded' }),
    ),
  };
  toast = { error: vi.fn(), success: vi.fn(), warning: vi.fn() };

  TestBed.configureTestingModule({
    imports: [AdaptiveTargetingCardComponent],
    providers: [
      { provide: TemplateService, useValue: svc },
      { provide: ToastService, useValue: toast },
    ],
  });

  const control = new FormControl<Record<string, unknown>>(opts.initial ?? {}, { nonNullable: true });
  const fixture = TestBed.createComponent(AdaptiveTargetingCardComponent);
  fixture.componentRef.setInput('control', control);
  if (opts.enabled !== undefined) fixture.componentRef.setInput('enabled', opts.enabled);
  fixture.detectChanges();
  return { fixture, control };
}

type Fixture = ReturnType<typeof build>['fixture'];

function el(fixture: Fixture, id: string): HTMLInputElement | HTMLSelectElement {
  const de = byTestId(fixture, id);
  if (!de) throw new Error(`no element with data-testid="${id}"`);
  return de.nativeElement as HTMLInputElement | HTMLSelectElement;
}

/** Edit a knob widget the way a user does: change its value, fire `change`. */
function setKnob(fixture: Fixture, id: string, value: string | number): void {
  const node = el(fixture, id);
  node.value = String(value);
  node.dispatchEvent(new Event('change'));
  fixture.detectChanges();
}

function toggle(fixture: Fixture, id: string): void {
  const node = el(fixture, id) as HTMLInputElement;
  node.checked = !node.checked;
  node.dispatchEvent(new Event('change'));
  fixture.detectChanges();
}

/**
 * Let the debounced preset-save window elapse. The card matches the training
 * form's autosave idiom (`template-autosave.service.ts`, 1200 ms), so nothing
 * reaches the TemplateService until this runs.
 *
 * House gotcha: `'Date'` MUST be in `toFake` or RxJS `debounceTime` reschedules
 * forever, and `useRealTimers()` in `afterEach` is mandatory.
 */
beforeEach(() => {
  vi.useFakeTimers({
    toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'],
  });
});
afterEach(() => {
  vi.useRealTimers();
});

function flushSave(fixture: Fixture): void {
  vi.advanceTimersByTime(PRESET_SAVE_DEBOUNCE_MS + 50);
  fixture.detectChanges();
}

describe('AdaptiveTargetingCard — materialized value (D1)', () => {
  it('materializes the full knob dict into an empty control on load', () => {
    const { control } = build();
    // Every knob the backend validates must be present and correctly TYPED —
    // the schema declares a bare `dict`, so coerceConfigNumbers cannot rescue
    // a stringified number on submit.
    expect(control.value).toEqual({
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
    });
  });

  it('keeps an already-materialized value (no spurious rewrite) and re-selects its preset', () => {
    const initial = {
      preset: 'factory:aggressive', warmup_pct: 0.15, interval_steps: 150,
      energy_threshold: 0.85, min_active_pct: 0.15, heat_ema: 0.35,
      reactivation: false, probe_every: 5, probe_steps: 30,
      action: 'freeze', rebuild_min_shrink_pct: 25,
    };
    const { fixture, control } = build({ initial: { ...initial } });
    expect(control.value).toEqual(initial);
    expect((el(fixture, 'adaptive-preset-select') as HTMLSelectElement).value)
      .toBe('factory:aggressive');
  });

  it('selecting a factory preset materializes ITS knobs into the control', () => {
    const { fixture, control } = build();
    setKnob(fixture, 'adaptive-preset-select', 'factory:aggressive');

    expect(control.value['interval_steps']).toBe(150);
    expect(control.value['warmup_pct']).toBe(0.15);
    expect(control.value['energy_threshold']).toBe(0.85);
    expect(control.value['preset']).toBe('factory:aggressive');
    // The knobs a factory preset does NOT carry keep their defaults, so the
    // dict stays complete.
    expect(control.value['probe_steps']).toBe(30);
    expect(control.value['action']).toBe('freeze');
    // Selecting is not editing: no template write, not even a deferred one.
    flushSave(fixture);
    expect(svc.createAdaptivePreset).not.toHaveBeenCalled();
    expect(svc.updateTemplate).not.toHaveBeenCalled();
  });

  it('reads round-tripped boolean flags through the shared predicate', () => {
    // A config that has been through a template row / the JSON editor can carry
    // "true"/"false" strings. A strict `=== true` would silently drop the user's
    // re-activation setting; a plain truthiness test would turn "false" ON.
    expect(build({ initial: { reactivation: 'true' } }).control.value['reactivation']).toBe(true);
    expect(build({ initial: { reactivation: 'false' } }).control.value['reactivation']).toBe(false);
    expect(build({ initial: { reactivation: true } }).control.value['reactivation']).toBe(true);
    expect(build({ initial: {} }).control.value['reactivation']).toBe(false);
  });

  it('lists factory presets first, user presets nested under their parent', () => {
    const { fixture } = build({ presets: [...FACTORY, BRANCHED] });
    const opts = allByTestId(fixture, 'adaptive-preset-option')
      .map(d => (d.nativeElement as HTMLOptionElement));
    expect(opts.map(o => o.value)).toEqual([
      'factory:conservative', 'factory:balanced', 'user-1', 'factory:aggressive',
    ]);
    // The branched child is visually nested under Balanced.
    expect(opts[2].textContent).toContain('Balanced (custom)');
    expect(opts[2].textContent).toContain('↳');
    expect(opts[1].textContent).not.toContain('↳');
  });
});

describe('AdaptiveTargetingCard — preset persistence (D2)', () => {
  it('editing a knob on a factory preset auto-branches a user preset, then autosaves into it', () => {
    const { fixture, control } = build();
    setKnob(fixture, 'adaptive-knob-interval_steps', 120);
    // Nothing is written until the debounce window closes.
    expect(svc.createAdaptivePreset).not.toHaveBeenCalled();
    flushSave(fixture);

    expect(svc.createAdaptivePreset).toHaveBeenCalledTimes(1);
    const payload = svc.createAdaptivePreset.mock.calls[0][0];
    expect(payload.name).toBe('Balanced (custom)');
    expect(payload.branched_from).toBe('factory-balanced');
    expect(payload.config.interval_steps).toBe(120);

    // Selection follows the branch — provenance now points at the user preset.
    expect(control.value['preset']).toBe('user-1');
    expect(control.value['interval_steps']).toBe(120);

    // …and the STORED row is corrected to match: the POST body could only carry
    // the parent ref, because the new id did not exist yet.
    expect(svc.updateTemplate).toHaveBeenCalledWith(
      'adaptive', 'user-1', { config: expect.objectContaining({ preset: 'user-1', interval_steps: 120 }) },
    );

    // A second edit autosaves into the branched preset instead of branching again.
    setKnob(fixture, 'adaptive-knob-interval_steps', 130);
    flushSave(fixture);
    expect(svc.createAdaptivePreset).toHaveBeenCalledTimes(1);
    expect(svc.updateTemplate).toHaveBeenCalledWith(
      'adaptive', 'user-1', { config: expect.objectContaining({ interval_steps: 130 }) },
    );
  });

  it('auto-branches exactly ONCE even when edits land in separate save windows', () => {
    // Each flush is a separate reach into the write path — the debounce is not
    // what keeps this to one branch, the selection move is.
    const { fixture } = build();
    setKnob(fixture, 'adaptive-knob-interval_steps', 120);
    flushSave(fixture);
    setKnob(fixture, 'adaptive-knob-probe_every', 7);
    flushSave(fixture);
    setKnob(fixture, 'adaptive-knob-interval_steps', 140);
    flushSave(fixture);
    expect(svc.createAdaptivePreset).toHaveBeenCalledTimes(1);
  });

  it('collapses a burst of edits into a single save', () => {
    // A slider drag emits many commits; they must not each hit the API.
    const { fixture } = build({ presets: [...FACTORY, BRANCHED], initial: { preset: 'user-1' } });
    setKnob(fixture, 'adaptive-knob-interval_steps', 120);
    setKnob(fixture, 'adaptive-knob-interval_steps', 130);
    setKnob(fixture, 'adaptive-knob-interval_steps', 140);
    flushSave(fixture);
    expect(svc.updateTemplate).toHaveBeenCalledTimes(1);
    expect(svc.updateTemplate.mock.calls[0][2].config.interval_steps).toBe(140);
  });

  it('does NOT start a second branch while the first POST is still in flight', () => {
    // The synchronous `of()` used elsewhere hides this: with a real (pending)
    // request, a second edit must not POST a duplicate preset.
    const pending = new Subject<Template>();
    const { fixture, control } = build({ createObservable: pending });

    setKnob(fixture, 'adaptive-knob-interval_steps', 120);
    flushSave(fixture);
    setKnob(fixture, 'adaptive-knob-interval_steps', 130);
    flushSave(fixture);
    expect(svc.createAdaptivePreset).toHaveBeenCalledTimes(1);
    expect(svc.updateTemplate).not.toHaveBeenCalled();

    pending.next(BRANCHED);
    pending.complete();

    // The edit made mid-flight is not lost — it lands in the branched preset.
    expect(svc.createAdaptivePreset).toHaveBeenCalledTimes(1);
    expect(svc.updateTemplate).toHaveBeenCalledWith(
      'adaptive', 'user-1', { config: expect.objectContaining({ interval_steps: 130 }) },
    );
    expect(control.value['preset']).toBe('user-1');
    expect(control.value['interval_steps']).toBe(130);
  });

  it('editing a USER preset autosaves without creating a new one', () => {
    const { fixture, control } = build({
      presets: [...FACTORY, BRANCHED],
      initial: { preset: 'user-1' },
    });
    expect(control.value['preset']).toBe('user-1');

    setKnob(fixture, 'adaptive-knob-interval_steps', 111);
    flushSave(fixture);

    expect(svc.createAdaptivePreset).not.toHaveBeenCalled();
    expect(svc.updateTemplate).toHaveBeenCalledTimes(1);
    expect(svc.updateTemplate.mock.calls[0][0]).toBe('adaptive');
    expect(svc.updateTemplate.mock.calls[0][1]).toBe('user-1');
    expect(svc.updateTemplate.mock.calls[0][2].config.interval_steps).toBe(111);
  });

  it('a later save supersedes an in-flight one (no out-of-order preset row)', () => {
    // Every PUT carries the FULL dict, so an older response landing last would
    // persist a stale knob set. switchMap must tear the first request down.
    let subscribes = 0;
    let teardowns = 0;
    const neverSettles = new Observable<never>(() => {
      subscribes++;
      return () => { teardowns++; };
    });
    const { fixture } = build({
      presets: [...FACTORY, BRANCHED],
      initial: { preset: 'user-1' },
      updateObservable: neverSettles,
    });

    setKnob(fixture, 'adaptive-knob-interval_steps', 111);
    flushSave(fixture);
    expect(subscribes).toBe(1);
    expect(teardowns).toBe(0);

    setKnob(fixture, 'adaptive-knob-interval_steps', 222);
    flushSave(fixture);
    expect(subscribes).toBe(2);
    expect(teardowns).toBe(1); // the stale PUT was cancelled, not raced
    expect(svc.updateTemplate.mock.calls[1][2].config.interval_steps).toBe(222);
  });

  it('never branches (or saves) for a no-op edit', () => {
    const { fixture, control } = build();
    const before = { ...control.value };

    // Re-entering the value that is already set changes nothing.
    setKnob(fixture, 'adaptive-knob-interval_steps', 200);
    flushSave(fixture);

    expect(svc.createAdaptivePreset).not.toHaveBeenCalled();
    expect(svc.updateTemplate).not.toHaveBeenCalled();
    expect(control.value).toEqual(before);
  });

  it('treats an out-of-range entry that clamps back to the current value as a no-op', () => {
    // The reported shape: typing 5 into interval_steps when it is already at
    // the floor of 10 clamps straight back — and used to branch a preset for a
    // change that changed nothing.
    const { fixture } = build();
    setKnob(fixture, 'adaptive-knob-interval_steps', 10);
    flushSave(fixture);
    expect(svc.createAdaptivePreset).toHaveBeenCalledTimes(1);
    svc.updateTemplate.mockClear();

    setKnob(fixture, 'adaptive-knob-interval_steps', 5); // → clamps to 10 = current
    flushSave(fixture);

    expect(svc.createAdaptivePreset).toHaveBeenCalledTimes(1);
    expect(svc.updateTemplate).not.toHaveBeenCalled();
  });

  it('surfaces the backend rejection message instead of failing silently', () => {
    const { fixture } = build({
      createError: { error: { detail: 'Invalid adaptive preset config: probe_steps must be < interval_steps' } },
    });
    setKnob(fixture, 'adaptive-knob-interval_steps', 120);
    flushSave(fixture);
    expect(toast.error).toHaveBeenCalledWith(expect.stringContaining('probe_steps must be < interval_steps'));
  });

  it('surfaces a failed autosave PUT too', () => {
    const { fixture } = build({
      presets: [...FACTORY, BRANCHED],
      initial: { preset: 'user-1' },
      updateError: { error: { detail: 'preset gone' } },
    });
    setKnob(fixture, 'adaptive-knob-interval_steps', 111);
    flushSave(fixture);
    expect(toast.error).toHaveBeenCalledWith(expect.stringContaining('preset gone'));

    // …and the autosave pipe survives its own error — a later edit still saves.
    setKnob(fixture, 'adaptive-knob-interval_steps', 222);
    flushSave(fixture);
    expect(svc.updateTemplate).toHaveBeenCalledTimes(2);
  });

  it('a failed branch does not wedge the card — the next edit retries', () => {
    const { fixture } = build({ createError: { error: { detail: 'nope' } } });
    setKnob(fixture, 'adaptive-knob-interval_steps', 120);
    flushSave(fixture);
    expect(svc.createAdaptivePreset).toHaveBeenCalledTimes(1);
    setKnob(fixture, 'adaptive-knob-interval_steps', 130);
    flushSave(fixture);
    expect(svc.createAdaptivePreset).toHaveBeenCalledTimes(2);
  });
});

describe('AdaptiveTargetingCard — cross-field rules (D3)', () => {
  it('reactivation sub-knobs are disabled unless reactivation is on', () => {
    const { fixture, control } = build();
    expect(control.value['reactivation']).toBe(false);
    expect((el(fixture, 'adaptive-knob-probe_every') as HTMLInputElement).disabled).toBe(true);
    expect((el(fixture, 'adaptive-knob-probe_steps') as HTMLInputElement).disabled).toBe(true);

    toggle(fixture, 'adaptive-knob-reactivation');

    expect(control.value['reactivation']).toBe(true);
    expect((el(fixture, 'adaptive-knob-probe_every') as HTMLInputElement).disabled).toBe(false);
    expect((el(fixture, 'adaptive-knob-probe_steps') as HTMLInputElement).disabled).toBe(false);
  });

  it('choosing action=rebuild forces reactivation off IN THE SAME value and disables its toggle', () => {
    const { fixture, control } = build();
    toggle(fixture, 'adaptive-knob-reactivation');
    expect(control.value['reactivation']).toBe(true);

    setKnob(fixture, 'adaptive-knob-action', 'rebuild');

    // The committed value itself must already be legal — never a two-step
    // transition through a state the backend rejects.
    expect(control.value['action']).toBe('rebuild');
    expect(control.value['reactivation']).toBe(false);
    expect((el(fixture, 'adaptive-knob-reactivation') as HTMLInputElement).disabled).toBe(true);
    expect((el(fixture, 'adaptive-knob-reactivation') as HTMLInputElement).checked).toBe(false);
  });

  it('switching back to freeze re-enables the reactivation toggle', () => {
    const { fixture } = build();
    setKnob(fixture, 'adaptive-knob-action', 'rebuild');
    setKnob(fixture, 'adaptive-knob-action', 'freeze');
    expect((el(fixture, 'adaptive-knob-reactivation') as HTMLInputElement).disabled).toBe(false);
  });

  it('probe_steps can never reach interval_steps', () => {
    const { fixture, control } = build();
    toggle(fixture, 'adaptive-knob-reactivation');

    // Typing past the ceiling clamps to interval_steps - 1.
    setKnob(fixture, 'adaptive-knob-probe_steps', 900);
    expect(control.value['probe_steps']).toBe(199);
    expect(control.value['interval_steps']).toBe(200);

    // Lowering interval_steps under the current probe_steps clamps it too, in
    // the SAME commit (never an intermediate invalid pair).
    setKnob(fixture, 'adaptive-knob-interval_steps', 50);
    expect(control.value['interval_steps']).toBe(50);
    expect(control.value['probe_steps']).toBe(49);

    // Floor holds as well.
    setKnob(fixture, 'adaptive-knob-probe_steps', 0);
    expect(control.value['probe_steps']).toBe(1);
  });

  it('clamps every knob into its backend-validated range', () => {
    const { fixture, control } = build();
    setKnob(fixture, 'adaptive-knob-interval_steps', 1);      // ge=10
    expect(control.value['interval_steps']).toBe(10);
    toggle(fixture, 'adaptive-knob-reactivation');
    setKnob(fixture, 'adaptive-knob-probe_every', 1);         // ge=2
    expect(control.value['probe_every']).toBe(2);
    setKnob(fixture, 'adaptive-knob-warmup_pct', 1.5);        // lt=1
    expect(control.value['warmup_pct']).toBeLessThan(1);
    setKnob(fixture, 'adaptive-knob-energy_threshold', 2);    // le=1
    expect(control.value['energy_threshold']).toBe(1);
    setKnob(fixture, 'adaptive-knob-min_active_pct', 0);      // gt=0
    expect(control.value['min_active_pct'] as number).toBeGreaterThan(0);
    setKnob(fixture, 'adaptive-knob-heat_ema', 1);            // lt=1
    expect(control.value['heat_ema']).toBeLessThan(1);
  });

  it('rebuild_min_shrink_pct only appears (and only commits) in rebuild mode', () => {
    const { fixture, control } = build();
    expect(byTestId(fixture, 'adaptive-knob-rebuild_min_shrink_pct')).toBeNull();

    setKnob(fixture, 'adaptive-knob-action', 'rebuild');
    expect(byTestId(fixture, 'adaptive-knob-rebuild_min_shrink_pct')).not.toBeNull();
    setKnob(fixture, 'adaptive-knob-rebuild_min_shrink_pct', 200); // le=100
    expect(control.value['rebuild_min_shrink_pct']).toBe(100);
  });
});

describe('AdaptiveTargetingCard — feature-off + help wiring', () => {
  it('hides the knob body when the feature toggle is off but still materializes the value', () => {
    const { fixture, control } = build({ enabled: false });
    expect(byTestId(fixture, 'adaptive-preset-select')).toBeNull();
    expect(byTestId(fixture, 'adaptive-disabled-hint')).not.toBeNull();
    expect(control.value['interval_steps']).toBe(200);
  });

  it('emits namespaced help keys the parent resolves against config_help.json', () => {
    const { fixture } = build({
      // configHelp is an input so the card reuses the parent's already-loaded map.
    });
    fixture.componentRef.setInput('configHelp', {
      'adaptive_targeting.interval_steps': { tip: 'How often', detail: 'd' },
    });
    fixture.detectChanges();

    const seen: string[] = [];
    fixture.componentInstance.helpRequested.subscribe((k: string) => seen.push(k));
    (el(fixture, 'adaptive-help-interval_steps') as HTMLElement).click();
    expect(seen).toEqual(['adaptive_targeting.interval_steps']);
  });

  it('surfaces a failed preset list instead of silently rendering an empty dropdown', () => {
    TestBed.resetTestingModule();
    svc = {
      listAdaptivePresets: vi.fn().mockReturnValue(throwError(() => new Error('boom'))),
      createAdaptivePreset: vi.fn(),
      updateTemplate: vi.fn(),
      useTemplate: vi.fn(),
    };
    toast = { error: vi.fn(), success: vi.fn(), warning: vi.fn() };
    TestBed.configureTestingModule({
      imports: [AdaptiveTargetingCardComponent],
      providers: [
        { provide: TemplateService, useValue: svc },
        { provide: ToastService, useValue: toast },
      ],
    });
    const control = new FormControl<Record<string, unknown>>({}, { nonNullable: true });
    const fixture = TestBed.createComponent(AdaptiveTargetingCardComponent);
    fixture.componentRef.setInput('control', control);
    fixture.detectChanges();

    expect(toast.error).toHaveBeenCalled();
    // The control is still a complete, submittable dict.
    expect(control.value['interval_steps']).toBe(200);
  });
});

describe('AdaptiveTargetingCard — preset usage counter', () => {
  it('records a use against the row the user picked', () => {
    const { fixture } = build();
    expect(svc.useTemplate).not.toHaveBeenCalled();  // not on hydration

    setKnob(fixture, 'adaptive-preset-select', 'factory:aggressive');

    // The ROW id, not the `factory:*` provenance ref — /use is keyed on the
    // template's primary key like every other domain.
    expect(svc.useTemplate).toHaveBeenCalledWith('adaptive', 'factory-aggressive');
  });

  it('does not record a use when the same preset is re-selected', () => {
    const { fixture } = build();
    setKnob(fixture, 'adaptive-preset-select', 'factory:aggressive');
    expect(svc.useTemplate).toHaveBeenCalledTimes(1);

    setKnob(fixture, 'adaptive-preset-select', 'factory:aggressive');
    expect(svc.useTemplate).toHaveBeenCalledTimes(1);
  });

  it('does not record a use for a selection that resolves to no row', () => {
    // An empty list is the real shape of this: the dropdown has no options, so
    // a change event carries a ref no row can answer for. Recording against it
    // would POST /templates/adaptive//use.
    const { fixture } = build({ presets: [] });
    setKnob(fixture, 'adaptive-preset-select', 'factory:balanced');
    expect(svc.useTemplate).not.toHaveBeenCalled();
  });

  it('keeps the selection when the counter call fails — it is not the user\'s work', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { fixture, control } = build({ useError: new Error('offline') });

    setKnob(fixture, 'adaptive-preset-select', 'factory:aggressive');

    expect(control.value['preset']).toBe('factory:aggressive');
    expect(control.value['interval_steps']).toBe(150);  // the knobs still applied
    expect(toast.error).not.toHaveBeenCalled();
    expect(warn).toHaveBeenCalled();  // surfaced, not swallowed
    warn.mockRestore();
  });
});
