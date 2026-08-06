import { TestBed } from '@angular/core/testing';
import { FormControl, FormGroup } from '@angular/forms';
import { of } from 'rxjs';

import { TrainingDynamicConfigComponent } from './training-dynamic-config';
import { DatasetService } from '../../../services/dataset';
import { DatasetStore } from '../../../state/dataset.store';
import { ToastService } from '../../../services/toast';
import { SystemService } from '../../../services/system.service';
import { JobService } from '../../../services/job';
import { ConfigHelpService } from '../../../services/config-help.service';
import { ModelService } from '../../../services/model.service';
import { RegistryStore } from '../../../state/registry.store';
import { ModelCapabilitiesService } from '../../../services/model-capabilities.service';
import type { SchemaNode } from '../schema-node';
import type { TrainingConfig } from '../../../services/job';

/**
 * Adaptive-targeting wiring in the training form (Task 10, decision D4).
 *
 * `adaptive_targeting_config` is an OBJECT-typed schema field rendered by the
 * adaptive-targeting card, so — exactly like `block_swap_config` — it must be
 * built as a plain FormControl holding a dict rather than the empty FormGroup
 * placeholder every other object field gets.
 *
 * D4 is the load-bearing part: with the feature toggle OFF the submitted
 * payload must carry NO `adaptive_targeting_config` at all, so a run with the
 * feature off is byte-identical to one from before the feature existed. Its
 * `depends_on` is a BOOLEAN dependency, and `shouldHideField` deliberately
 * never hides those ("show but disable") — so the generic depends_on strip does
 * NOT cover this field and an explicit strip is required.
 */
const SCHEMA: SchemaNode = {
  type: 'object',
  properties: {
    definition_id: { type: 'string', default: '' },
    model_family: { type: 'string', default: '' },
    max_train_steps: { type: 'integer', default: 1000 },
    adaptive_targeting: { type: 'boolean', default: false, group: 'ENGINE' },
    adaptive_targeting_config: {
      type: 'object',
      group: 'ENGINE',
      ui_type: 'adaptive_targeting',
      depends_on: 'adaptive_targeting',
    },
    block_swap_config: {
      type: 'object',
      group: 'ENGINE',
      ui_type: 'block_swap_sliders',
    },
  },
} as unknown as SchemaNode;

const KNOBS = {
  preset: 'factory:balanced', warmup_pct: 0.25, interval_steps: 200,
  energy_threshold: 0.93, min_active_pct: 0.25, heat_ema: 0.5,
  reactivation: false, probe_every: 5, probe_steps: 30,
  action: 'freeze', rebuild_min_shrink_pct: 25,
};

function build() {
  TestBed.resetTestingModule();
  TestBed.configureTestingModule({
    imports: [TrainingDynamicConfigComponent],
    providers: [
      { provide: DatasetService, useValue: { listDatasets: () => of([]) } },
      { provide: DatasetStore, useValue: { entities: () => [] } },
      { provide: ToastService, useValue: { error: () => {}, success: () => {}, warning: () => {} } },
      { provide: SystemService, useValue: {} },
      { provide: JobService, useValue: { estimate: () => of(null) } },
      { provide: ConfigHelpService, useValue: { getConfigHelp: () => of({}) } },
      { provide: ModelService, useValue: { getGlobalSettings: () => of({ default_model_path: '' }) } },
      { provide: RegistryStore, useValue: {} },
      { provide: ModelCapabilitiesService, useValue: { getCapabilities: () => of(null) } },
    ],
  });
  TestBed.overrideComponent(TrainingDynamicConfigComponent, { set: { template: '', imports: [] } });
  const fixture = TestBed.createComponent(TrainingDynamicConfigComponent);
  fixture.componentRef.setInput('schema', SCHEMA);
  fixture.detectChanges(); // runs the effect → buildForm()
  const comp = fixture.componentInstance;
  const emitted: TrainingConfig[] = [];
  comp.configSubmitted.subscribe(c => emitted.push(c));
  return { comp, emitted };
}

describe('TrainingDynamicConfig — adaptive_targeting_config control wiring', () => {
  it('builds a plain FormControl holding a dict (not the FormGroup placeholder)', () => {
    const { comp } = build();
    const ctrl = comp.form.get('adaptive_targeting_config');
    expect(ctrl).toBeInstanceOf(FormControl);
    expect(ctrl).not.toBeInstanceOf(FormGroup);
    expect(ctrl!.value).toEqual({});
  });

  it('exposes it through getObjectControl for the card binding', () => {
    const { comp } = build();
    const ctrl = comp.getObjectControl('adaptive_targeting_config');
    expect(ctrl).toBe(comp.form.get('adaptive_targeting_config'));
    ctrl.setValue({ ...KNOBS });
    expect(comp.form.get('adaptive_targeting_config')!.value).toEqual(KNOBS);
  });

  it('resets an object control to a DICT, never the scalar "" fallback', () => {
    const { comp } = build();
    comp.form.get('adaptive_targeting_config')!.setValue({ ...KNOBS });
    comp.form.get('block_swap_config')!.setValue({ double_blocks: 50 });

    comp.resetFormToDefaults();

    expect(comp.form.get('adaptive_targeting_config')!.value).toEqual({});
    expect(comp.form.get('block_swap_config')!.value).toEqual({});
  });
});

describe('TrainingDynamicConfig — feature-off submit strip (D4)', () => {
  it('omits adaptive_targeting_config entirely when the feature is off', () => {
    const { comp, emitted } = build();
    // The card materializes its knobs regardless of the toggle, so a stale dict
    // IS present in the form when the feature is off — it must not be submitted.
    comp.form.get('adaptive_targeting_config')!.setValue({ ...KNOBS });
    comp.form.get('adaptive_targeting')!.setValue(false);

    comp.onSubmit();

    expect(emitted.length).toBe(1);
    expect('adaptive_targeting_config' in emitted[0]).toBe(false);
    expect(emitted[0]['adaptive_targeting']).toBe(false);
  });

  it('submits the materialized knob dict verbatim when the feature is on', () => {
    const { comp, emitted } = build();
    comp.form.get('adaptive_targeting')!.setValue(true);
    comp.form.get('adaptive_targeting_config')!.setValue({ ...KNOBS });

    comp.onSubmit();

    expect(emitted[0]['adaptive_targeting']).toBe(true);
    // Self-contained: every knob rides along, no template reference to resolve.
    expect(emitted[0]['adaptive_targeting_config']).toEqual(KNOBS);
  });

  it('strips it for a truthy-string toggle too (never leaks on a stringified flag)', () => {
    const { comp, emitted } = build();
    comp.form.get('adaptive_targeting_config')!.setValue({ ...KNOBS });
    // A select/checkbox round-trip can hand back "false"; that is still OFF.
    comp.form.get('adaptive_targeting')!.setValue('false');

    comp.onSubmit();
    expect('adaptive_targeting_config' in emitted[0]).toBe(false);
  });

  it('leaves block_swap_config alone (the strip is scoped to this one field)', () => {
    const { comp, emitted } = build();
    comp.form.get('block_swap_config')!.setValue({ double_blocks: 50 });
    comp.form.get('adaptive_targeting')!.setValue(false);

    comp.onSubmit();
    expect(emitted[0]['block_swap_config']).toEqual({ double_blocks: 50 });
  });
});

describe('TrainingDynamicConfig — namespaced help keys', () => {
  it('titles a dotted card help key from its leaf segment', () => {
    const { comp } = build();
    expect(comp.getHelpTitle('adaptive_targeting.interval_steps')).toBe('Interval Steps');
    // Plain keys are unchanged.
    expect(comp.getHelpTitle('adaptive_targeting')).toBe('Adaptive Targeting');
    expect(comp.getHelpTitle('max_train_steps')).toBe('Max Train Steps');
  });
});
