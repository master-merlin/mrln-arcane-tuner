import { TestBed } from '@angular/core/testing';
import { FormArray, FormControl } from '@angular/forms';
import { of } from 'rxjs';

import { TrainingDynamicConfigComponent } from './training-dynamic-config';
import { TrainingTemplateSelectorComponent } from '../training-template-selector/training-template-selector';
import { VramBudgetCardComponent } from '../vram-budget-card/vram-budget-card';
import { AdvancedVramCardComponent } from '../advanced-vram-card/advanced-vram-card';
import { TargetLayersCardComponent } from '../target-layers-card/target-layers-card';
import { DynamicFormGroupComponent } from '../dynamic-form-group/dynamic-form-group';
import { DatasetService } from '../../../services/dataset';
import { DatasetStore } from '../../../state/dataset.store';
import { ToastService } from '../../../services/toast';
import { SystemService } from '../../../services/system.service';
import { JobService } from '../../../services/job';
import { ConfigHelpService } from '../../../services/config-help.service';
import { ModelService } from '../../../services/model.service';
import { RegistryStore } from '../../../state/registry.store';
import { ModelCapabilitiesService } from '../../../services/model-capabilities.service';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { TemplateService } from '../../../services/template.service';
import { ProjectService } from '../../../services/project.service';
import { OverlayStore } from '../../../state/overlay.store';
import { FilesystemService } from '../../../services/filesystem.service';
import type { SchemaNode } from '../schema-node';
import type {
  ModelCapabilities,
  FieldVisibility,
} from '../../../services/model-capabilities.service';
import type { TrainingConfig } from '../../../services/job';

/**
 * W4-3 — submit-strip for NESTED (per-item) capability-gated fields.
 *
 * The top-level strip in onSubmit already drops family-unsupported run-level
 * fields (proven by the temporal spec). Per-dataset fields were capability-
 * blind: e.g. `masking_enabled` on a paired edit model (gated by
 * archetypes.py `supports_masking_variants`) could still ride to the backend
 * with a stale `true` from the form model. The strip must consult the SAME
 * `field_visibility` descriptor for nested array items too.
 */
const SCHEMA: SchemaNode = {
  type: 'object',
  properties: {
    definition_id: { type: 'string', default: '' },
    model_family: { type: 'string', default: 'std' },
    num_frames: { type: 'integer', default: 25, group: 'VIDEO' },
    datasets: {
      type: 'array',
      group: 'CONCEPTS',
      items: {
        type: 'object',
        properties: {
          dataset_name: { type: 'string', default: '' },
          masking_enabled: { type: 'boolean', default: false, inline_group: 'masking_toggles' },
          use_captions: { type: 'boolean', default: true, inline_group: 'caption_toggles' },
        },
      },
    },
  },
} as unknown as SchemaNode;

function caps(overrides: Record<string, FieldVisibility>): ModelCapabilities {
  return {
    enriched: true,
    block_topology: [],
    lora_targetable_modules: [],
    trainable_layers: [],
    archetype: 'edit',
    capabilities: {} as ModelCapabilities['capabilities'],
    field_visibility: overrides,
    defaults: {},
  };
}

function build() {
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
      { provide: FilesystemService, useValue: {} },
      { provide: RegistryStore, useValue: {} },
      { provide: ModelCapabilitiesService, useValue: { getCapabilities: () => of(null) } },
      { provide: RuntimeConfigService, useValue: { apiUrl: '/api', mediaBaseUrl: '/media' } },
      { provide: TemplateService, useValue: { listTrainingTemplates: () => of([]) } },
      { provide: ProjectService, useValue: { getPreferences: () => of(null) } },
      { provide: OverlayStore, useValue: { openModal: () => {}, closeModal: () => {}, topModal: () => null } },
    ],
  });
  const stub = { template: '', imports: [] };
  TestBed.overrideComponent(TrainingTemplateSelectorComponent, { set: stub });
  TestBed.overrideComponent(VramBudgetCardComponent, { set: stub });
  TestBed.overrideComponent(AdvancedVramCardComponent, { set: stub });
  TestBed.overrideComponent(TargetLayersCardComponent, { set: stub });
  TestBed.overrideComponent(DynamicFormGroupComponent, { set: stub });

  const fixture = TestBed.createComponent(TrainingDynamicConfigComponent);
  fixture.componentRef.setInput('schema', SCHEMA);
  fixture.detectChanges(); // runs the effect → buildForm() (auto-adds one datasets row)
  return fixture;
}

function setCaps(fixture: ReturnType<typeof build>, value: ModelCapabilities | null): void {
  (fixture.componentInstance as unknown as {
    capabilities: { set(v: ModelCapabilities | null): void };
  }).capabilities.set(value);
}

/** Set the (auto-added) first dataset row's fields for the submit assertions. */
function seedRow(fixture: ReturnType<typeof build>): void {
  const arr = fixture.componentInstance.form.get('datasets') as FormArray;
  const row = arr.at(0);
  row.get('dataset_name')?.setValue('my_set');
  row.get('masking_enabled')?.setValue(true);
  row.get('use_captions')?.setValue(true);
}

function submitAndCapture(fixture: ReturnType<typeof build>): TrainingConfig {
  let emitted: TrainingConfig | undefined;
  fixture.componentInstance.configSubmitted.subscribe((c: TrainingConfig) => (emitted = c));
  fixture.componentInstance.onSubmit();
  expect(emitted).toBeDefined(); // form must be valid, else nothing emits
  return emitted as TrainingConfig;
}

describe('TrainingDynamicConfig — nested capability submit-strip', () => {
  it('STRIPS masking_enabled from the dataset item for an edit model', () => {
    const fixture = build();
    seedRow(fixture);
    setCaps(fixture, caps({ masking_enabled: { supported: false, reason: 'paired edit' } }));

    const payload = submitAndCapture(fixture) as unknown as {
      datasets: Array<Record<string, unknown>>;
    };
    const row = payload.datasets[0];
    expect('masking_enabled' in row).toBe(false); // stale value never reaches backend
    expect(row['dataset_name']).toBe('my_set'); // unrelated fields survive
    expect(row['use_captions']).toBe(true);
  });

  it('KEEPS masking_enabled when the family supports it', () => {
    const fixture = build();
    seedRow(fixture);
    setCaps(fixture, caps({ masking_enabled: { supported: true } }));

    const payload = submitAndCapture(fixture) as unknown as {
      datasets: Array<Record<string, unknown>>;
    };
    expect(payload.datasets[0]['masking_enabled']).toBe(true);
  });

  it('KEEPS masking_enabled when there is no descriptor (fail-open)', () => {
    const fixture = build();
    seedRow(fixture);
    setCaps(fixture, null);

    const payload = submitAndCapture(fixture) as unknown as {
      datasets: Array<Record<string, unknown>>;
    };
    expect(payload.datasets[0]['masking_enabled']).toBe(true);
  });

  it('leaves TOP-LEVEL strip behaviour intact alongside the nested strip', () => {
    const fixture = build();
    seedRow(fixture);
    // Gate BOTH a run-level field (num_frames) and the nested one.
    setCaps(fixture, caps({
      num_frames: { supported: false, reason: 'image model' },
      masking_enabled: { supported: false, reason: 'paired edit' },
    }));

    const payload = submitAndCapture(fixture) as unknown as Record<string, unknown> & {
      datasets: Array<Record<string, unknown>>;
    };
    // Top-level gated field stripped (pre-existing behaviour, still works).
    expect('num_frames' in payload).toBe(false);
    // Non-gated top-level field survives.
    expect(payload['model_family']).toBe('std');
    // Nested gated field stripped.
    expect('masking_enabled' in payload.datasets[0]).toBe(false);
  });
});
