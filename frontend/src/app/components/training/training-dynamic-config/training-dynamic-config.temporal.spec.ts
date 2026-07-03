import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { TrainingDynamicConfigComponent } from './training-dynamic-config';
import { DynamicFormFieldComponent } from '../dynamic-form-field/dynamic-form-field';
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

/**
 * Phase-1 temporal-sampling knobs surface in the training config UI, gated by
 * the backend `field_visibility`/`is_video` capability map (mirrored here as
 * {@link videoCaps}/{@link imageCaps}). The four ACTIVE knobs:
 *   - temporal_coverage — enum select (first/tiled/sliding), VIDEO group
 *   - window_overlap / max_windows — VIDEO, only when temporal_coverage=tiled
 *   - frame_stride — VIDEO group
 * All four are hidden on image-only models (supported:false → shouldHideField).
 *
 * The schema fixture mirrors the backend contract verified by
 * backend/tests/engine/test_video_contract_temporal.py (enum/defaults/group/
 * depends_on). The renderer reads the JSON schema (field definitions) AND the
 * capabilities() signal (field_visibility map) — both must carry the keys.
 */
const SCHEMA: SchemaNode = {
  type: 'object',
  properties: {
    definition_id: { type: 'string', default: '' },
    model_family: { type: 'string', default: '' },
    // num_frames is the run-level is_video gate the component already keys on.
    num_frames: { type: 'integer', default: 25, group: 'VIDEO' },
    temporal_coverage: {
      type: 'string',
      default: 'first',
      enum: ['first', 'tiled', 'sliding'],
      group: 'VIDEO',
    },
    window_overlap: {
      type: 'number',
      default: 0.0,
      group: 'VIDEO',
      depends_on: 'temporal_coverage:tiled',
    },
    max_windows: {
      type: 'integer',
      default: 10,
      group: 'VIDEO',
      depends_on: 'temporal_coverage:tiled',
    },
    frame_stride: { type: 'integer', default: 1, group: 'VIDEO' },
    sliding_max_clip_seconds: {
      type: 'number',
      default: 0.0,
      group: 'VIDEO',
      depends_on: 'temporal_coverage:sliding',
    },
  },
} as unknown as SchemaNode;

/** Keys gated by the is_video rule in the backend's _FIELD_RULES. */
const VIDEO_GATED_KEYS = [
  'num_frames',
  'temporal_coverage',
  'window_overlap',
  'max_windows',
  'frame_stride',
  'sliding_max_clip_seconds',
];

/** Build a `field_visibility` map with the temporal keys at `supported`. */
function fieldVisibility(supported: boolean): Record<string, FieldVisibility> {
  const fv: Record<string, FieldVisibility> = {};
  for (const key of VIDEO_GATED_KEYS) {
    fv[key] = { supported };
  }
  return fv;
}

/** Minimal capabilities descriptor; only field_visibility drives hiding here. */
function caps(supported: boolean): ModelCapabilities {
  return {
    enriched: true,
    block_topology: [],
    lora_targetable_modules: [],
    trainable_layers: [],
    archetype: supported ? 'video' : 'image',
    capabilities: {} as ModelCapabilities['capabilities'],
    field_visibility: fieldVisibility(supported),
    defaults: {},
  };
}

/** Video model: num_frames + temporal keys supported. */
function videoCaps(): ModelCapabilities {
  return caps(true);
}

/** Image-only model: num_frames + temporal keys NOT supported. */
function imageCaps(): ModelCapabilities {
  return caps(false);
}

function build() {
  TestBed.configureTestingModule({
    imports: [TrainingDynamicConfigComponent],
    providers: [
      { provide: DatasetService, useValue: { listDatasets: () => of([]) } },
      { provide: DatasetStore, useValue: { entities: () => [] } },
      { provide: ToastService, useValue: { error: () => {}, success: () => {}, warning: () => {} } },
      { provide: SystemService, useValue: {} },
      { provide: JobService, useValue: { estimate: () => of(null), getConfigHelp: () => of({}) } },
      { provide: ModelService, useValue: { getGlobalSettings: () => of({ default_model_path: '' }) } },
      { provide: FilesystemService, useValue: {} },
      { provide: RegistryStore, useValue: {} },
      { provide: ModelCapabilitiesService, useValue: { getCapabilities: () => of(null) } },
      { provide: RuntimeConfigService, useValue: { apiUrl: '/api', mediaBaseUrl: '/media' } },
      // The (template-stubbed) selector still runs its ngOnInit load; give it
      // real arrays so its filteredTemplates computed doesn't choke on `{}`.
      { provide: TemplateService, useValue: { listTrainingTemplates: () => of([]) } },
      { provide: ProjectService, useValue: { getPreferences: () => of(null) } },
      { provide: OverlayStore, useValue: { openModal: () => {}, closeModal: () => {}, topModal: () => null } },
    ],
  });
  // Keep the REAL component template + the REAL DynamicFormFieldComponent (it
  // owns the config-select-/config-input- data-testids). Stub only the heavy
  // sibling cards so the test does not need their services / async init.
  const stub = { template: '', imports: [] };
  TestBed.overrideComponent(TrainingTemplateSelectorComponent, { set: stub });
  TestBed.overrideComponent(VramBudgetCardComponent, { set: stub });
  TestBed.overrideComponent(AdvancedVramCardComponent, { set: stub });
  TestBed.overrideComponent(TargetLayersCardComponent, { set: stub });
  TestBed.overrideComponent(DynamicFormGroupComponent, { set: stub });

  const fixture = TestBed.createComponent(TrainingDynamicConfigComponent);
  fixture.componentRef.setInput('schema', SCHEMA);
  fixture.detectChanges(); // runs the effect → buildForm()
  return fixture;
}

/** Access the protected capabilities signal without fighting TS visibility. */
function setCaps(fixture: ReturnType<typeof build>, value: ModelCapabilities): void {
  (fixture.componentInstance as unknown as {
    capabilities: { set(v: ModelCapabilities): void };
  }).capabilities.set(value);
}

describe('TrainingDynamicConfig — Phase 1 temporal-sampling fields', () => {
  it('renders temporal_coverage as a select with first/tiled/sliding for video models', () => {
    const fixture = build();
    setCaps(fixture, videoCaps()); // num_frames + temporal keys supported
    fixture.detectChanges();

    const select = fixture.nativeElement.querySelector(
      '[data-testid="config-select-temporal_coverage"]',
    ) as HTMLSelectElement | null;
    expect(select).toBeTruthy();
    const opts = Array.from(select!.options).map((o) => o.value);
    expect(opts).toEqual(expect.arrayContaining(['first', 'tiled', 'sliding']));
  });

  it('shows window_overlap + max_windows only when temporal_coverage = tiled', () => {
    const fixture = build();
    setCaps(fixture, videoCaps());
    fixture.componentInstance.form.get('temporal_coverage')?.setValue('first');
    fixture.detectChanges();
    expect(
      fixture.nativeElement.querySelector('[data-testid="config-input-window_overlap"]'),
    ).toBeNull();

    fixture.componentInstance.form.get('temporal_coverage')?.setValue('tiled');
    fixture.detectChanges();
    expect(
      fixture.nativeElement.querySelector('[data-testid="config-input-window_overlap"]'),
    ).toBeTruthy();
    expect(
      fixture.nativeElement.querySelector('[data-testid="config-input-max_windows"]'),
    ).toBeTruthy();
  });

  it('shows sliding_max_clip_seconds only when temporal_coverage = sliding', () => {
    const fixture = build();
    setCaps(fixture, videoCaps());
    fixture.componentInstance.form.get('temporal_coverage')?.setValue('tiled');
    fixture.detectChanges();
    expect(
      fixture.nativeElement.querySelector('[data-testid="config-input-sliding_max_clip_seconds"]'),
    ).toBeNull();

    fixture.componentInstance.form.get('temporal_coverage')?.setValue('sliding');
    fixture.detectChanges();
    expect(
      fixture.nativeElement.querySelector('[data-testid="config-input-sliding_max_clip_seconds"]'),
    ).toBeTruthy();
  });

  it('hides sliding_max_clip_seconds for image-only models even under sliding', () => {
    const fixture = build();
    setCaps(fixture, imageCaps());
    fixture.componentInstance.form.get('temporal_coverage')?.setValue('sliding');
    fixture.detectChanges();
    expect(
      fixture.nativeElement.querySelector('[data-testid="config-input-sliding_max_clip_seconds"]'),
    ).toBeNull();
  });

  it('hides all temporal fields for image-only models', () => {
    const fixture = build();
    setCaps(fixture, imageCaps()); // num_frames + temporal keys NOT supported
    fixture.detectChanges();

    expect(
      fixture.nativeElement.querySelector('[data-testid="config-select-temporal_coverage"]'),
    ).toBeNull();
    expect(
      fixture.nativeElement.querySelector('[data-testid="config-input-frame_stride"]'),
    ).toBeNull();
    expect(
      fixture.nativeElement.querySelector('[data-testid="config-input-window_overlap"]'),
    ).toBeNull();
    expect(
      fixture.nativeElement.querySelector('[data-testid="config-input-max_windows"]'),
    ).toBeNull();
  });

  it('renders frame_stride for video models', () => {
    const fixture = build();
    setCaps(fixture, videoCaps());
    fixture.detectChanges();
    expect(
      fixture.nativeElement.querySelector('[data-testid="config-input-frame_stride"]'),
    ).toBeTruthy();
  });
});
