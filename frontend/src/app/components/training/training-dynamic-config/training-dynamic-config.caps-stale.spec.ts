import { TestBed } from '@angular/core/testing';
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
import type { ModelCapabilities } from '../../../services/model-capabilities.service';

/**
 * LIVE BUG (UAT): switching the model FAMILY dropdown left the `capabilities`
 * descriptor pointing at the PREVIOUS model. Root cause: `definition_id` carries
 * `depends_on: model_family` + `backend_map` (family → definition ids), so a
 * family change fires the dependent-dropdown cascade in buildForm(), which
 * re-selects the first valid definition with `{ emitEvent: false }` (to avoid
 * re-triggering the targeted-layers modal) and MANUALLY syncs the
 * `currentDefinitionId` signal — but never reloaded the capability descriptor.
 * With a stale IMAGE descriptor, every `is_video`-gated field reads unsupported
 * → the whole VIDEO group hides and onSubmit strips the video fields.
 *
 * Fix: drive `_loadFieldCapabilities` from the `currentDefinitionId` SIGNAL via
 * an effect, so capabilities follow the selected definition through EVERY path
 * (family cascade, definition change, jobs handoff, template apply, model-change
 * modal Keep/Reset) uniformly. Plus: order the VIDEO group with the other
 * training-shape sections and give it a proper display label.
 */

// Two families; the image definition is only valid under `image`, the video
// definition only under `ltx2` — mirrors the backend's injected `backend_map`.
const CAPS_SCHEMA: SchemaNode = {
  type: 'object',
  properties: {
    model_family: {
      type: 'string',
      default: 'image',
      group: 'MODEL_SELECTION',
      enum: ['image', 'ltx2'],
      enum_labels: ['IMAGE', 'LTX2'],
    },
    definition_id: {
      type: 'string',
      default: 'img-base',
      group: 'MODEL_SELECTION',
      depends_on: 'model_family',
      hide_unsupported: true,
      enum: ['img-base', 'ltx2-3-base'],
      enum_labels: ['Image Base', 'LTX2 Base'],
      backend_map: { image: ['img-base'], ltx2: ['ltx2-3-base'] },
    },
  },
} as unknown as SchemaNode;

// Group-ordering fixture: one field in each of STRATEGY / VIDEO / NETWORK.
const GROUP_SCHEMA: SchemaNode = {
  type: 'object',
  properties: {
    definition_id: { type: 'string', default: '' },
    model_family: { type: 'string', default: '' },
    max_train_steps: { type: 'integer', default: 1000, group: 'STRATEGY' },
    num_frames: { type: 'integer', default: 25, group: 'VIDEO' },
    network_dim: { type: 'integer', default: 32, group: 'NETWORK' },
  },
} as unknown as SchemaNode;

/** Per-definition descriptor: the ltx2 definition is a video model. */
function capsFor(id: string): ModelCapabilities {
  const video = id === 'ltx2-3-base';
  return {
    enriched: true,
    block_topology: [],
    lora_targetable_modules: [],
    trainable_layers: [],
    archetype: video ? 'video' : 'image',
    capabilities: {} as ModelCapabilities['capabilities'],
    field_visibility: { num_frames: { supported: video } },
    defaults: {},
  };
}

function build(schema: SchemaNode) {
  const getCapabilitiesSpy = vi.fn((id: string) => of(capsFor(id)));
  TestBed.configureTestingModule({
    imports: [TrainingDynamicConfigComponent],
    providers: [
      { provide: DatasetService, useValue: { listDatasets: () => of([]) } },
      { provide: DatasetStore, useValue: { entities: () => [] } },
      { provide: ToastService, useValue: { error: () => {}, success: () => {}, warning: () => {} } },
      { provide: SystemService, useValue: {} },
      { provide: JobService, useValue: { estimate: () => of(null) } },
      { provide: ConfigHelpService, useValue: { getConfigHelp: () => of({}) } },
      {
        provide: ModelService,
        useValue: {
          getGlobalSettings: () => of({ default_model_path: '' }),
          getCapabilities: () => of({ enriched: false, block_topology: [] }),
        },
      },
      { provide: FilesystemService, useValue: {} },
      { provide: RegistryStore, useValue: { loadFor: () => Promise.resolve(), byId: () => () => undefined } },
      { provide: ModelCapabilitiesService, useValue: { getCapabilities: getCapabilitiesSpy } },
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
  fixture.componentRef.setInput('schema', schema);
  fixture.detectChanges(); // runs the rebuild effect → buildForm() + caps effect
  const comp = fixture.componentInstance as any;
  return { fixture, comp, getCapabilitiesSpy };
}

describe('TrainingDynamicConfig — capabilities follow the selected definition', () => {
  beforeEach(() => TestBed.resetTestingModule());

  it('reloads capabilities for the NEW definition when the family dropdown changes (depends_on cascade / emitEvent:false)', () => {
    const { fixture, comp, getCapabilitiesSpy } = build(CAPS_SCHEMA);

    // Precondition: on load the image definition's descriptor is loaded.
    expect(comp.currentDefinitionId()).toBe('img-base');
    expect(comp.capabilities()?.archetype).toBe('image');
    getCapabilitiesSpy.mockClear();

    // User switches the FAMILY dropdown → the model_family control changes.
    // The dependent-dropdown cascade silently swaps definition_id (emitEvent:false).
    comp.form.get('model_family')!.setValue('ltx2');
    fixture.detectChanges(); // flush the currentDefinitionId effect

    expect(comp.form.get('definition_id')!.value).toBe('ltx2-3-base');
    expect(comp.currentDefinitionId()).toBe('ltx2-3-base');
    // The bug: NO caps fetch fired for the new definition, so the descriptor
    // stayed the stale IMAGE one and every VIDEO field read unsupported.
    expect(getCapabilitiesSpy).toHaveBeenCalledWith('ltx2-3-base');
    expect(comp.capabilities()?.archetype).toBe('video');
  });

  it('onModelChangeKeep reloads capabilities for the REVERTED definition', () => {
    const { fixture, comp, getCapabilitiesSpy } = build(CAPS_SCHEMA);

    // Simulate having switched to the ltx2 definition first.
    comp.form.get('definition_id')!.setValue('ltx2-3-base');
    fixture.detectChanges();
    expect(comp.currentDefinitionId()).toBe('ltx2-3-base');

    // The model-change modal snapshots the previous selection for "Keep".
    comp._previousModelFamily = 'image';
    comp._previousDefinitionId = 'img-base';
    getCapabilitiesSpy.mockClear();

    comp.onModelChangeKeep();
    fixture.detectChanges();

    expect(comp.form.get('definition_id')!.value).toBe('img-base');
    expect(comp.currentDefinitionId()).toBe('img-base');
    // Keep must restore the previous model's capability descriptor too, not
    // leave the ltx2 (video) descriptor stranded on an image model.
    expect(getCapabilitiesSpy).toHaveBeenCalledWith('img-base');
    expect(comp.capabilities()?.archetype).toBe('image');
  });
});

describe('TrainingDynamicConfig — VIDEO group ordering + label', () => {
  beforeEach(() => TestBed.resetTestingModule());

  it('places the VIDEO group right after STRATEGY and before NETWORK', () => {
    const { comp } = build(GROUP_SCHEMA);
    const names = comp.groups().map((g: { name: string }) => g.name);

    expect(names).toContain('Video Settings');
    const iStrategy = names.indexOf('Training Dynamics'); // STRATEGY label
    const iVideo = names.indexOf('Video Settings');
    const iNetwork = names.indexOf('LoRA Parameters'); // NETWORK label
    expect(iVideo).toBe(iStrategy + 1);
    expect(iVideo).toBeLessThan(iNetwork);
  });

  it('formatGroupName maps VIDEO to a human "Video Settings" label', () => {
    const { comp } = build(GROUP_SCHEMA);
    expect(comp.formatGroupName('VIDEO')).toBe('Video Settings');
  });
});
