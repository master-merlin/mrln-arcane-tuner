import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { TrainingDynamicConfigComponent } from './training-dynamic-config';
import { TrainingTemplateSelectorComponent } from '../training-template-selector/training-template-selector';
import { VramBudgetCardComponent } from '../vram-budget-card/vram-budget-card';
import { AdvancedVramCardComponent } from '../advanced-vram-card/advanced-vram-card';
import { TargetLayersCardComponent } from '../target-layers-card/target-layers-card';
import { DynamicFormGroupComponent } from '../dynamic-form-group/dynamic-form-group';
import { DynamicFormFieldComponent } from '../dynamic-form-field/dynamic-form-field';
import { DatasetService } from '../../../services/dataset';
import { DatasetStore } from '../../../state/dataset.store';
import { ToastService } from '../../../services/toast';
import { SystemService } from '../../../services/system.service';
import { JobService } from '../../../services/job';
import { ConfigHelpService } from '../../../services/config-help.service';
import { ModelService, type ModelDefinition } from '../../../services/model.service';
import { RegistryStore } from '../../../state/registry.store';
import { ModelCapabilitiesService } from '../../../services/model-capabilities.service';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { TemplateService } from '../../../services/template.service';
import { ProjectService } from '../../../services/project.service';
import { OverlayStore } from '../../../state/overlay.store';
import { FilesystemService } from '../../../services/filesystem.service';
import type { SchemaNode } from '../schema-node';

/**
 * Task 4 (LANE-132) — the model picker's non-commercial licence notice.
 *
 * RULE-21: the licence text has ONE producer, `ModelDefinition.license`
 * (`backend/app/engine/core/definitions.py`), serialized straight into
 * `availableModels()`. This component projects it onto the `definition_id`
 * schema node's `license_map` (`organizeGroups`) so the notice renders
 * from the SAME text the definition carries — never a second copy.
 */
const SCHEMA: SchemaNode = {
  type: 'object',
  properties: {
    definition_id: {
      type: 'string',
      title: 'Model Definition',
      group: 'MODEL_SELECTION',
      enum: ['flux-dev', 'qwen-image-2512', 'qwen-image-2.1'],
      enum_labels: {
        'flux-dev': 'FLUX.1 Dev',
        'qwen-image-2512': 'Qwen-Image-2512',
        'qwen-image-2.1': 'Qwen-Image-2.1',
      },
      default: 'flux-dev',
    },
    model_family: { type: 'string', default: 'flux', group: 'MODEL_SELECTION' },
  },
} as unknown as SchemaNode;

const MODELS: ModelDefinition[] = [
  { id: 'flux-dev', name: 'FLUX.1 Dev', family: 'flux' },
  { id: 'qwen-image-2512', name: 'Qwen-Image-2512', family: 'qwen_image' },
  { id: 'qwen-image-2.1', name: 'Qwen-Image-2.1', family: 'qwen_image2', license: 'qwen-research (non-commercial)' },
];

function providers() {
  return [
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
    { provide: ModelCapabilitiesService, useValue: { getCapabilities: () => of(null) } },
    { provide: RuntimeConfigService, useValue: { apiUrl: '/api', mediaBaseUrl: '/media' } },
    { provide: TemplateService, useValue: { listTrainingTemplates: () => of([]) } },
    { provide: ProjectService, useValue: { getPreferences: () => of(null) } },
  ];
}

/** Full build keeping the component's OWN template (the MODEL_SELECTION
 *  section is hardcoded there, not delegated to a child) while stubbing
 *  every heavy child component. */
function buildDom() {
  TestBed.configureTestingModule({ imports: [TrainingDynamicConfigComponent], providers: providers() });
  const stub = { template: '', imports: [] };
  TestBed.overrideComponent(TrainingTemplateSelectorComponent, { set: stub });
  TestBed.overrideComponent(VramBudgetCardComponent, { set: stub });
  TestBed.overrideComponent(AdvancedVramCardComponent, { set: stub });
  TestBed.overrideComponent(TargetLayersCardComponent, { set: stub });
  TestBed.overrideComponent(DynamicFormGroupComponent, { set: stub });
  TestBed.overrideComponent(DynamicFormFieldComponent, { set: stub });
  const fixture = TestBed.createComponent(TrainingDynamicConfigComponent);
  fixture.componentRef.setInput('schema', SCHEMA);
  fixture.componentRef.setInput('availableModels', MODELS);
  fixture.detectChanges();
  return { fixture, comp: fixture.componentInstance as any };
}

describe('TrainingDynamicConfig — model picker licence notice', () => {
  beforeEach(() => TestBed.resetTestingModule());

  it('shows no notice for the default selection (flux-dev, no licence)', () => {
    const { fixture } = buildDom();
    const notice = fixture.nativeElement.querySelector('[data-testid="model-license-notice"]');
    expect(notice).toBeNull();
  });

  it('shows the notice, carrying the definition\'s own licence text, once qwen-image-2.1 is selected', () => {
    const { fixture, comp } = buildDom();
    comp.form.get('definition_id')!.setValue('qwen-image-2.1');
    fixture.detectChanges();

    const notice = fixture.nativeElement.querySelector('[data-testid="model-license-notice"]');
    expect(notice).not.toBeNull();
    expect(notice.textContent).toContain('non-commercial');
    expect(notice.textContent).toContain('qwen-research (non-commercial)');
  });

  it('shows no notice for qwen-image-2512, which declares no licence', () => {
    const { fixture, comp } = buildDom();
    comp.form.get('definition_id')!.setValue('qwen-image-2512');
    fixture.detectChanges();

    const notice = fixture.nativeElement.querySelector('[data-testid="model-license-notice"]');
    expect(notice).toBeNull();
  });
});
