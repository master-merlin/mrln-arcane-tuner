import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { FormControl } from '@angular/forms';

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
import { ConfirmModalComponent, type ConfirmModalData } from '../../../modals/confirm/confirm.component';
import type { ConfigHelpData } from '../../../modals/config-help/config-help.component';
import type { SchemaNode } from '../schema-node';

/**
 * P4a trigger specs — training-dynamic-config opens its (former inline)
 * modals through the modal-layer registry with the right ModalKind + payload:
 *   - '?' help icon → openHelpModal(key) → 'config-help' with resolved
 *     {title, tip, detailHtml} built from config_help.json data;
 *   - model source button → openSourceConfigModal() → 'model-source-config'
 *     with the definition context + onSaved callback;
 *   - model change with targeted layers → the generic 'confirm' with
 *     Keep/Switch handlers, where DISMISSAL (backdrop / Esc, i.e. closing
 *     without an explicit choice) is equivalent to "Keep Model & Layers":
 *     the form reverts to the previous model and the layers are untouched.
 */
const SCHEMA: SchemaNode = {
  type: 'object',
  properties: {
    definition_id: { type: 'string', default: '' },
    model_family: { type: 'string', default: '' },
  },
} as unknown as SchemaNode;

function build() {
  TestBed.configureTestingModule({
    imports: [TrainingDynamicConfigComponent, ConfirmModalComponent],
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
          getGlobalSettings: () => of({ default_model_path: 'D:\\Models' }),
          // Exercised by the definition_id valueChanges path (real setValue,
          // not the direct guard call) → AdvancedVramCardComponent.loadCapabilities().
          getCapabilities: () => of({ enriched: false, block_topology: [] }),
        },
      },
      { provide: FilesystemService, useValue: {} },
      {
        provide: RegistryStore,
        useValue: {
          // Exercised by the definition_id valueChanges path (real setValue,
          // not the direct guard call) → the _sourceOverrideEffect's
          // loadSourceOverride().
          loadFor: () => Promise.resolve(),
          byId: () => () => undefined,
        },
      },
      { provide: ModelCapabilitiesService, useValue: { getCapabilities: () => of(null) } },
      { provide: RuntimeConfigService, useValue: { apiUrl: '/api', mediaBaseUrl: '/media' } },
      { provide: TemplateService, useValue: { listTrainingTemplates: () => of([]) } },
      { provide: ProjectService, useValue: { getPreferences: () => of(null) } },
      // REAL OverlayStore — these specs assert against the actual modal stack.
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
  fixture.detectChanges(); // runs the effect → buildForm()
  const overlay = TestBed.inject(OverlayStore);
  return { fixture, comp: fixture.componentInstance as any, overlay };
}

describe('TrainingDynamicConfig — modal-registry triggers (P4a)', () => {
  beforeEach(() => TestBed.resetTestingModule());

  it("help icon opens ModalKind 'config-help' with the resolved payload", () => {
    const { comp, overlay } = build();
    comp.configHelp.set({
      learning_rate: { tip: 'How fast the model learns.', detail: 'Use **small** values like `1e-4`.' },
    });
    comp.openHelpModal('learning_rate');

    const top = overlay.topModal();
    expect(top?.kind).toBe('config-help');
    const data = top?.data as ConfigHelpData;
    // Title derived from the key (schema fixture carries no title).
    expect(data.title).toBe('Learning Rate');
    expect(data.tip).toBe('How fast the model learns.');
    // detailHtml is the markdown-lite render of `detail`.
    expect(data.detailHtml).toContain('<strong>small</strong>');
    expect(data.detailHtml).toContain('<code>1e-4</code>');
  });

  it("model-source button opens ModalKind 'model-source-config' with the definition context", () => {
    const { comp, overlay } = build();
    comp.currentDefinitionId.set('flux-dev');
    comp.openSourceConfigModal();

    const top = overlay.topModal();
    expect(top?.kind).toBe('model-source-config');
    const data = top?.data as { definitionId: string; initialBrowsePath: string; onSaved: unknown };
    expect(data.definitionId).toBe('flux-dev');
    expect(data.initialBrowsePath).toBe('D:\\Models');
    expect(typeof data.onSaved).toBe('function');
  });

  it("model change with targeted layers opens 'confirm' with Keep/Switch wiring", () => {
    const { comp, overlay } = build();
    comp.form.addControl('targeted_layers', new FormControl<string[]>(['blocks.0.attn']));
    comp._checkTargetLayersOnModelChange('oldFam', 'old-def');

    const top = overlay.topModal();
    expect(top?.kind).toBe('confirm');
    const data = top?.data as ConfirmModalData;
    expect(data.title).toBe('Model Changed');
    expect(data.cancelLabel).toBe('Keep Model & Layers');
    expect(data.confirmLabel).toBe('Switch & Reset');
    expect(typeof data.onConfirm).toBe('function');
    expect(typeof data.onCancel).toBe('function');

    // Confirm path resets the layer selection (switch & reset).
    data.onConfirm!();
    expect(comp.form.get('targeted_layers')!.value).toEqual([]);
  });

  it('DISMISSING the model-change confirm (backdrop/Esc) is equivalent to Keep', () => {
    const { comp, overlay } = build();
    comp.form.addControl('targeted_layers', new FormControl<string[]>(['blocks.0.attn']));

    // Simulate: the form already switched to the NEW model when the guard runs
    // (that's the real call site — the valueChanges handler fires after the
    // new value landed, passing the OLD family/def as arguments).
    comp.form.get('model_family')!.setValue('newFam', { emitEvent: false });
    comp.form.get('definition_id')!.setValue('new-def', { emitEvent: false });
    comp._checkTargetLayersOnModelChange('oldFam', 'old-def');
    expect(overlay.topModal()?.kind).toBe('confirm');

    // Mount the real confirm modal (as modal-layer would), then dismiss it the
    // way the backdrop click / global Esc shortcut do: closeModal() directly,
    // followed by component destruction — NO explicit choice.
    const confirmFixture = TestBed.createComponent(ConfirmModalComponent);
    confirmFixture.detectChanges();
    overlay.closeModal();
    confirmFixture.destroy();

    // Keep-path ran: model reverted to the previous selection…
    expect(comp.form.get('model_family')!.value).toBe('oldFam');
    expect(comp.form.get('definition_id')!.value).toBe('old-def');
    expect(comp.currentDefinitionId()).toBe('old-def');
    expect(comp.selectedFamily()).toBe('oldFam');
    // …and the targeted layers were NOT wiped.
    expect(comp.form.get('targeted_layers')!.value).toEqual(['blocks.0.attn']);
    expect(overlay.modalStack().length).toBe(0);
  });

  it('DISMISSING the model-change confirm is equivalent to Keep when triggered via the REAL definition_id valueChanges path (not the guard method directly)', () => {
    const { comp, overlay } = build();
    comp.form.addControl('targeted_layers', new FormControl<string[]>(['blocks.0.attn']));

    // Seed the "previously known" model state the component tracks internally
    // (normally populated the first time the rebuild effect runs against a
    // real definition_id) — this is setup, not the trigger under test.
    comp._lastKnownModelFamily = 'oldFam';
    comp._lastKnownDefinitionId = 'old-def';
    // The stubbed TrainingTemplateSelectorComponent still runs its REAL
    // ngOnInit (only its template is overridden), which auto-applies on load
    // and latches `_isTemplateApplying` true for 1500ms (onTemplateApplied's
    // setTimeout release) — clear it so it doesn't mask the guard below.
    comp._isTemplateApplying = false;

    // Trigger via the PRODUCTION path: a real `setValue` on the `definition_id`
    // control fires its `valueChanges` subscription, which itself calls
    // `_checkTargetLayersOnModelChange` — the guard is never invoked directly.
    comp.form.get('definition_id')!.setValue('new-def');
    expect(overlay.topModal()?.kind).toBe('confirm');

    // Mount the real confirm modal (as modal-layer would), then dismiss it the
    // way the backdrop click / global Esc shortcut do: closeModal() directly,
    // followed by component destruction — NO explicit choice.
    const confirmFixture = TestBed.createComponent(ConfirmModalComponent);
    confirmFixture.detectChanges();
    overlay.closeModal();
    confirmFixture.destroy();

    // Keep-path ran: model reverted to the previous selection…
    expect(comp.form.get('model_family')!.value).toBe('oldFam');
    expect(comp.form.get('definition_id')!.value).toBe('old-def');
    expect(comp.currentDefinitionId()).toBe('old-def');
    expect(comp.selectedFamily()).toBe('oldFam');
    // …and the targeted layers were NOT wiped.
    expect(comp.form.get('targeted_layers')!.value).toEqual(['blocks.0.attn']);
    expect(overlay.modalStack().length).toBe(0);
  });
});
