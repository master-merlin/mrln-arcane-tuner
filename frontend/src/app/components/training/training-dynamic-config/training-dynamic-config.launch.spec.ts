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
import { ModelService } from '../../../services/model.service';
import { RegistryStore } from '../../../state/registry.store';
import { ModelCapabilitiesService } from '../../../services/model-capabilities.service';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { TemplateService } from '../../../services/template.service';
import { ProjectService } from '../../../services/project.service';
import { OverlayStore } from '../../../state/overlay.store';
import { FilesystemService } from '../../../services/filesystem.service';
import type { SchemaNode } from '../schema-node';

/**
 * T10 / T11 — training launch UX.
 *
 * T10 surfaces WHY the launch is blocked: an `invalidSectionCount` derived from
 * the current form state, an ordered `invalidSegments` list (DOM order), and a
 * `jumpToFirstInvalid()` action that expands (and scrolls to) the first section
 * that needs attention.
 *
 * T11 pins the launch + a compact echo of the key estimate (peak VRAM + wall
 * time) into a sticky footer so it stays reachable while scrolling. The echo
 * reuses the existing estimate signals (never recomputes).
 *
 * A required field in a normally-collapsed group (`ema_decay` in ENGINE /
 * "Advanced Engine") lets us prove the jump both TARGETS and EXPANDS the first
 * invalid group even when its errors are hidden inside a collapsed card.
 */
const SCHEMA: SchemaNode = {
  type: 'object',
  required: ['lora_name', 'network_rank', 'ema_decay'],
  properties: {
    definition_id: { type: 'string', default: 'flux-dev', group: 'MODEL_SELECTION' },
    model_family: { type: 'string', default: 'flux', group: 'MODEL_SELECTION' },
    // BASE → "General Settings". Required + empty default ⇒ invalid at build.
    lora_name: { type: 'string', default: '', group: 'BASE' },
    // NETWORK → "LoRA Parameters". Required but has a valid default ⇒ valid.
    network_rank: { type: 'integer', default: 16, group: 'NETWORK' },
    network_alpha: { type: 'integer', default: 8, group: 'NETWORK' },
    // ENGINE → "Advanced Engine" (collapsed by default). Required + empty.
    ema_decay: { type: 'string', default: '', group: 'ENGINE' },
  },
} as unknown as SchemaNode;

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

/** Lightweight build with the template overridden to '' (logic-only specs). */
function buildLogic() {
  TestBed.configureTestingModule({ imports: [TrainingDynamicConfigComponent], providers: providers() });
  TestBed.overrideComponent(TrainingDynamicConfigComponent, { set: { template: '', imports: [] } });
  const fixture = TestBed.createComponent(TrainingDynamicConfigComponent);
  fixture.componentRef.setInput('schema', SCHEMA);
  fixture.detectChanges();
  return { fixture, comp: fixture.componentInstance as any };
}

/** Full build keeping the real template but stubbing every heavy child so the
 *  component's OWN launch footer renders (DOM assertions). */
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
  fixture.detectChanges();
  return { fixture, comp: fixture.componentInstance as any };
}

describe('TrainingDynamicConfig — launch UX (T10 invalid-section surfacing)', () => {
  beforeEach(() => TestBed.resetTestingModule());

  it('counts the sections that need attention from the live form state', () => {
    const { comp } = buildLogic();
    // At build: lora_name (General Settings) + ema_decay (Advanced Engine) are
    // required-and-empty; network_rank holds its valid default.
    expect(comp.invalidSectionCount()).toBe(2);
    const labels = comp.invalidSegments().map((s: { label: string }) => s.label);
    expect(labels).toContain('General Settings');
    expect(labels).toContain('Advanced Engine');
    expect(labels).not.toContain('LoRA Parameters');
  });

  it('drops the count as invalid fields are filled', () => {
    const { comp, fixture } = buildLogic();
    comp.form.get('lora_name').setValue('my_lora');
    comp.form.get('ema_decay').setValue('0.999');
    fixture.detectChanges();
    expect(comp.invalidSectionCount()).toBe(0);
    expect(comp.invalidSegments()).toEqual([]);
    expect(comp.form.valid).toBe(true);
  });

  it('reacts to a newly-invalidated section', () => {
    const { comp, fixture } = buildLogic();
    comp.form.get('lora_name').setValue('my_lora');
    comp.form.get('ema_decay').setValue('0.999');
    fixture.detectChanges();
    expect(comp.invalidSectionCount()).toBe(0);

    comp.form.get('network_rank').setValue(null); // required → invalid
    fixture.detectChanges();
    expect(comp.invalidSectionCount()).toBe(1);
    expect(comp.firstInvalidSegment().label).toBe('LoRA Parameters');
  });

  it('firstInvalidSegment follows DOM order (General Settings before Advanced Engine)', () => {
    const { comp } = buildLogic();
    expect(comp.firstInvalidSegment().label).toBe('General Settings');
    expect(comp.firstInvalidSegment().id).toBe('general');
  });
});

describe('TrainingDynamicConfig — launch UX (T10 jump to first invalid)', () => {
  beforeEach(() => TestBed.resetTestingModule());

  it('jumpToFirstInvalid expands the first invalid group even when collapsed', () => {
    const { comp, fixture } = buildLogic();
    // Fill General Settings so the FIRST invalid section is the collapsed
    // Advanced Engine group.
    comp.form.get('lora_name').setValue('my_lora');
    fixture.detectChanges();

    expect(comp.firstInvalidSegment().id).toBe('advanced');
    expect(comp.isGroupCollapsed('Advanced Engine')).toBe(true);

    comp.jumpToFirstInvalid();
    expect(comp.isGroupCollapsed('Advanced Engine')).toBe(false);
  });

  it('jumpToFirstInvalid is a safe no-op when the form is valid', () => {
    const { comp, fixture } = buildLogic();
    comp.form.get('lora_name').setValue('my_lora');
    comp.form.get('ema_decay').setValue('0.999');
    fixture.detectChanges();
    expect(comp.firstInvalidSegment()).toBeNull();
    expect(() => comp.jumpToFirstInvalid()).not.toThrow();
  });
});

describe('TrainingDynamicConfig — launch UX (T11 sticky launch + estimate echo)', () => {
  beforeEach(() => TestBed.resetTestingModule());

  it('echoes the existing estimate signals (peak VRAM + wall time) without recomputing', () => {
    const { comp } = buildLogic();
    comp.vramEstimation.vramReport.set({
      peak_mb: 20480, available_mb: 98304, total_mb: 98304, fits: true,
    } as any);
    comp.vramEstimation.estimate.set({
      wall_time: { display: '2h 30m', seconds: 9000, samples: 3, calibrated: true },
    } as any);

    expect(comp.stickyWallTime()).toBe('2h 30m');
    // 20480 MB / 1024 = 20.0 GB peak.
    expect(comp.stickyPeakVram()).toContain('20.0');
    expect(comp.stickyPeakVram()).toContain('96.0'); // 98304 / 1024
  });

  it('renders the sticky launch bar with the estimate echo and an enabled CTA when valid', () => {
    const { comp, fixture } = buildDom();
    comp.form.get('lora_name').setValue('my_lora');
    comp.form.get('ema_decay').setValue('0.999');
    comp.vramEstimation.vramReport.set({
      peak_mb: 20480, available_mb: 98304, total_mb: 98304, fits: true,
    } as any);
    comp.vramEstimation.estimate.set({
      wall_time: { display: '2h 30m', seconds: 9000, samples: 3, calibrated: true },
    } as any);
    fixture.detectChanges();

    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('[data-testid="launch-bar"]')).toBeTruthy();
    expect(el.querySelector('[data-testid="launch-wall-time"]')?.textContent).toContain('2h 30m');
    expect(el.querySelector('[data-testid="launch-peak-vram"]')?.textContent).toContain('20.0');

    const cta = el.querySelector('[data-testid="submit-config-btn"]') as HTMLButtonElement;
    expect(cta).toBeTruthy();
    expect(cta.disabled).toBe(false);
    // No "needs attention" affordance while valid.
    expect(el.querySelector('[data-testid="jump-to-invalid"]')).toBeFalsy();
  });

  it('keeps the CTA disabled-but-informative while invalid (count + jump affordance)', () => {
    const { fixture } = buildDom();
    const el: HTMLElement = fixture.nativeElement;

    const cta = el.querySelector('[data-testid="submit-config-btn"]') as HTMLButtonElement;
    expect(cta.disabled).toBe(true);

    const jump = el.querySelector('[data-testid="jump-to-invalid"]') as HTMLButtonElement;
    expect(jump).toBeTruthy();
    // Real, labelled button (keyboard-reachable even though the CTA is disabled).
    expect(jump.tagName).toBe('BUTTON');
    expect(el.querySelector('[data-testid="invalid-count"]')?.textContent).toContain('2 sections');
  });
});
