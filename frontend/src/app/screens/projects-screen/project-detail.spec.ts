import { describe, it, expect, vi } from 'vitest';
import { of } from 'rxjs';
import { ProjectDetail } from './project-detail';

interface ConfirmData {
  destructive?: boolean;
  onConfirm?: () => void | Promise<void>;
}

function invoke(method: string, ctx: Record<string, unknown>, ...args: unknown[]): unknown {
  const proto = ProjectDetail.prototype as unknown as Record<string, (...a: unknown[]) => unknown>;
  return proto[method].apply(ctx, args);
}

function signalStub<T>(value: T): (() => T) & { set: (v: T) => void } {
  return Object.assign(() => value, { set: vi.fn() }) as (() => T) & { set: (v: T) => void };
}

describe('ProjectDetail.resolveState — loading vs not-found gate (P4)', () => {
  it('is "loading" before the first project load resolves (no premature not-found)', () => {
    expect(ProjectDetail.resolveState({ hasId: true, loaded: false, loading: true, hasProject: false })).toBe('loading');
    // Even if a stale `loaded` is true, an in-flight refresh with no match yet reads as loading.
    expect(ProjectDetail.resolveState({ hasId: true, loaded: true, loading: true, hasProject: false })).toBe('loading');
  });

  it('is "ready" as soon as the project resolves, regardless of load flags', () => {
    expect(ProjectDetail.resolveState({ hasId: true, loaded: false, loading: true, hasProject: true })).toBe('ready');
    expect(ProjectDetail.resolveState({ hasId: true, loaded: true, loading: false, hasProject: true })).toBe('ready');
  });

  it('is "not-found" ONLY after load resolves and the id is genuinely absent', () => {
    expect(ProjectDetail.resolveState({ hasId: true, loaded: true, loading: false, hasProject: false })).toBe('not-found');
  });

  it('stays "loading" while the route id has not been read yet', () => {
    expect(ProjectDetail.resolveState({ hasId: false, loaded: true, loading: false, hasProject: false })).toBe('loading');
  });
});

describe('ProjectDetail — themed confirm migrations', () => {
  it('removeAllDatasets opens the destructive confirm and only removes on confirm', async () => {
    const removeProjectDataset = vi.fn().mockReturnValue(of({}));
    const openModal = vi.fn();
    const ctx = {
      projectId: () => 'proj1',
      projectDatasets: () => [{ id: 'd1', name: 'A' }, { id: 'd2', name: 'B' }],
      removingAll: signalStub(false),
      overlay: { openModal },
      projects: { removeProjectDataset, loadProjects: vi.fn() },
      toast: { success: vi.fn(), warning: vi.fn(), error: vi.fn() },
      loadDatasets: vi.fn().mockResolvedValue(undefined),
    };

    await invoke('removeAllDatasets', ctx);

    expect(openModal).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
    expect(removeProjectDataset).not.toHaveBeenCalled();

    const data = openModal.mock.calls.at(-1)![1] as ConfirmData;
    await data.onConfirm!();
    expect(removeProjectDataset).toHaveBeenCalledTimes(2);
  });

  it('removeDatasetFromProject opens the destructive confirm and only removes on confirm', async () => {
    const removeProjectDataset = vi.fn().mockReturnValue(of({}));
    const openModal = vi.fn();
    const ctx = {
      projectId: () => 'proj1',
      overlay: { openModal },
      projects: { removeProjectDataset, loadProjects: vi.fn() },
      toast: { success: vi.fn(), error: vi.fn() },
      loadDatasets: vi.fn().mockResolvedValue(undefined),
    };

    await invoke('removeDatasetFromProject', ctx, { id: 'd1', name: 'A' }, { stopPropagation: vi.fn() });

    expect(openModal).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
    expect(removeProjectDataset).not.toHaveBeenCalled();

    const data = openModal.mock.calls.at(-1)![1] as ConfirmData;
    await data.onConfirm!();
    expect(removeProjectDataset).toHaveBeenCalledWith('proj1', 'd1');
  });

  it('deleteProjectTemplate opens the destructive confirm and only deletes on confirm', async () => {
    const deleteTemplate = vi.fn().mockReturnValue(of({}));
    const openModal = vi.fn();
    const ctx = {
      projectId: () => 'proj1',
      overlay: { openModal },
      templates: { deleteTemplate },
      projects: { loadProjects: vi.fn() },
      toast: { success: vi.fn(), error: vi.fn() },
      loadTemplates: vi.fn().mockResolvedValue(undefined),
    };

    await invoke('deleteProjectTemplate', ctx, 'training', { id: 't1', name: 'Tpl' });

    expect(openModal).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
    expect(deleteTemplate).not.toHaveBeenCalled();

    const data = openModal.mock.calls.at(-1)![1] as ConfirmData;
    await data.onConfirm!();
    expect(deleteTemplate).toHaveBeenCalledWith('training', 't1');
  });

  it('editTemplate: training hands off to /training (no modal); caption/mask open the edit modal', () => {
    const set = vi.fn();
    const navigate = vi.fn();
    const openModal = vi.fn();
    const scope = { setProject: vi.fn(), setGlobal: vi.fn() };

    const trainingCtx = {
      templates: { getTemplate: () => of({ id: 't1', name: 'T', config: { a: 1 }, definition_id: 'flux', project_id: 'p1' }) },
      scope, handoff: { set }, router: { navigate }, overlay: { openModal },
      toast: { error: vi.fn() },
      projectId: () => 'p1',
      loadTemplates: vi.fn(),
      openTrainingTemplate: (ProjectDetail.prototype as unknown as Record<string, (...a: unknown[]) => unknown>)['openTrainingTemplate'],
    };
    invoke('editTemplate', trainingCtx, 'training', { id: 't1' });
    expect(set).toHaveBeenCalledTimes(1);
    expect(navigate).toHaveBeenCalledWith(['/training']);
    expect(openModal).not.toHaveBeenCalled();

    openModal.mockClear();
    const captionCtx = {
      templates: { getTemplate: () => of({ id: 'c1', name: 'C', config: {}, project_id: null }) },
      scope, handoff: { set }, router: { navigate }, overlay: { openModal },
      toast: { error: vi.fn() },
      projectId: () => 'p1',
      loadTemplates: vi.fn(),
    };
    invoke('editTemplate', captionCtx, 'captioning', { id: 'c1' });
    expect(openModal).toHaveBeenCalledWith('template-edit', expect.objectContaining({ domain: 'captioning' }));
  });

  it('deleteProject opens the destructive confirm and only deletes on confirm', () => {
    const deleteProject = vi.fn().mockReturnValue(of({}));
    const openModal = vi.fn();
    const ctx = {
      project: () => ({ id: 'p1', name: 'Demo' }),
      overlay: { openModal },
      projects: { deleteProject, loadProjects: vi.fn() },
      toast: { success: vi.fn(), error: vi.fn() },
      scope: { projectId: () => null, setGlobal: vi.fn() },
      router: { navigate: vi.fn() },
    };

    invoke('deleteProject', ctx);

    expect(openModal).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
    expect(deleteProject).not.toHaveBeenCalled();

    const data = openModal.mock.calls.at(-1)![1] as ConfirmData;
    (data.onConfirm as () => void)!();
    expect(deleteProject).toHaveBeenCalledWith('p1');
  });
});

describe('ProjectDetail — P5 tab order + guided Quick Train', () => {
  it('orders the sub-tabs Overview · Datasets · Templates · Quick Train · Runs (dataset → template → train)', () => {
    expect(ProjectDetail.TABS.map(t => t.value)).toEqual([
      'overview', 'datasets', 'templates', 'quick-train', 'runs',
    ]);
  });

  it('inline "jump to tab" action switches the active tab in place, WITHOUT navigating away', () => {
    const navigate = vi.fn();
    const ctx = {
      tab: signalStub<string>('quick-train'),
      projectId: () => 'p1',
      loadDatasets: vi.fn().mockResolvedValue(undefined),
      loadTemplates: vi.fn().mockResolvedValue(undefined),
      ensureDatasetsSchema: vi.fn(),
      router: { navigate },
    };

    invoke('onTabChange', ctx, 'datasets');
    expect(ctx.tab.set).toHaveBeenCalledWith('datasets');
    expect(navigate).not.toHaveBeenCalled();

    invoke('onTabChange', ctx, 'templates');
    expect(ctx.tab.set).toHaveBeenCalledWith('templates');
    expect(navigate).not.toHaveBeenCalled();
  });
});

describe('ProjectDetail — P6 "Full configuration" handoff', () => {
  it('sets the training handoff BEFORE navigating, carrying template + resolved name + dataset rows', () => {
    const order: string[] = [];
    const set = vi.fn((_h?: unknown) => { order.push('handoff'); });
    const navigate = vi.fn((_c?: unknown) => { order.push('navigate'); return Promise.resolve(true); });
    const setProject = vi.fn();

    const ctx = {
      projectId: () => 'p1',
      selectedTemplate: () => ({ id: 't1', name: 'My Tpl', definition_id: 'flux-dev', config: { max_train_steps: 1000 } }),
      loraPrefix: () => 'pre',
      loraSuffix: () => 'suf',
      triggerWord: () => 'ohwx',
      loraName: () => '{lora_prefix}_{definition_id}',
      launchForm: { get: () => ({ value: [{ dataset_name: 'ds1', num_repeats: 5 }, { dataset_name: '' }] }) },
      scope: { setProject, setGlobal: vi.fn() },
      handoff: { set },
      router: { navigate },
    };

    invoke('openFullConfiguration', ctx);

    expect(set).toHaveBeenCalledTimes(1);
    const arg = set.mock.calls[0][0] as { mode: string; templateId?: string; config: Record<string, unknown> };
    expect(arg.templateId).toBe('t1');
    expect(arg.mode).toBe('template');
    // {placeholders} resolved against the merged config, like startQuickTrain.
    expect(arg.config['lora_name']).toBe('pre_flux-dev');
    // Only rows that actually selected a dataset survive.
    expect(arg.config['datasets']).toEqual([{ dataset_name: 'ds1', num_repeats: 5 }]);
    expect(arg.config['project_id']).toBe('p1');
    expect(arg.config['max_train_steps']).toBe(1000);
    // Handoff MUST be set before the route change so /training continues setup.
    expect(order).toEqual(['handoff', 'navigate']);
    expect(navigate).toHaveBeenCalledWith(['/training']);
  });
});

describe('ProjectDetail — P7 training-template edit affordance', () => {
  it('uses a distinct "leaves the project" icon for training vs the in-place pencil for caption/mask', () => {
    const editIcon = (ProjectDetail.prototype as unknown as Record<string, (d: string) => string>)['editIcon'];
    expect(editIcon('training')).toBe('ExternalLink');
    expect(editIcon('captioning')).toBe('Pencil');
    expect(editIcon('masking')).toBe('Pencil');
  });
});

describe('ProjectDetail — adaptive presets are a first-class project domain', () => {
  it('loadTemplates lists the project\'s adaptive presets alongside the other three', async () => {
    const sections = signalStub<unknown[]>([]);
    const globals = () => signalStub<unknown[]>([]);
    const row = (id: string, project_id: string | null) => ({ id, project_id });
    const ctx = {
      templates: {
        listCaptioningTemplates: vi.fn().mockReturnValue(of([row('c1', 'p1')])),
        listMaskingTemplates: vi.fn().mockReturnValue(of([row('m1', 'p1')])),
        listTrainingTemplates: vi.fn().mockReturnValue(of([row('t1', 'p1')])),
        // Global rows are the readonly factory presets; only 'a1' is the project's.
        listAdaptivePresets: vi.fn().mockReturnValue(of([row('a1', 'p1'), row('fac', null)])),
      },
      templateSections: sections,
      globalCaptionTpls: globals(), globalMaskTpls: globals(),
      globalTrainTpls: globals(), globalAdaptiveTpls: globals(),
    };
    await invoke('loadTemplates', ctx, 'p1');

    expect(ctx.templates.listAdaptivePresets).toHaveBeenCalledWith('p1');
    const setSpy = sections.set as unknown as ReturnType<typeof vi.fn>;
    const written = setSpy.mock.calls[0][0] as { domain: string; items: unknown[] }[];
    expect(written.map(s => s.domain))
      .toEqual(['captioning', 'masking', 'training', 'adaptive']);
    expect(written[3].items).toEqual([row('a1', 'p1')]);  // the global row is not the project's
  });

  it('edits an adaptive preset as JSON — the caption/mask dialog cannot render knobs', () => {
    const openModal = vi.fn();
    const full = { id: 'a1', name: 'Mine', config: { warmup_pct: 0.25 } };
    const ctx = {
      templates: { getTemplate: vi.fn().mockReturnValue(of(full)) },
      overlay: { openModal },
      projectId: () => 'p1',
      loadTemplates: vi.fn(),
      toast: { error: vi.fn() },
    };
    invoke('editTemplate', ctx, 'adaptive', { id: 'a1' });

    expect(openModal).toHaveBeenCalledWith('template-json',
      expect.objectContaining({ domain: 'adaptive', template: full }));
    expect(openModal).not.toHaveBeenCalledWith('template-edit', expect.anything());
    // One fetch, not two — the modal opens on the row already loaded here.
    expect(ctx.templates.getTemplate).toHaveBeenCalledTimes(1);
  });

  it('gives the adaptive row its own affordance so the pencil never lies', () => {
    const proto = ProjectDetail.prototype as unknown as Record<string, (...a: unknown[]) => string>;
    expect(proto['editIcon']('training')).toBe('ExternalLink');
    expect(proto['editIcon']('captioning')).toBe('Pencil');
    expect(proto['editIcon']('adaptive')).toBe('Braces');
    expect(proto['editTitle']('adaptive')).toContain('JSON');
    expect(proto['editAria']('adaptive', 'Mine')).toContain('Mine');
  });

  it('the TEMPLATES stat counts adaptive presets', () => {
    const proto = ProjectDetail.prototype as unknown as
      Record<string, (s: unknown) => number>;
    expect(proto['templateCount']({
      captioning_templates: 1, masking_templates: 2, training_templates: 4,
      adaptive_preset_templates: 3, datasets: 0, jobs: 0,
    })).toBe(10);
    expect(proto['templateCount'](undefined)).toBe(0);
  });
});
