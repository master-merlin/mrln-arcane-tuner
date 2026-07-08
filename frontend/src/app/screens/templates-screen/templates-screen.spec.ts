import { describe, it, expect, vi } from 'vitest';
import { of, Subject } from 'rxjs';
import { TemplatesScreen } from './templates-screen';

/** Minimal writable-signal stub that actually stores the last set value. */
function writable<T>(init: T): (() => T) & { set: (v: T) => void } {
  let v = init;
  const fn = (() => v) as (() => T) & { set: (x: T) => void };
  fn.set = (x: T) => { v = x; };
  return fn;
}

function callLoad(ctx: Record<string, unknown>): Promise<void> {
  return (TemplatesScreen.prototype as unknown as Record<string, (...a: unknown[]) => Promise<void>>)['load']
    .call(ctx) as Promise<void>;
}

const flush = () => new Promise(r => setTimeout(r, 0));

describe('TemplatesScreen.load — parallel per-project fetch (P11)', () => {
  it('dispatches every project fetch concurrently, before any resolves, and keeps partial results', async () => {
    const projects = [{ id: 'p1', name: 'P1' }, { id: 'p2', name: 'P2' }, { id: 'p3', name: 'P3' }];

    // Per-project captioning calls resolve only when WE emit — so if the loop
    // still awaited sequentially, p2/p3 would never be dispatched.
    const capSubjects: Record<string, Subject<unknown[]>> = {
      p1: new Subject(), p2: new Subject(), p3: new Subject(),
    };

    const listCaptioningTemplates = vi.fn((_m: unknown, pid: string | null) =>
      pid == null ? of([]) : capSubjects[pid]);
    const listMaskingTemplates = vi.fn(() => of([]));
    const listTrainingTemplates = vi.fn(() => of([]));

    const loading = writable(false);
    const rows = writable<unknown[]>([]);

    const ctx = {
      loading,
      rows,
      projects: {
        listProjects: () => of(projects),
        allProjects: () => projects,
      },
      templates: { listCaptioningTemplates, listMaskingTemplates, listTrainingTemplates },
      toast: { warning: vi.fn(), error: vi.fn() },
      msg: () => 'err',
    };

    const done = callLoad(ctx);

    // Let the global calls settle and the per-project loop dispatch.
    await flush();

    // CONCURRENCY: all three projects' captioning calls fired even though NONE
    // of their observables has emitted yet. Sequential awaiting would stall
    // after p1 (2 project ids missing).
    const projectPids = listCaptioningTemplates.mock.calls
      .map(c => c[1]).filter(pid => pid != null);
    expect(projectPids).toEqual(expect.arrayContaining(['p1', 'p2', 'p3']));
    expect(loading()).toBe(true); // still in flight

    // p2 fails; p1 + p3 succeed with a project-scoped template each.
    capSubjects['p1'].next([{ id: 'c1', project_id: 'p1', name: 'A' }]);
    capSubjects['p1'].complete();
    capSubjects['p3'].next([{ id: 'c3', project_id: 'p3', name: 'C' }]);
    capSubjects['p3'].complete();
    capSubjects['p2'].error(new Error('nope'));

    await done;

    // Partial results survive the one failure.
    const ids = rows().map(r => (r as { tpl: { id: string } }).tpl.id);
    expect(ids).toContain('c1');
    expect(ids).toContain('c3');
    expect(ctx.toast.warning).toHaveBeenCalledWith(expect.stringContaining('P2'));
    expect(loading()).toBe(false);
  });
});

const T = (over: Partial<{ id: string; name: string; project_id: string|null; is_default: boolean; readonly: boolean; definition_id: string; model_id: string }>) => ({
  id: 'x', name: 'n', project_id: null, is_default: false, readonly: false, config: {}, created_at: 0, updated_at: 0, used_count: 0, ...over,
});
const R = (domain: 'training'|'captioning'|'masking', scopeId: string|null, over = {}) =>
  ({ domain, scopeId, scopeLabel: scopeId ?? 'General', tpl: T({ ...over }) as never });

describe('TemplatesScreen.filterRows', () => {
  const rows = [
    R('training', null, { name: 'Anime', definition_id: 'flux-dev' }),
    R('captioning', 'p1', { name: 'Caps', model_id: 'qwen3-vl', is_default: true }),
    R('masking', 'p1', { name: 'Mask', model_id: 'sam3', readonly: true }),
  ];
  it('filters by domain', () => {
    expect(TemplatesScreen.filterRows(rows, { domain: 'training', scope: 'all', search: '', flag: 'all' })).toHaveLength(1);
  });
  it('filters by scope general vs project', () => {
    expect(TemplatesScreen.filterRows(rows, { domain: 'all', scope: 'general', search: '', flag: 'all' })).toHaveLength(1);
    expect(TemplatesScreen.filterRows(rows, { domain: 'all', scope: 'p1', search: '', flag: 'all' })).toHaveLength(2);
  });
  it('search matches name, definition_id and model_id', () => {
    expect(TemplatesScreen.filterRows(rows, { domain: 'all', scope: 'all', search: 'anime', flag: 'all' })).toHaveLength(1); // name
    expect(TemplatesScreen.filterRows(rows, { domain: 'all', scope: 'all', search: 'flux', flag: 'all' })).toHaveLength(1); // definition_id
    expect(TemplatesScreen.filterRows(rows, { domain: 'all', scope: 'all', search: 'qwen', flag: 'all' })).toHaveLength(1); // model_id
  });
  it('filters by default/system flag', () => {
    expect(TemplatesScreen.filterRows(rows, { domain: 'all', scope: 'all', search: '', flag: 'default' })).toHaveLength(1);
    expect(TemplatesScreen.filterRows(rows, { domain: 'all', scope: 'all', search: '', flag: 'system' })).toHaveLength(1);
  });
});

describe('TemplatesScreen.remove — themed confirm', () => {
  it('opens the destructive confirm modal and only deletes on confirm', () => {
    const deleteTemplate = vi.fn().mockReturnValue(of({}));
    const openModal = vi.fn();
    const ctx = {
      overlay: { openModal },
      templates: { deleteTemplate },
      toast: { success: vi.fn(), error: vi.fn() },
      load: vi.fn().mockResolvedValue(undefined),
      msg: () => 'err',
    };
    const r = { domain: 'training', tpl: { id: 't1', name: 'Tpl', readonly: false } };

    (TemplatesScreen.prototype as unknown as Record<string, (...a: unknown[]) => unknown>)['remove']
      .call(ctx, r);

    expect(openModal).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
    expect(deleteTemplate).not.toHaveBeenCalled();

    const data = openModal.mock.calls.at(-1)![1] as { onConfirm?: () => void };
    data.onConfirm!();
    expect(deleteTemplate).toHaveBeenCalledWith('training', 't1');
  });
});
