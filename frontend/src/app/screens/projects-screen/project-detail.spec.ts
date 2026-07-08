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
